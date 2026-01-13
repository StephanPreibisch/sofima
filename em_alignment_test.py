#!/usr/bin/env python
"""
SOFIMA EM Alignment - Test Version

Processes 20 sections to demonstrate the full pipeline including:
- Flow field computation and filtering visualization (section 14)
- Flow reconciliation visualization (section 14)
- Full stack output as 3D TIFF and ZARR
"""

from concurrent import futures
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorstore as ts
import tifffile
import zarr
from tqdm import tqdm

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_SECTIONS = 20  # Process 20 sections to include section 14
FLOW_VIZ_SECTION = 14  # Section to visualize flow fields


def save_figure(fig, name):
    """Save figure to the output directory."""
    path = os.path.join(OUTPUT_DIR, f"temp_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print(f"SOFIMA EM Alignment - Test ({NUM_SECTIONS} sections)")
    print("=" * 60)

    devices = jax.devices()
    print(f"\nJAX devices: {devices}")

    patch_size = 160
    stride = 40

    # Step 1: Load data
    print("\n[Step 1] Loading EM data from Google Cloud Storage...")

    unaligned_1x = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': 'gs://sofima-sample-data/fmi-friedrich-dp/subvol_5800_5500_6250',
        "scale_metadata": {"resolution": [11, 11, 25]},
        "context": {"cache_pool": {"total_bytes_limit": 1_000_000_000}},
    }).result()

    unaligned_2x = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': 'gs://sofima-sample-data/fmi-friedrich-dp/subvol_5800_5500_6250',
        "scale_metadata": {"resolution": [22, 22, 25]},
        "context": {"cache_pool": {"total_bytes_limit": 1_000_000_000}},
    }).result()

    print(f"Data shape (1x): {unaligned_1x.shape}")
    print(f"Processing first {NUM_SECTIONS} sections")

    # Step 2: Compute flow fields
    print("\n[Step 2] Computing optical flow fields...")

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()

    # 1x resolution
    flows1x_list = []
    prev = unaligned_1x[..., 0, 0].T.read().result()
    for z in tqdm(range(1, NUM_SECTIONS), desc="Flow 1x"):
        curr = unaligned_1x[..., z, 0].T.read().result()
        flows1x_list.append(mfc.flow_field(prev, curr, (patch_size, patch_size),
                                           (stride, stride), batch_size=256))
        prev = curr

    # 2x resolution
    flows2x_list = []
    prev = unaligned_2x[..., 0, 0].T.read().result()
    for z in tqdm(range(1, NUM_SECTIONS), desc="Flow 2x"):
        curr = unaligned_2x[..., z, 0].T.read().result()
        flows2x_list.append(mfc.flow_field(prev, curr, (patch_size, patch_size),
                                           (stride, stride), batch_size=256))
        prev = curr

    flows1x = np.array(flows1x_list)
    flows2x = np.array(flows2x_list)

    # Transpose and pad
    flows2x = np.transpose(flows2x, [1, 0, 2, 3])
    flows1x = np.transpose(flows1x, [1, 0, 2, 3])

    pad = patch_size // 2 // stride
    flows1x = np.pad(flows1x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
    flows2x = np.pad(flows2x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    print(f"Flow shape (1x): {flows1x.shape}")

    # Step 3: Clean flow fields
    print("\n[Step 3] Cleaning flow fields...")

    f1 = flow_utils.clean_flow(flows1x, min_peak_ratio=1.6, min_peak_sharpness=1.6,
                                max_magnitude=80, max_deviation=20)
    f2 = flow_utils.clean_flow(flows2x, min_peak_ratio=1.6, min_peak_sharpness=1.6,
                                max_magnitude=80, max_deviation=20)

    # === Flow visualization: before and after filtering (section 14) ===
    print(f"\n[Visualization] Flow filtering for section {FLOW_VIZ_SECTION}...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(flows1x[0, FLOW_VIZ_SECTION, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[0].set_title(f'Before Filtering (Section {FLOW_VIZ_SECTION})')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    plt.colorbar(im0, ax=ax[0], label='X displacement (pixels)')

    im1 = ax[1].imshow(f1[0, FLOW_VIZ_SECTION, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[1].set_title(f'After Filtering (Section {FLOW_VIZ_SECTION})')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    plt.colorbar(im1, ax=ax[1], label='X displacement (pixels)')

    fig.suptitle('Flow Vector Field - Before (left) and After (right) Filtering', fontsize=14)
    save_figure(fig, "05_flow_before_after_filtering")

    # Step 4: Resample and reconcile flows
    print("\n[Step 4] Resampling and reconciling flows...")

    f2_hires = np.zeros_like(f1)
    scale = 0.5

    box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1))
    box2x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1))

    for z in range(f2.shape[1]):
        resampled = map_utils.resample_map(f2[:, z:z + 1, ...], box2x, box1x, 1 / scale, 1)
        f2_hires[:, z:z + 1, ...] = resampled / scale

    final_flow = flow_utils.reconcile_flows((f1, f2_hires), max_gradient=0,
                                             max_deviation=20, min_patch_size=400)

    # === Flow visualization: high res, upsampled low res, combined (section 14) ===
    print(f"\n[Visualization] Flow reconciliation for section {FLOW_VIZ_SECTION}...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    im0 = ax[0].imshow(f1[0, FLOW_VIZ_SECTION, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[0].set_title(f'High Res. Flow (1x)')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    plt.colorbar(im0, ax=ax[0], label='X displacement')

    im1 = ax[1].imshow(f2_hires[0, FLOW_VIZ_SECTION, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[1].set_title(f'Upsampled Low Res. Flow (2x)')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    plt.colorbar(im1, ax=ax[1], label='X displacement')

    im2 = ax[2].imshow(final_flow[0, FLOW_VIZ_SECTION, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[2].set_title(f'Combined Flow (for alignment)')
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Y')
    plt.colorbar(im2, ax=ax[2], label='X displacement')

    fig.suptitle(f'Flow Reconciliation (Section {FLOW_VIZ_SECTION}): High Res., Upsampled Low Res., Combined', fontsize=14)
    save_figure(fig, "06_flow_reconciliation")

    # Step 5: Mesh relaxation
    print("\n[Step 5] Mesh relaxation...")

    config = mesh.IntegrationConfig(
        dt=0.001, gamma=0.0, k0=0.01, k=0.1,
        stride=(stride, stride), num_iters=1000,
        max_iters=100000, stop_v_max=0.005, dt_max=1000,
        start_cap=0.01, final_cap=10, prefer_orig_order=True
    )

    solved = [np.zeros_like(final_flow[:, 0:1, ...])]
    origin = jnp.array([0., 0.])

    for z in tqdm(range(0, final_flow.shape[1]), desc="Mesh relaxation"):
        prev = map_utils.compose_maps_fast(final_flow[:, z:z+1, ...], origin, stride,
                                           solved[-1], origin, stride)
        x = np.zeros_like(solved[0])
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
        x = np.array(x)
        solved.append(x)

    solved = np.concatenate(solved, axis=1)
    print(f"Solved mesh shape: {solved.shape}")

    # Step 6: Invert map
    print("\n[Step 6] Inverting coordinate map...")
    inv_map = map_utils.invert_map(solved, box1x, box1x, stride)
    print(f"Inverse map shape: {inv_map.shape}")

    # Step 7: Warp images and build full stacks
    print("\n[Step 7] Warping images and building stacks...")

    # Build unaligned stack
    unaligned_stack = []
    for z in tqdm(range(NUM_SECTIONS), desc="Loading unaligned"):
        slice_data = unaligned_1x[1000:2000, 2000:3000, z, 0].read().result()
        unaligned_stack.append(slice_data)
    unaligned_stack = np.array(unaligned_stack)  # Shape: [Z, X, Y]
    print(f"Unaligned stack shape: {unaligned_stack.shape}")

    # Build warped/aligned stack
    warped = [np.transpose(unaligned_1x[1000:2000, 2000:3000, 0:1, 0].read().result(), [2, 1, 0])]

    for z in tqdm(range(1, NUM_SECTIONS), desc="Warping"):
        data_box = bounding_box.BoundingBox(start=(500, 1500, 0), size=(2000, 2000, 1))
        out_box = bounding_box.BoundingBox(start=(1000, 2000, 0), size=(1000, 1000, 1))

        data = np.transpose(unaligned_1x[data_box.start[0]:data_box.end[0],
                                         data_box.start[1]:data_box.end[1],
                                         z:z+1, 0:1].read().result(), [3, 2, 1, 0])
        warped.append(
            warp.warp_subvolume(data, data_box, inv_map[:, z:z+1, ...], box1x, stride,
                               out_box, 'lanczos', parallelism=1)[0, ...])

    warped_xyz = np.transpose(np.concatenate(warped, axis=0), [2, 1, 0])
    # Convert to [Z, Y, X] for standard TIFF orientation
    aligned_stack = np.transpose(warped_xyz, [2, 1, 0])
    print(f"Aligned stack shape: {aligned_stack.shape}")

    # === Save XY slices (before/after for first and last section) ===
    print("\n[Step 8] Saving visualizations...")

    section_a = 0
    section_b = NUM_SECTIONS - 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(unaligned_stack[section_a].T, cmap='gray')
    ax.set_title(f'Section {section_a} - Before Alignment')
    ax.axis('off')
    save_figure(fig, f"01_section{section_a}_before")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(aligned_stack[section_a].T, cmap='gray')
    ax.set_title(f'Section {section_a} - After Alignment')
    ax.axis('off')
    save_figure(fig, f"02_section{section_a}_after")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(unaligned_stack[section_b].T, cmap='gray')
    ax.set_title(f'Section {section_b} - Before Alignment')
    ax.axis('off')
    save_figure(fig, f"03_section{section_b}_before")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(aligned_stack[section_b].T, cmap='gray')
    ax.set_title(f'Section {section_b} - After Alignment')
    ax.axis('off')
    save_figure(fig, f"04_section{section_b}_after")

    # === Save 3D TIFF files ===
    print("\n[Step 9] Saving 3D TIFF files...")

    tiff_before_path = os.path.join(OUTPUT_DIR, "temp_stack_before_alignment.tif")
    tiff_after_path = os.path.join(OUTPUT_DIR, "temp_stack_after_alignment.tif")

    # Ensure correct dtype for TIFF
    tifffile.imwrite(tiff_before_path, unaligned_stack.astype(np.uint8),
                     imagej=True, metadata={'axes': 'ZYX'})
    print(f"Saved: {tiff_before_path}")

    tifffile.imwrite(tiff_after_path, aligned_stack.astype(np.uint8),
                     imagej=True, metadata={'axes': 'ZYX'})
    print(f"Saved: {tiff_after_path}")

    # === Save ZARR files ===
    print("\n[Step 10] Saving ZARR files...")

    import shutil

    zarr_before_path = os.path.join(OUTPUT_DIR, "temp_stack_before_alignment.zarr")
    zarr_after_path = os.path.join(OUTPUT_DIR, "temp_stack_after_alignment.zarr")

    # Remove existing zarr directories if they exist
    if os.path.exists(zarr_before_path):
        shutil.rmtree(zarr_before_path)
    if os.path.exists(zarr_after_path):
        shutil.rmtree(zarr_after_path)

    zarr.save(zarr_before_path, unaligned_stack, zarr_format=2)
    print(f"Saved: {zarr_before_path}")

    zarr.save(zarr_after_path, aligned_stack, zarr_format=2)
    print(f"Saved: {zarr_after_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUCCESS! All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)
    print("\nGenerated files:")
    print("\n  PNG visualizations:")
    print(f"    - temp_01_section{section_a}_before.png")
    print(f"    - temp_02_section{section_a}_after.png")
    print(f"    - temp_03_section{section_b}_before.png")
    print(f"    - temp_04_section{section_b}_after.png")
    print(f"    - temp_05_flow_before_after_filtering.png")
    print(f"    - temp_06_flow_reconciliation.png")
    print("\n  3D TIFF stacks:")
    print(f"    - temp_stack_before_alignment.tif ({unaligned_stack.shape})")
    print(f"    - temp_stack_after_alignment.tif ({aligned_stack.shape})")
    print("\n  ZARR arrays:")
    print(f"    - temp_stack_before_alignment.zarr/")
    print(f"    - temp_stack_after_alignment.zarr/")


if __name__ == "__main__":
    main()
