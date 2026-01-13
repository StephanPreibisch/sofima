#!/usr/bin/env python
"""
SOFIMA EM Alignment Example

This script reproduces the EM sections alignment notebook from:
https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

It demonstrates:
1. Loading EM data from Google Cloud Storage via TensorStore
2. Computing optical flow between consecutive sections
3. Cleaning and reconciling flow fields at multiple resolutions
4. Mesh-based relaxation for regularized alignment
5. Inverting maps and warping images

All visualizations are saved as images in the current directory.
"""

from concurrent import futures
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorstore as ts
from tqdm import tqdm

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp

# Output directory for images
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_figure(fig, name):
    """Save figure to the output directory."""
    path = os.path.join(OUTPUT_DIR, f"temp_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def load_data():
    """Load the sample EM data at 1x and 2x resolution."""
    print("Loading unaligned data at 1x resolution...")
    unaligned_1x = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': 'gs://sofima-sample-data/fmi-friedrich-dp/subvol_5800_5500_6250',
        "scale_metadata": {"resolution": [11, 11, 25]},
        "context": {"cache_pool": {"total_bytes_limit": 1_000_000_000}},
    }).result()

    print("Loading unaligned data at 2x resolution...")
    unaligned_2x = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': 'gs://sofima-sample-data/fmi-friedrich-dp/subvol_5800_5500_6250',
        "scale_metadata": {"resolution": [22, 22, 25]},
        "context": {"cache_pool": {"total_bytes_limit": 1_000_000_000}},
    }).result()

    print(f"Data shape (1x): {unaligned_1x.shape}")
    print(f"Data shape (2x): {unaligned_2x.shape}")

    return unaligned_1x, unaligned_2x


def compute_flow(volume, patch_size=160, stride=40):
    """
    Calculate flow fields between consecutive sections.

    Uses JAX-based masked cross-correlation for optical flow estimation.
    """
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flows = []
    prev = volume[..., 0, 0].T.read().result()

    fs = []
    with futures.ThreadPoolExecutor() as tpe:
        for z in range(1, volume.shape[2]):
            fs.append(tpe.submit(lambda z=z: volume[..., z, 0].T.read().result()))

        fs = fs[::-1]

        for z in tqdm(range(1, volume.shape[2]), desc="Computing flow"):
            curr = fs.pop().result()
            flows.append(mfc.flow_field(prev, curr, (patch_size, patch_size),
                                        (stride, stride), batch_size=256))
            prev = curr

    return flows


def visualize_flow_cleaning(flows1x, f1, section_idx=14):
    """Visualize flow field before and after cleaning."""
    f, ax = plt.subplots(1, 2, figsize=(10, 5))

    im0 = ax[0].imshow(flows1x[0, section_idx, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[0].set_title('Raw Flow (X component)')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    plt.colorbar(im0, ax=ax[0], label='Displacement (pixels)')

    im1 = ax[1].imshow(f1[0, section_idx, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[1].set_title('Cleaned Flow (X component)')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    plt.colorbar(im1, ax=ax[1], label='Displacement (pixels)')

    f.suptitle(f'Flow Field Cleaning (Section {section_idx})', fontsize=14)
    save_figure(f, "01_flow_cleaning")


def visualize_flow_reconciliation(f1, f2_hires, final_flow, section_idx=14):
    """Visualize flow reconciliation from multiple resolutions."""
    f, ax = plt.subplots(1, 3, figsize=(15, 5))

    im0 = ax[0].imshow(f1[0, section_idx, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[0].set_title('1x Resolution Flow')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    plt.colorbar(im0, ax=ax[0], label='Displacement')

    im1 = ax[1].imshow(f2_hires[0, section_idx, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[1].set_title('2x Resolution Flow (upsampled)')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    plt.colorbar(im1, ax=ax[1], label='Displacement')

    im2 = ax[2].imshow(final_flow[0, section_idx, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
    ax[2].set_title('Reconciled Flow')
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Y')
    plt.colorbar(im2, ax=ax[2], label='Displacement')

    f.suptitle(f'Flow Reconciliation (Section {section_idx})', fontsize=14)
    save_figure(f, "02_flow_reconciliation")


def visualize_warped_result(warped_xyz):
    """Visualize the final warped/aligned result."""
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(warped_xyz[500, ...], cmap='gray')
    ax.set_title('Aligned Section (XZ slice at Y=500)')
    ax.set_xlabel('Z (section)')
    ax.set_ylabel('X')
    save_figure(f, "03_aligned_xz_slice")


def visualize_before_after(unaligned_1x, warped_xyz):
    """Compare unaligned vs aligned data."""
    # Get a comparable slice from unaligned data
    unaligned_slice = unaligned_1x[1500, 2000:3000, :, 0].read().result()

    f, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].imshow(unaligned_slice.T, cmap='gray', aspect='auto')
    ax[0].set_title('Before Alignment (XZ slice)')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Z (section)')

    ax[1].imshow(warped_xyz[500, ...], cmap='gray', aspect='auto')
    ax[1].set_title('After Alignment (XZ slice)')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Z (section)')

    f.suptitle('SOFIMA EM Section Alignment Result', fontsize=14)
    save_figure(f, "04_before_after_comparison")


def visualize_mesh_solved(solved):
    """Visualize the solved mesh deformation."""
    f, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Show accumulated X displacement across all sections
    im0 = ax[0].imshow(solved[0, :, solved.shape[2]//2, :].T, cmap=plt.cm.RdBu, aspect='auto')
    ax[0].set_title('X Displacement (YZ cross-section at mid-X)')
    ax[0].set_xlabel('Z (section)')
    ax[0].set_ylabel('Y position')
    plt.colorbar(im0, ax=ax[0], label='Displacement (pixels)')

    # Show accumulated Y displacement
    im1 = ax[1].imshow(solved[1, :, solved.shape[2]//2, :].T, cmap=plt.cm.RdBu, aspect='auto')
    ax[1].set_title('Y Displacement (YZ cross-section at mid-X)')
    ax[1].set_xlabel('Z (section)')
    ax[1].set_ylabel('Y position')
    plt.colorbar(im1, ax=ax[1], label='Displacement (pixels)')

    f.suptitle('Mesh Solution: Accumulated Displacements', fontsize=14)
    save_figure(f, "05_mesh_solution")


def main():
    print("=" * 60)
    print("SOFIMA EM Alignment Example")
    print("=" * 60)

    # Check JAX devices
    devices = jax.devices()
    print(f"\nJAX devices: {devices}")
    if devices[0].platform != 'gpu':
        print("WARNING: Running on CPU. GPU recommended for faster processing.")

    # Parameters
    patch_size = 160
    stride = 40

    # Step 1: Load data
    print("\n" + "=" * 60)
    print("Step 1: Loading EM data from Google Cloud Storage")
    print("=" * 60)
    unaligned_1x, unaligned_2x = load_data()

    # Step 2: Compute flow fields at both resolutions
    print("\n" + "=" * 60)
    print("Step 2: Computing optical flow fields")
    print("=" * 60)

    print("\nComputing 1x resolution flows...")
    flows1x = np.array(compute_flow(unaligned_1x, patch_size, stride))

    print("\nComputing 2x resolution flows...")
    flows2x = np.array(compute_flow(unaligned_2x, patch_size, stride))

    # Transpose and pad flows
    flows2x = np.transpose(flows2x, [1, 0, 2, 3])
    flows1x = np.transpose(flows1x, [1, 0, 2, 3])

    pad = patch_size // 2 // stride
    flows1x = np.pad(flows1x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
    flows2x = np.pad(flows2x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    print(f"Flow shape (1x): {flows1x.shape}")
    print(f"Flow shape (2x): {flows2x.shape}")

    # Step 3: Clean flow fields
    print("\n" + "=" * 60)
    print("Step 3: Cleaning flow fields")
    print("=" * 60)

    f1 = flow_utils.clean_flow(flows1x, min_peak_ratio=1.6, min_peak_sharpness=1.6,
                                max_magnitude=80, max_deviation=20)
    f2 = flow_utils.clean_flow(flows2x, min_peak_ratio=1.6, min_peak_sharpness=1.6,
                                max_magnitude=80, max_deviation=20)

    # Visualize flow cleaning
    visualize_flow_cleaning(flows1x, f1)

    # Step 4: Resample 2x flow to 1x resolution and reconcile
    print("\n" + "=" * 60)
    print("Step 4: Resampling and reconciling flows")
    print("=" * 60)

    f2_hires = np.zeros_like(f1)
    scale = 0.5

    box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1))
    box2x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1))

    for z in tqdm(range(f2.shape[1]), desc="Resampling 2x flow"):
        resampled = map_utils.resample_map(
            f2[:, z:z + 1, ...],
            box2x, box1x, 1 / scale, 1)
        f2_hires[:, z:z + 1, ...] = resampled / scale

    final_flow = flow_utils.reconcile_flows((f1, f2_hires), max_gradient=0,
                                             max_deviation=20, min_patch_size=400)

    # Visualize flow reconciliation
    visualize_flow_reconciliation(f1, f2_hires, final_flow)

    # Step 5: Mesh relaxation
    print("\n" + "=" * 60)
    print("Step 5: Mesh relaxation (elastic optimization)")
    print("=" * 60)

    config = mesh.IntegrationConfig(
        dt=0.001, gamma=0.0, k0=0.01, k=0.1,
        stride=(stride, stride), num_iters=1000,
        max_iters=100000, stop_v_max=0.005, dt_max=1000,
        start_cap=0.01, final_cap=10, prefer_orig_order=True
    )

    solved = [np.zeros_like(final_flow[:, 0:1, ...])]
    origin = jnp.array([0., 0.])

    for z in tqdm(range(0, final_flow.shape[1]), desc="Relaxing mesh"):
        prev = map_utils.compose_maps_fast(final_flow[:, z:z+1, ...], origin, stride,
                                           solved[-1], origin, stride)
        x = np.zeros_like(solved[0])
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
        x = np.array(x)
        solved.append(x)

    solved = np.concatenate(solved, axis=1)
    print(f"Solved mesh shape: {solved.shape}")

    # Visualize mesh solution
    visualize_mesh_solved(solved)

    # Step 6: Invert map
    print("\n" + "=" * 60)
    print("Step 6: Inverting coordinate map")
    print("=" * 60)

    inv_map = map_utils.invert_map(solved, box1x, box1x, stride)
    print(f"Inverse map shape: {inv_map.shape}")

    # Step 7: Warp images
    print("\n" + "=" * 60)
    print("Step 7: Warping images to aligned coordinates")
    print("=" * 60)

    warped = [np.transpose(unaligned_1x[1000:2000, 2000:3000, 0:1, 0].read().result(), [2, 1, 0])]

    for z in tqdm(range(1, unaligned_1x.shape[2]), desc="Warping sections"):
        data_box = bounding_box.BoundingBox(start=(500, 1500, 0), size=(2000, 2000, 1))
        out_box = bounding_box.BoundingBox(start=(1000, 2000, 0), size=(1000, 1000, 1))

        data = np.transpose(unaligned_1x[data_box.start[0]:data_box.end[0],
                                         data_box.start[1]:data_box.end[1],
                                         z:z+1, 0:1].read().result(), [3, 2, 1, 0])
        warped.append(
            warp.warp_subvolume(data, data_box, inv_map[:, z:z+1, ...], box1x, stride,
                               out_box, 'lanczos', parallelism=1)[0, ...])

    warped_xyz = np.transpose(np.concatenate(warped, axis=0), [2, 1, 0])
    print(f"Warped volume shape: {warped_xyz.shape}")

    # Visualize final result
    visualize_warped_result(warped_xyz)
    visualize_before_after(unaligned_1x, warped_xyz)

    print("\n" + "=" * 60)
    print("Done! All images saved to:", OUTPUT_DIR)
    print("=" * 60)
    print("\nGenerated images:")
    print("  - temp_01_flow_cleaning.png")
    print("  - temp_02_flow_reconciliation.png")
    print("  - temp_03_aligned_xz_slice.png")
    print("  - temp_04_before_after_comparison.png")
    print("  - temp_05_mesh_solution.png")


if __name__ == "__main__":
    main()
