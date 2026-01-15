#!/usr/bin/env python
"""
SOFIMA Multi-Slice Stack Alignment

Aligns multiple consecutive 2D slices using optical flow, multi-resolution
reconciliation, and elastic mesh relaxation.

Usage:
    # Single 3D file
    pixi run python align_stack.py --input stack.zarr --output aligned.zarr

    # Multiple 2D files (glob pattern)
    pixi run python align_stack.py --input-pattern "slice_*.zarr" --output aligned.zarr

    # Explicit list of files
    pixi run python align_stack.py --input-list s0.zarr s1.zarr s2.zarr --output aligned.zarr

    # HDF5 files with multiresolution pyramid
    pixi run python align_stack.py --input-pattern "*.h5" --mipmap 3 --output aligned.zarr

    # List available mipmap levels
    pixi run python align_stack.py --input-pattern "*.h5" --list-mipmaps
"""

import argparse
import glob
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp
from connectomics.common import bounding_box

from sofima_utils import (
    load_image, save_image, save_figure, compute_flow_single, downsample_image,
    load_h5_mipmap, get_h5_mipmap_info,
    add_device_args, add_flow_args, add_cleaning_args, add_mesh_args,
    handle_device_args, create_mesh_config
)


def load_stack(args):
    """Load stack from various input sources.

    Args:
        args: Parsed arguments with input, input_pattern, or input_list

    Returns:
        stack: numpy array [z, y, x]
        filenames: list of source filenames (for reference)
    """
    mipmap_level = getattr(args, 'mipmap', 0)
    h5_group = getattr(args, 'h5_group', '0-0-0')

    if args.input:
        # Single 3D file
        print(f"Loading 3D stack from: {args.input}")
        if args.input.endswith('.h5'):
            stack = load_h5_mipmap(args.input, mipmap_level, h5_group)
        else:
            stack = load_image(args.input)
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]  # Add z dimension
        filenames = [args.input]

    elif args.input_pattern:
        # Glob pattern
        files = sorted(glob.glob(args.input_pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {args.input_pattern}")
        print(f"Loading {len(files)} files matching: {args.input_pattern}")

        # Check if HDF5 files
        is_h5 = files[0].endswith('.h5')
        if is_h5:
            print(f"    HDF5 mode: mipmap level {mipmap_level}, group '{h5_group}'")

        slices = []
        for f in tqdm(files, desc="Loading slices"):
            if is_h5:
                slices.append(load_h5_mipmap(f, mipmap_level, h5_group))
            else:
                slices.append(load_image(f))
        stack = np.stack(slices, axis=0)
        filenames = files

    elif args.input_list:
        # Explicit list
        files = args.input_list
        print(f"Loading {len(files)} files from list")

        # Check if HDF5 files
        is_h5 = files[0].endswith('.h5')
        if is_h5:
            print(f"    HDF5 mode: mipmap level {mipmap_level}, group '{h5_group}'")

        slices = []
        for f in tqdm(files, desc="Loading slices"):
            if is_h5:
                slices.append(load_h5_mipmap(f, mipmap_level, h5_group))
            else:
                slices.append(load_image(f))
        stack = np.stack(slices, axis=0)
        filenames = files

    else:
        raise ValueError("Must specify --input, --input-pattern, or --input-list")

    return stack, filenames


def list_h5_mipmaps(args):
    """List available mipmap levels from an HDF5 file.

    Args:
        args: Parsed arguments

    Returns:
        True if mipmaps were listed, False otherwise
    """
    # Find an HDF5 file to inspect
    h5_file = None
    if args.input and args.input.endswith('.h5'):
        h5_file = args.input
    elif args.input_pattern:
        files = sorted(glob.glob(args.input_pattern))
        if files and files[0].endswith('.h5'):
            h5_file = files[0]
    elif args.input_list:
        if args.input_list[0].endswith('.h5'):
            h5_file = args.input_list[0]

    if not h5_file:
        print("No HDF5 file found to inspect. Provide --input, --input-pattern, or --input-list with .h5 files.")
        return False

    h5_group = getattr(args, 'h5_group', '0-0-0')
    print(f"Mipmap levels in: {h5_file}")
    print(f"Group: {h5_group}")
    print("-" * 40)

    info = get_h5_mipmap_info(h5_file, h5_group)
    for level in sorted(info.keys()):
        shape = info[level]['shape']
        scale = info[level]['scale']
        print(f"  mipmap.{level}: {shape[0]:>5} x {shape[1]:<5} ({scale}x downsampled)")

    return True


def compute_flows_for_stack(stack, patch_size, stride, batch_size):
    """Compute flow fields between consecutive slices.

    Args:
        stack: Image stack [z, y, x]
        patch_size: Correlation patch size
        stride: Flow vector spacing
        batch_size: GPU batch size

    Returns:
        flows: numpy array [2, z-1, y, x]
    """
    flows = []
    for z in tqdm(range(1, stack.shape[0]), desc="Computing flows"):
        flow = compute_flow_single(
            stack[z-1].astype(np.float32),
            stack[z].astype(np.float32),
            patch_size, stride, batch_size
        )
        flows.append(flow)

    # Stack and transpose to [2, z, y, x]
    flows = np.array(flows)  # [z-1, 2, y, x]
    flows = np.transpose(flows, [1, 0, 2, 3])  # [2, z-1, y, x]

    # Pad edges
    pad = patch_size // 2 // stride
    flows = np.pad(flows, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    return flows


def resample_flows_to_target(flows, source_box, target_box, scale):
    """Resample flows from one resolution to another.

    Args:
        flows: Flow field [2, z, y, x]
        source_box: BoundingBox for source resolution
        target_box: BoundingBox for target resolution
        scale: Scale factor (e.g., 0.5 for 2x upsampling)

    Returns:
        resampled: Flow field at target resolution
    """
    resampled = np.zeros((flows.shape[0], flows.shape[1], target_box.size[1], target_box.size[0]))

    for z in range(flows.shape[1]):
        r = map_utils.resample_map(
            flows[:, z:z+1, ...],
            source_box, target_box,
            1 / scale, 1
        )
        resampled[:, z:z+1, ...] = r / scale

    return resampled


def solve_mesh_cumulative(final_flow, config, stride):
    """Solve mesh relaxation with cumulative map composition.

    Each section's deformation builds on the cumulative transformation from
    all previous sections.

    Args:
        final_flow: Reconciled flow field [2, z, y, x]
        config: mesh.IntegrationConfig
        stride: Flow stride

    Returns:
        solved: Cumulative solution [2, z+1, y, x]
    """
    # Initialize with zero deformation for first section
    solved = [np.zeros_like(final_flow[:, 0:1, ...])]
    origin = jnp.array([0., 0.])

    for z in tqdm(range(final_flow.shape[1]), desc="Mesh relaxation"):
        # Compose current flow with accumulated solution
        prev = map_utils.compose_maps_fast(
            final_flow[:, z:z+1, ...],
            origin, stride,
            solved[-1],
            origin, stride
        )

        # Relax this composed map
        x = np.zeros_like(solved[0])
        x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
        solved.append(np.array(x))

    # Concatenate all solutions
    solved = np.concatenate(solved, axis=1)  # [2, z+1, y, x]
    return solved


def warp_stack(stack, inv_map, stride):
    """Warp all sections using inverse map.

    Args:
        stack: Image stack [z, y, x]
        inv_map: Inverse coordinate map [2, z, y, x]
        stride: Map stride

    Returns:
        warped: Warped stack [z, y, x]
    """
    box = bounding_box.BoundingBox(
        start=(0, 0, 0),
        size=(inv_map.shape[-1], inv_map.shape[-2], 1)
    )
    data_box = bounding_box.BoundingBox(
        start=(0, 0, 0),
        size=(stack.shape[2], stack.shape[1], 1)
    )

    warped = [stack[0]]  # First section unchanged

    for z in tqdm(range(1, stack.shape[0]), desc="Warping"):
        # Prepare 4D input [c, z, y, x]
        section_4d = stack[z:z+1, np.newaxis, :, :].astype(np.float32)
        section_4d = np.transpose(section_4d, [1, 0, 2, 3])  # [1, 1, y, x]

        warped_section = warp.warp_subvolume(
            section_4d, data_box,
            inv_map[:, z:z+1, ...], box, stride,
            data_box, 'lanczos', parallelism=1
        )
        warped.append(warped_section[0, 0, :, :])

    return np.stack(warped, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description='SOFIMA Multi-Slice Stack Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single 3D file:
    python align_stack.py --input stack.zarr --output aligned.zarr

  Glob pattern:
    python align_stack.py --input-pattern "slices/z*.zarr" --output aligned.zarr

  Explicit list:
    python align_stack.py --input-list z0.zarr z1.zarr z2.zarr --output aligned.zarr

  HDF5 files with multiresolution (list available levels first):
    python align_stack.py --input-pattern "fibsem/*.h5" --list-mipmaps
    python align_stack.py --input-pattern "fibsem/*.h5" --mipmap 3 --output aligned.zarr

  With multi-resolution (more robust):
    python align_stack.py --input stack.zarr --output aligned.zarr --downsample-factor 2

  Skip multi-resolution (faster):
    python align_stack.py --input stack.zarr --output aligned.zarr --skip-multi-res

Scale considerations:
  When running on downsampled images, pixel-based parameters may need adjustment:
  - max-magnitude, max-deviation: Scale with image resolution (e.g., divide by 2 for 2x downsampled)
  - patch-size, stride: Usually keep similar values across scales for consistent correlation quality

HDF5 format:
  Each .h5 file should contain a group (default: '0-0-0') with mipmap datasets:
    mipmap.0 = full resolution
    mipmap.1 = 2x downsampled
    mipmap.2 = 4x downsampled, etc.
"""
    )

    # Device selection
    add_device_args(parser)

    # Input arguments (mutually exclusive)
    input_group = parser.add_argument_group('Input (choose one)')
    input_mutex = input_group.add_mutually_exclusive_group()
    input_mutex.add_argument('--input', '-i',
                             help='Single 3D ZARR/TIFF/H5 file [z, y, x]')
    input_mutex.add_argument('--input-pattern',
                             help='Glob pattern for 2D files (e.g., "slice_*.zarr" or "*.h5")')
    input_mutex.add_argument('--input-list', nargs='+',
                             help='Explicit list of 2D files in order')

    # HDF5-specific arguments
    h5_group = parser.add_argument_group('HDF5 Options',
        'Options for loading HDF5 files with multiresolution pyramids')
    h5_group.add_argument('--mipmap', type=int, default=0,
                          help='Mipmap level to load (0=full res, 1=2x down, 2=4x down, etc.). '
                               'Default: 0 (full resolution)')
    h5_group.add_argument('--h5-group', default='0-0-0',
                          help='HDF5 group containing mipmaps (default: "0-0-0")')
    h5_group.add_argument('--list-mipmaps', action='store_true',
                          help='List available mipmap levels and exit')

    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', '-o',
                              help='Output aligned stack (3D ZARR or TIFF). Required unless --list-mipmaps.')
    output_group.add_argument('--output-unaligned',
                              help='Output unaligned stack (3D ZARR or TIFF). Useful for comparison.')
    output_group.add_argument('--output-map',
                              help='Output inverse coordinate map (ZARR). Shape: [2, z, y, x]')
    output_group.add_argument('--output-flow',
                              help='Output reconciled flow field (ZARR). Shape: [2, z-1, y, x]')
    output_group.add_argument('--visualize', '-v', action='store_true',
                              help='Save visualization PNGs')

    # Multi-resolution arguments
    multires_group = parser.add_argument_group('Multi-Resolution',
        'Use multiple resolutions for more robust flow estimation')
    multires_group.add_argument('--downsample-factor', type=int, default=2,
                                help='Downsample factor for second resolution (default: 2)')
    multires_group.add_argument('--skip-multi-res', action='store_true',
                                help='Skip multi-resolution, use single resolution only')

    # Reconciliation parameters
    multires_group.add_argument('--reconcile-max-gradient', type=float, default=0,
                                help='Max gradient for flow reconciliation (default: 0)')
    multires_group.add_argument('--reconcile-min-patch-size', type=int, default=400,
                                help='Min patch size for reconciliation (default: 400)')

    # Add shared argument groups
    add_flow_args(parser)
    add_cleaning_args(parser)
    add_mesh_args(parser)

    args = parser.parse_args()

    # Handle device listing/selection
    handle_device_args(args, parser)

    # Handle mipmap listing
    if args.list_mipmaps:
        if list_h5_mipmaps(args):
            sys.exit(0)
        else:
            sys.exit(1)

    # Validate input
    if not args.input and not args.input_pattern and not args.input_list:
        parser.error("Must specify --input, --input-pattern, or --input-list")

    # Validate output is provided (unless listing mipmaps)
    if not args.output:
        parser.error("--output is required")

    print("=" * 60)
    print("SOFIMA Multi-Slice Stack Alignment")
    print("=" * 60)

    print(f"\nJAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")

    # Step 1: Load stack
    print(f"\n[1] Loading stack...")
    stack, filenames = load_stack(args)
    print(f"    Stack shape: {stack.shape} (z={stack.shape[0]}, y={stack.shape[1]}, x={stack.shape[2]})")

    if stack.shape[0] < 2:
        raise ValueError("Stack must have at least 2 sections for alignment")

    # Step 2: Compute flows
    print(f"\n[2] Computing optical flows (patch={args.patch_size}, stride={args.stride})...")

    # Full resolution flows
    flows_1x = compute_flows_for_stack(stack, args.patch_size, args.stride, args.batch_size)
    print(f"    1x flow shape: {flows_1x.shape}")

    # Multi-resolution flows
    if not args.skip_multi_res:
        print(f"\n[3] Computing {args.downsample_factor}x downsampled flows...")
        stack_ds = downsample_image(stack, args.downsample_factor)
        print(f"    Downsampled stack shape: {stack_ds.shape}")

        flows_ds = compute_flows_for_stack(stack_ds, args.patch_size, args.stride, args.batch_size)
        print(f"    {args.downsample_factor}x flow shape: {flows_ds.shape}")

    # Step 3: Clean flows
    print(f"\n[4] Cleaning flow fields...")
    f1 = flow_utils.clean_flow(
        flows_1x,
        min_peak_ratio=args.min_peak_ratio,
        min_peak_sharpness=args.min_peak_sharpness,
        max_magnitude=args.max_magnitude,
        max_deviation=args.max_deviation
    )

    valid_1x = np.sum(~np.isnan(f1[0]))
    print(f"    1x valid vectors: {valid_1x}")

    if not args.skip_multi_res:
        # Scale cleaning parameters for downsampled resolution
        f2 = flow_utils.clean_flow(
            flows_ds,
            min_peak_ratio=args.min_peak_ratio,
            min_peak_sharpness=args.min_peak_sharpness,
            max_magnitude=args.max_magnitude / args.downsample_factor,
            max_deviation=args.max_deviation / args.downsample_factor
        )
        valid_ds = np.sum(~np.isnan(f2[0]))
        print(f"    {args.downsample_factor}x valid vectors: {valid_ds}")

    # Step 4: Reconcile flows (if multi-res)
    if not args.skip_multi_res:
        print(f"\n[5] Resampling and reconciling flows...")

        # Resample downsampled flows to full resolution
        box_1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1))
        box_ds = bounding_box.BoundingBox(start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1))

        f2_hires = resample_flows_to_target(f2, box_ds, box_1x, 1.0 / args.downsample_factor)

        # Reconcile
        final_flow = flow_utils.reconcile_flows(
            (f1, f2_hires),
            max_gradient=args.reconcile_max_gradient,
            max_deviation=args.max_deviation,
            min_patch_size=args.reconcile_min_patch_size
        )
        print(f"    Reconciled flow shape: {final_flow.shape}")
    else:
        print(f"\n[5] Skipping multi-resolution reconciliation...")
        final_flow = f1

    # Step 5: Mesh relaxation
    print(f"\n[6] Mesh relaxation (k0={args.k0}, k={args.k})...")
    config = create_mesh_config(args, args.stride)
    solved = solve_mesh_cumulative(final_flow, config, args.stride)
    print(f"    Solved mesh shape: {solved.shape}")

    # Step 6: Invert map
    print(f"\n[7] Inverting coordinate map...")
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(solved.shape[-1], solved.shape[-2], 1))
    inv_map = map_utils.invert_map(solved, box, box, args.stride)
    print(f"    Inverse map shape: {inv_map.shape}")

    # Step 7: Warp stack
    print(f"\n[8] Warping stack...")
    warped_stack = warp_stack(stack, inv_map, args.stride)
    print(f"    Warped stack shape: {warped_stack.shape}")

    # Step 8: Save outputs
    print(f"\n[9] Saving results...")
    save_image(args.output, warped_stack.astype(np.uint8))
    print(f"    Aligned stack: {args.output}")

    if args.output_unaligned:
        save_image(args.output_unaligned, stack.astype(np.uint8))
        print(f"    Unaligned stack: {args.output_unaligned}")

    if args.output_map:
        save_image(args.output_map, inv_map.astype(np.float32))
        print(f"    Inverse map: {args.output_map}")

    if args.output_flow:
        save_image(args.output_flow, final_flow.astype(np.float32))
        print(f"    Flow field: {args.output_flow}")

    # Visualizations
    if args.visualize:
        print(f"\n[10] Saving visualizations...")
        output_dir = os.path.dirname(args.output) or '.'

        # Before/after for first and last section
        for idx, name in [(0, 'first'), (stack.shape[0]-1, 'last')]:
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            ax[0].imshow(stack[idx], cmap='gray')
            ax[0].set_title(f'Section {idx} - Before')
            ax[0].axis('off')
            ax[1].imshow(warped_stack[idx], cmap='gray')
            ax[1].set_title(f'Section {idx} - After')
            ax[1].axis('off')
            save_figure(fig, os.path.join(output_dir, f'section_{name}.png'))

        # XZ slice (cross-section)
        mid_y = stack.shape[1] // 2
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(stack[:, mid_y, :], cmap='gray', aspect='auto')
        ax[0].set_title(f'XZ slice (y={mid_y}) - Before')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Z')
        ax[1].imshow(warped_stack[:, mid_y, :], cmap='gray', aspect='auto')
        ax[1].set_title(f'XZ slice (y={mid_y}) - After')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Z')
        save_figure(fig, os.path.join(output_dir, 'xz_slice.png'))

        # Flow field for middle section
        if final_flow.shape[1] > 0:
            mid_z = final_flow.shape[1] // 2
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            im0 = ax[0].imshow(final_flow[0, mid_z], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
            ax[0].set_title(f'X Flow (z={mid_z})')
            plt.colorbar(im0, ax=ax[0])
            im1 = ax[1].imshow(final_flow[1, mid_z], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
            ax[1].set_title(f'Y Flow (z={mid_z})')
            plt.colorbar(im1, ax=ax[1])
            save_figure(fig, os.path.join(output_dir, 'flow_field.png'))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
