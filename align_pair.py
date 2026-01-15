#!/usr/bin/env python
"""
SOFIMA 2D Pairwise Alignment

Aligns two 2D images using optical flow and elastic mesh relaxation.

Usage:
    pixi run python align_pair.py <reference.zarr> <moving.zarr> [options]

Example:
    pixi run python align_pair.py ref.zarr moving.zarr --output aligned.zarr
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp
from connectomics.common import bounding_box

from sofima_utils import (
    load_image, save_image, save_figure, compute_flow_single,
    add_device_args, add_flow_args, add_cleaning_args, add_mesh_args,
    handle_device_args, create_mesh_config
)


def main():
    parser = argparse.ArgumentParser(
        description='SOFIMA 2D Pairwise Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic alignment:
    python align_pair.py ref.zarr moving.zarr --output aligned.zarr

  With flow field outputs:
    python align_pair.py ref.zarr moving.zarr -o aligned.zarr \\
        --output-flow-raw flow_raw.zarr --output-flow flow_clean.zarr --output-map inv_map.zarr

  Relaxed cleaning for well-aligned images:
    python align_pair.py ref.zarr moving.zarr -o aligned.zarr \\
        --min-peak-ratio 1.2 --min-peak-sharpness 1.2 --max-deviation 40

  Stiffer mesh (less deformation):
    python align_pair.py ref.zarr moving.zarr -o aligned.zarr --k0 0.1 --k 1.0

Scale considerations:
  When running on downsampled images, pixel-based parameters may need adjustment:
  - max-magnitude, max-deviation: Scale with image resolution (e.g., divide by 2 for 2x downsampled)
  - patch-size, stride: Usually keep similar values across scales for consistent correlation quality
"""
    )

    # Device selection
    add_device_args(parser)

    # Input/Output arguments
    parser.add_argument('reference', nargs='?', help='Reference image (ZARR or TIFF) - the target coordinate space')
    parser.add_argument('moving', nargs='?', help='Moving image (ZARR or TIFF) - will be warped to match reference')
    parser.add_argument('--output', '-o', default='aligned.zarr',
                        help='Output aligned image (default: aligned.zarr)')
    parser.add_argument('--output-flow-raw',
                        help='Output raw flow field before cleaning (ZARR). Shape: [2, 1, y, x] where '
                             'channels are [x_displacement, y_displacement]. Includes quality stats.')
    parser.add_argument('--output-flow',
                        help='Output cleaned flow field (ZARR). Same shape as raw. Invalid vectors are NaN.')
    parser.add_argument('--output-map',
                        help='Output inverse coordinate map used to warp image (ZARR). This is the '
                             'final field applied by warp.warp_subvolume() to deform the moving image.')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Save visualization PNGs (flow field, before/after, overlay)')

    # Add shared argument groups
    add_flow_args(parser)
    add_cleaning_args(parser)
    add_mesh_args(parser)

    args = parser.parse_args()

    # Handle device listing/selection
    handle_device_args(args, parser)

    # Validate required arguments when not listing devices
    if not args.reference or not args.moving:
        parser.error("the following arguments are required: reference, moving")

    print("=" * 60)
    print("SOFIMA 2D Pairwise Alignment")
    print("=" * 60)

    print(f"\nJAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")

    # Load images
    print(f"\n[1] Loading images...")
    ref = load_image(args.reference).astype(np.float32)
    mov = load_image(args.moving).astype(np.float32)
    print(f"    Reference: {args.reference} {ref.shape}")
    print(f"    Moving:    {args.moving} {mov.shape}")

    if ref.shape != mov.shape:
        raise ValueError(f"Image shapes must match: {ref.shape} vs {mov.shape}")

    # Compute flow
    print(f"\n[2] Computing optical flow (patch={args.patch_size}, stride={args.stride}, batch={args.batch_size})...")
    flow = compute_flow_single(ref, mov, args.patch_size, args.stride, args.batch_size)
    print(f"    Flow shape: {flow.shape}")

    # Save raw flow if requested
    if args.output_flow_raw:
        flow_raw_4d = flow[:, np.newaxis, :, :].astype(np.float32)
        valid_mask = ~np.isnan(flow_raw_4d[0])
        n_valid = np.sum(valid_mask)
        if n_valid > 0:
            valid_vals = flow_raw_4d[0][valid_mask]
            print(f"    Raw flow stats - valid: {n_valid}, min: {valid_vals.min():.4f}, max: {valid_vals.max():.4f}, mean: {valid_vals.mean():.4f}")
        else:
            print(f"    Raw flow stats - WARNING: no valid flow vectors!")
        save_image(args.output_flow_raw, flow_raw_4d)
        print(f"    Raw flow field: {args.output_flow_raw}")

    # Clean flow
    print(f"\n[3] Cleaning flow field...")
    # Add z dimension for compatibility (flow_utils expects [c, z, y, x])
    flow_4d = flow[:, np.newaxis, :, :]
    cleaned = flow_utils.clean_flow(
        flow_4d,
        min_peak_ratio=args.min_peak_ratio,
        min_peak_sharpness=args.min_peak_sharpness,
        max_magnitude=args.max_magnitude,
        max_deviation=args.max_deviation
    )

    valid_before = np.sum(~np.isnan(flow_4d[0]))
    valid_after = np.sum(~np.isnan(cleaned[0]))
    print(f"    Valid flow vectors: {valid_before} -> {valid_after}")

    # Mesh relaxation
    print(f"\n[4] Mesh relaxation (k0={args.k0}, k={args.k}, dt={args.dt}, stop_v_max={args.stop_v_max})...")
    config = create_mesh_config(args, args.stride)

    # For single pair, we solve directly
    x = np.zeros_like(cleaned)
    origin = jnp.array([0., 0.])

    x, e_kin, num_steps = mesh.relax_mesh(x, cleaned, config)
    x = np.array(x)
    print(f"    Converged in {num_steps} steps")

    # Invert map
    print(f"\n[5] Inverting coordinate map...")
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(x.shape[-1], x.shape[-2], 1))
    inv_map = map_utils.invert_map(x, box, box, args.stride)

    # Warp image
    print(f"\n[6] Warping moving image...")
    # Prepare data for warping
    mov_4d = mov[np.newaxis, np.newaxis, :, :]  # [c, z, y, x]
    data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(mov.shape[1], mov.shape[0], 1))

    warped = warp.warp_subvolume(
        mov_4d, data_box, inv_map, box, args.stride, data_box, 'lanczos', parallelism=1
    )
    aligned = warped[0, 0, :, :]
    print(f"    Aligned shape: {aligned.shape}")

    # Save output
    print(f"\n[7] Saving results...")
    save_image(args.output, aligned.astype(np.uint8))
    print(f"    Aligned image: {args.output}")

    if args.output_flow:
        # Ensure float32 dtype is preserved
        flow_out = cleaned.astype(np.float32)
        valid_mask = ~np.isnan(flow_out[0])
        n_valid = np.sum(valid_mask)
        if n_valid > 0:
            valid_vals = flow_out[0][valid_mask]
            print(f"    Flow stats - valid: {n_valid}, min: {valid_vals.min():.4f}, max: {valid_vals.max():.4f}, mean: {valid_vals.mean():.4f}")
        else:
            print(f"    Flow stats - WARNING: no valid flow vectors after cleaning!")
        save_image(args.output_flow, flow_out)
        print(f"    Flow field: {args.output_flow} (dtype: {flow_out.dtype})")

    if args.output_map:
        map_out = inv_map.astype(np.float32)
        print(f"    Map stats - min: {np.nanmin(map_out):.4f}, max: {np.nanmax(map_out):.4f}, mean: {np.nanmean(map_out):.4f}")
        save_image(args.output_map, map_out)
        print(f"    Inverse map: {args.output_map} (shape: {map_out.shape}, dtype: {map_out.dtype})")

    # Visualizations
    if args.visualize:
        print(f"\n[8] Saving visualizations...")
        output_dir = os.path.dirname(args.output) or '.'

        # Flow field
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im0 = ax[0].imshow(cleaned[0, 0], cmap=plt.cm.RdBu, vmin=-20, vmax=20)
        ax[0].set_title('X Flow')
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(cleaned[1, 0], cmap=plt.cm.RdBu, vmin=-20, vmax=20)
        ax[1].set_title('Y Flow')
        plt.colorbar(im1, ax=ax[1])
        save_figure(fig, os.path.join(output_dir, 'flow_field.png'))

        # Before/after comparison
        fig, ax = plt.subplots(2, 2, figsize=(14, 14))
        ax[0, 0].imshow(ref, cmap='gray')
        ax[0, 0].set_title('Reference')
        ax[0, 1].imshow(mov, cmap='gray')
        ax[0, 1].set_title('Moving (before)')
        ax[1, 0].imshow(ref, cmap='gray')
        ax[1, 0].set_title('Reference')
        ax[1, 1].imshow(aligned, cmap='gray')
        ax[1, 1].set_title('Moving (after alignment)')
        for a in ax.flat:
            a.axis('off')
        save_figure(fig, os.path.join(output_dir, 'before_after.png'))

        # Overlay comparison
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        overlay_before = np.stack([ref/255, mov/255, ref/255], axis=-1)
        overlay_after = np.stack([ref/255, aligned/255, ref/255], axis=-1)
        ax[0].imshow(np.clip(overlay_before, 0, 1))
        ax[0].set_title('Before (green=moving, magenta=reference)')
        ax[0].axis('off')
        ax[1].imshow(np.clip(overlay_after, 0, 1))
        ax[1].set_title('After (green=aligned, magenta=reference)')
        ax[1].axis('off')
        save_figure(fig, os.path.join(output_dir, 'overlay_comparison.png'))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
