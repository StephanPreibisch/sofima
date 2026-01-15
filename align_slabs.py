#!/usr/bin/env python
"""
SOFIMA Multi-SEM Slab Alignment

Aligns consecutive slab face pairs for multi-SEM data.
For each pair: slab N's bot-face aligns with slab (N+1)'s top-face.

Usage:
    pixi run python align_slabs.py \\
        --pattern "./data/flat-*-w61_s{slab}_r00-{face}-face.zarr" \\
        --output-dir ./aligned \\
        --output-aligned --output-maps

The pattern uses {slab} for slab number and {face} for bot/top.
"""

import argparse
import glob
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

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


def find_slab_files(pattern, face):
    """Find all files matching pattern for a specific face type.

    Args:
        pattern: Pattern with {slab} and {face} placeholders
        face: 'bot' or 'top'

    Returns:
        dict: {slab_number: filepath}
    """
    # Replace {face} with actual value, {slab} with wildcard for glob
    glob_pattern = pattern.replace('{face}', face).replace('{slab}', '*')

    # Create regex to extract slab number
    # First replace {face}, then escape special chars, then handle {slab} and wildcards
    regex_pattern = pattern.replace('{face}', face)
    # Replace {slab} with a placeholder before escaping
    regex_pattern = regex_pattern.replace('{slab}', '___SLAB___')
    # Replace glob wildcards with placeholder
    regex_pattern = regex_pattern.replace('*', '___STAR___')
    # Escape regex special characters
    regex_pattern = re.escape(regex_pattern)
    # Restore placeholders as regex patterns
    regex_pattern = regex_pattern.replace('___SLAB___', r'(\d+)')
    regex_pattern = regex_pattern.replace('___STAR___', r'.*')
    regex = re.compile(regex_pattern)

    files = {}
    for filepath in glob.glob(glob_pattern):
        match = regex.match(filepath)
        if match:
            slab_num = int(match.group(1))
            files[slab_num] = filepath

    return files


def discover_pairs(pattern):
    """Discover all bot-top pairs from pattern.

    Args:
        pattern: Pattern with {slab} and {face} placeholders

    Returns:
        list of tuples: [(slab_n, bot_path, slab_n+1, top_path), ...]
    """
    bot_files = find_slab_files(pattern, 'bot')
    top_files = find_slab_files(pattern, 'top')

    print(f"Found {len(bot_files)} bot faces: {sorted(bot_files.keys())}")
    print(f"Found {len(top_files)} top faces: {sorted(top_files.keys())}")

    pairs = []
    for slab_n in sorted(bot_files.keys()):
        slab_n1 = slab_n + 1
        if slab_n1 in top_files:
            pairs.append((slab_n, bot_files[slab_n], slab_n1, top_files[slab_n1]))

    return pairs


def align_pair(ref_path, mov_path, args):
    """Align a single pair of images.

    Args:
        ref_path: Path to reference image (bot face)
        mov_path: Path to moving image (top face)
        args: Parsed arguments

    Returns:
        aligned: Aligned image
        inv_map: Inverse coordinate map
        flow_cleaned: Cleaned flow field
    """
    # Load images
    ref = load_image(ref_path).astype(np.float32)
    mov = load_image(mov_path).astype(np.float32)

    if ref.shape != mov.shape:
        raise ValueError(f"Image shapes must match: {ref.shape} vs {mov.shape}")

    # Compute flow
    flow = compute_flow_single(ref, mov, args.patch_size, args.stride, args.batch_size)

    # Clean flow
    flow_4d = flow[:, np.newaxis, :, :]
    cleaned = flow_utils.clean_flow(
        flow_4d,
        min_peak_ratio=args.min_peak_ratio,
        min_peak_sharpness=args.min_peak_sharpness,
        max_magnitude=args.max_magnitude,
        max_deviation=args.max_deviation
    )

    # Mesh relaxation
    config = create_mesh_config(args, args.stride)
    x = np.zeros_like(cleaned)
    x, e_kin, num_steps = mesh.relax_mesh(x, cleaned, config)
    x = np.array(x)

    # Invert map
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(x.shape[-1], x.shape[-2], 1))
    inv_map = map_utils.invert_map(x, box, box, args.stride)

    # Warp image
    mov_4d = mov[np.newaxis, np.newaxis, :, :]
    data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(mov.shape[1], mov.shape[0], 1))

    warped = warp.warp_subvolume(
        mov_4d, data_box, inv_map, box, args.stride, data_box, 'lanczos', parallelism=1
    )
    aligned = warped[0, 0, :, :]

    return aligned, inv_map, cleaned, ref, mov


def main():
    parser = argparse.ArgumentParser(
        description='SOFIMA Multi-SEM Slab Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Align all slab pairs, save aligned images:
    python align_slabs.py \\
        --pattern "./pass03-scale5/flat-*-w61_s{slab}_r00-{face}-face.zarr" \\
        --output-dir ./results \\
        --output-aligned

  Save inverse maps (for later application):
    python align_slabs.py \\
        --pattern "./pass03-scale5/flat-*-w61_s{slab}_r00-{face}-face.zarr" \\
        --output-dir ./results \\
        --output-maps

  Save everything including combined stack:
    python align_slabs.py \\
        --pattern "./pass03-scale5/flat-*-w61_s{slab}_r00-{face}-face.zarr" \\
        --output-dir ./results \\
        --output-aligned --output-maps --output-stack

  Process specific slab range:
    python align_slabs.py \\
        --pattern "./data/flat-*-w61_s{slab}_r00-{face}-face.zarr" \\
        --slabs 079-083 \\
        --output-dir ./results \\
        --output-aligned

Pattern placeholders:
  {slab}  - Slab number (e.g., 079, 080, 081)
  {face}  - Face type (bot or top)

Alignment pairs:
  For each consecutive pair: slab N bot-face → slab (N+1) top-face
  The bot-face is the reference, top-face is warped to match.
"""
    )

    # Device selection
    add_device_args(parser)

    # Input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--pattern', '-p', required=True,
                             help='File pattern with {slab} and {face} placeholders')
    input_group.add_argument('--slabs',
                             help='Slab range to process (e.g., "079-083"). Default: all found.')

    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', '-o', required=True,
                              help='Output directory for results')
    output_group.add_argument('--output-aligned', action='store_true',
                              help='Save aligned (warped) images')
    output_group.add_argument('--output-maps', action='store_true',
                              help='Save inverse coordinate maps')
    output_group.add_argument('--output-flow', action='store_true',
                              help='Save cleaned flow fields')
    output_group.add_argument('--output-stack', action='store_true',
                              help='Save all faces as 3D stack (original + aligned)')
    output_group.add_argument('--visualize', '-v', action='store_true',
                              help='Save visualization PNGs for each pair')

    # Add shared argument groups
    add_flow_args(parser)
    add_cleaning_args(parser)
    add_mesh_args(parser)

    args = parser.parse_args()

    # Handle device listing/selection
    handle_device_args(args, parser)

    # Validate outputs
    if not any([args.output_aligned, args.output_maps, args.output_flow,
                args.output_stack, args.visualize]):
        parser.error("Must specify at least one output: --output-aligned, --output-maps, "
                     "--output-flow, --output-stack, or --visualize")

    print("=" * 60)
    print("SOFIMA Multi-SEM Slab Alignment")
    print("=" * 60)

    print(f"\nJAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")

    # Discover pairs
    print(f"\n[1] Discovering slab pairs...")
    print(f"    Pattern: {args.pattern}")
    pairs = discover_pairs(args.pattern)

    if not pairs:
        print("ERROR: No valid pairs found!")
        sys.exit(1)

    # Filter by slab range if specified
    if args.slabs:
        start, end = args.slabs.split('-')
        start, end = int(start), int(end)
        pairs = [(n, bp, n1, tp) for n, bp, n1, tp in pairs
                 if start <= n <= end - 1]  # n is the bot slab, need n+1 for top

    print(f"\n    Found {len(pairs)} pairs to align:")
    for slab_n, bot_path, slab_n1, top_path in pairs:
        print(f"      {slab_n:03d}-bot → {slab_n1:03d}-top")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each pair
    results = []
    all_faces = []  # For stack output: (slab_num, face_type, image)

    for i, (slab_n, bot_path, slab_n1, top_path) in enumerate(pairs):
        print(f"\n[{i+2}] Aligning pair {slab_n:03d}-bot → {slab_n1:03d}-top...")
        print(f"    Reference: {os.path.basename(bot_path)}")
        print(f"    Moving:    {os.path.basename(top_path)}")

        aligned, inv_map, flow_cleaned, ref_img, mov_img = align_pair(
            bot_path, top_path, args
        )

        valid_flow = np.sum(~np.isnan(flow_cleaned[0]))
        print(f"    Valid flow vectors: {valid_flow}")
        print(f"    Aligned shape: {aligned.shape}")

        results.append({
            'slab_n': slab_n,
            'slab_n1': slab_n1,
            'aligned': aligned,
            'inv_map': inv_map,
            'flow': flow_cleaned,
            'ref': ref_img,
            'mov': mov_img
        })

        # Collect faces for stack output
        if args.output_stack:
            # Add bot face (reference, unchanged)
            if i == 0 or all_faces[-1][0] != slab_n:
                all_faces.append((slab_n, 'bot', ref_img))
            # Add aligned top face
            all_faces.append((slab_n1, 'top', aligned))

        # Save individual outputs
        prefix = f"slab_{slab_n:03d}_bot_to_{slab_n1:03d}_top"

        if args.output_aligned:
            path = os.path.join(args.output_dir, f"{prefix}_aligned.zarr")
            save_image(path, aligned.astype(np.uint8))
            print(f"    Saved: {path}")

        if args.output_maps:
            path = os.path.join(args.output_dir, f"{prefix}_inv_map.zarr")
            save_image(path, inv_map.astype(np.float32))
            print(f"    Saved: {path}")

        if args.output_flow:
            path = os.path.join(args.output_dir, f"{prefix}_flow.zarr")
            save_image(path, flow_cleaned.astype(np.float32))
            print(f"    Saved: {path}")

        if args.visualize:
            import matplotlib.pyplot as plt

            # Overlay comparison
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            overlay_before = np.stack([ref_img/255, mov_img/255, ref_img/255], axis=-1)
            overlay_after = np.stack([ref_img/255, aligned/255, ref_img/255], axis=-1)
            ax[0].imshow(np.clip(overlay_before, 0, 1))
            ax[0].set_title(f'Before: {slab_n:03d}-bot (magenta) vs {slab_n1:03d}-top (green)')
            ax[0].axis('off')
            ax[1].imshow(np.clip(overlay_after, 0, 1))
            ax[1].set_title(f'After: {slab_n:03d}-bot (magenta) vs {slab_n1:03d}-top aligned (green)')
            ax[1].axis('off')
            path = os.path.join(args.output_dir, f"{prefix}_overlay.png")
            save_figure(fig, path)

    # Save combined stack
    if args.output_stack:
        print(f"\n[{len(pairs)+2}] Saving combined stack...")
        # Sort by slab number and face type (bot before top for same slab)
        all_faces.sort(key=lambda x: (x[0], 0 if x[1] == 'bot' else 1))

        stack = np.stack([img for _, _, img in all_faces], axis=0)
        path = os.path.join(args.output_dir, "aligned_stack.zarr")
        save_image(path, stack.astype(np.uint8))
        print(f"    Stack shape: {stack.shape}")
        print(f"    Saved: {path}")

        # Save metadata about stack ordering
        meta_path = os.path.join(args.output_dir, "stack_order.txt")
        with open(meta_path, 'w') as f:
            f.write("# Stack slice ordering (z=0 is first)\n")
            for i, (slab, face, _) in enumerate(all_faces):
                f.write(f"{i}: slab_{slab:03d}_{face}\n")
        print(f"    Saved: {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nProcessed {len(pairs)} pairs")
    print(f"Results saved to: {args.output_dir}")

    if args.output_aligned:
        print(f"  - Aligned images: *_aligned.zarr")
    if args.output_maps:
        print(f"  - Inverse maps: *_inv_map.zarr")
    if args.output_flow:
        print(f"  - Flow fields: *_flow.zarr")
    if args.output_stack:
        print(f"  - Combined stack: aligned_stack.zarr")
    if args.visualize:
        print(f"  - Visualizations: *_overlay.png")


if __name__ == "__main__":
    main()
