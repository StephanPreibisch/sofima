#!/usr/bin/env python
"""
SOFIMA Shared Utilities

Common functions for SOFIMA alignment scripts (align_pair.py, align_stack.py).
"""

import argparse
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import zarr
import tifffile
import h5py

from sofima import flow_field


def load_image(path):
    """Load image from ZARR or TIFF.

    Args:
        path: Path to ZARR directory or TIFF file

    Returns:
        numpy array of image data
    """
    if path.endswith('.zarr'):
        return np.array(zarr.open(path)[:])
    elif path.endswith(('.tif', '.tiff')):
        return tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


def load_h5_mipmap(path, mipmap_level=0, group='0-0-0'):
    """Load image from HDF5 file with mipmap pyramid.

    Args:
        path: Path to HDF5 file
        mipmap_level: Mipmap level to load (0=full res, 1=2x down, 2=4x down, etc.)
        group: HDF5 group containing mipmaps (default: '0-0-0')

    Returns:
        numpy array of image data (2D: [y, x])
    """
    with h5py.File(path, 'r') as f:
        dataset_name = f'{group}/mipmap.{mipmap_level}'
        if dataset_name not in f:
            available = [k for k in f[group].keys() if k.startswith('mipmap.')]
            raise ValueError(f"Mipmap level {mipmap_level} not found. Available: {available}")
        data = f[dataset_name][:]
        # Remove z dimension if present (shape is typically [1, y, x])
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        return data


def get_h5_mipmap_info(path, group='0-0-0'):
    """Get information about available mipmap levels in an HDF5 file.

    Args:
        path: Path to HDF5 file
        group: HDF5 group containing mipmaps

    Returns:
        dict: {level: {'shape': (y, x), 'scale': factor}}
    """
    info = {}
    with h5py.File(path, 'r') as f:
        for key in f[group].keys():
            if key.startswith('mipmap.'):
                level = int(key.split('.')[1])
                shape = f[f'{group}/{key}'].shape
                # Remove z dimension for display
                if len(shape) == 3 and shape[0] == 1:
                    shape = shape[1:]
                info[level] = {
                    'shape': shape,
                    'scale': 2 ** level
                }
    return info


def save_image(path, data):
    """Save image to ZARR (v2) or TIFF.

    Args:
        path: Output path (ZARR directory or TIFF file)
        data: numpy array to save
    """
    if path.endswith('.zarr'):
        if os.path.exists(path):
            shutil.rmtree(path)
        zarr.save(path, data, zarr_format=2)
    elif path.endswith(('.tif', '.tiff')):
        tifffile.imwrite(path, data)
    else:
        raise ValueError(f"Unsupported format: {path}")


def save_figure(fig, path):
    """Save matplotlib figure and close it.

    Args:
        fig: matplotlib figure
        path: Output path for PNG
    """
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def compute_flow_single(ref, mov, patch_size, stride, batch_size=256):
    """Compute flow field between two images.

    Args:
        ref: Reference image [y, x]
        mov: Moving image [y, x]
        patch_size: Size of correlation patches
        stride: Spacing between flow vectors
        batch_size: Number of patches to process in parallel

    Returns:
        flow: Flow field [2, y, x] with channels [x_disp, y_disp]
              Also contains quality statistics (peak_ratio, sharpness, etc.)
    """
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flow = mfc.flow_field(ref, mov, (patch_size, patch_size), (stride, stride), batch_size=batch_size)
    return flow


def downsample_image(img, factor):
    """Downsample image by averaging blocks.

    Args:
        img: Input image [y, x] or [z, y, x]
        factor: Downsample factor (2 = half size)

    Returns:
        Downsampled image
    """
    from skimage.transform import downscale_local_mean

    if img.ndim == 2:
        return downscale_local_mean(img, (factor, factor))
    elif img.ndim == 3:
        return downscale_local_mean(img, (1, factor, factor))
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")


def add_device_args(parser):
    """Add device selection arguments to parser.

    Args:
        parser: argparse.ArgumentParser
    """
    group = parser.add_argument_group('Device Selection')
    group.add_argument('--list-devices', action='store_true',
                       help='List available JAX devices and exit')
    group.add_argument('--device',
                       help='JAX device to use (e.g., "cpu", "gpu", "gpu:0", "tpu"). '
                            'Default: auto-select best available (GPU/TPU over CPU)')


def add_flow_args(parser):
    """Add flow computation arguments to parser.

    Args:
        parser: argparse.ArgumentParser
    """
    group = parser.add_argument_group('Flow Computation',
        'Parameters for optical flow estimation via masked cross-correlation.')
    group.add_argument('--patch-size', type=int, default=160,
                       help='Size of patches for cross-correlation (pixels). Larger patches are more '
                            'robust but capture less local detail. Usually keep similar across scales. '
                            '(default: 160)')
    group.add_argument('--stride', type=int, default=40,
                       help='Spacing between flow vectors (pixels). Smaller stride = denser flow field '
                            'but slower computation. Typically patch_size/4. Usually keep similar across '
                            'scales. (default: 40)')
    group.add_argument('--batch-size', type=int, default=256,
                       help='Number of patches to process in parallel on GPU. Reduce if running '
                            'out of GPU memory. (default: 256)')


def add_cleaning_args(parser):
    """Add flow cleaning arguments to parser.

    Args:
        parser: argparse.ArgumentParser
    """
    group = parser.add_argument_group('Flow Cleaning',
        'Parameters to filter out unreliable flow vectors. Vectors failing any criterion are set to NaN.')
    group.add_argument('--min-peak-ratio', type=float, default=1.6,
                       help='Minimum ratio of best to second-best correlation peak. Higher values '
                            'require more distinct matches. Set lower (1.2-1.4) for well-aligned '
                            'images. Scale-independent. (default: 1.6)')
    group.add_argument('--min-peak-sharpness', type=float, default=1.6,
                       help='Minimum sharpness of correlation peak. Higher values require sharper '
                            'peaks. Set lower (1.2-1.4) for smooth/low-contrast regions. Scale-independent. '
                            '(default: 1.6)')
    group.add_argument('--max-magnitude', type=float, default=80,
                       help='Maximum allowed flow magnitude (pixels). Vectors exceeding this are '
                            'rejected as outliers. SCALE-DEPENDENT: divide by 2 for each downsample level. '
                            '(default: 80)')
    group.add_argument('--max-deviation', type=float, default=20,
                       help='Maximum deviation from local median flow (pixels). Filters spatially '
                            'inconsistent vectors. SCALE-DEPENDENT: divide by 2 for each downsample level. '
                            '(default: 20)')


def add_mesh_args(parser):
    """Add mesh relaxation arguments to parser.

    Args:
        parser: argparse.ArgumentParser
    """
    group = parser.add_argument_group('Mesh Relaxation',
        'Parameters for elastic mesh optimization using FIRE (Fast Inertial Relaxation Engine). '
        'The mesh treats flow vectors as springs connecting sections.')
    group.add_argument('--k0', type=float, default=0.01,
                       help='Inter-section spring constant. Controls how strongly the mesh follows '
                            'the flow vectors. Higher = more faithful to flow, lower = smoother. '
                            '(default: 0.01)')
    group.add_argument('--k', type=float, default=0.1,
                       help='Intra-section spring constant. Controls mesh stiffness/rigidity. '
                            'Higher = more rigid (less local deformation), lower = more elastic. '
                            '(default: 0.1)')
    group.add_argument('--dt', type=float, default=0.001,
                       help='Initial integration timestep for FIRE optimizer. (default: 0.001)')
    group.add_argument('--gamma', type=float, default=0.0,
                       help='Damping coefficient. Usually 0 for FIRE. (default: 0.0)')
    group.add_argument('--num-iters', type=int, default=1000,
                       help='Iterations between convergence checks. Higher = fewer checks, more '
                            'efficient for large problems. (default: 1000)')
    group.add_argument('--max-iters', type=int, default=100000,
                       help='Maximum total iterations before giving up. (default: 100000)')
    group.add_argument('--stop-v-max', type=float, default=0.005,
                       help='Convergence threshold: max velocity. Optimization stops when all '
                            'mesh nodes move slower than this. (default: 0.005)')
    group.add_argument('--dt-max', type=float, default=1000,
                       help='Maximum timestep (adaptive). FIRE increases dt when making progress. '
                            '(default: 1000)')
    group.add_argument('--start-cap', type=float, default=0.01,
                       help='Initial cap on displacement per iteration. Prevents instability at '
                            'start. (default: 0.01)')
    group.add_argument('--final-cap', type=float, default=10,
                       help='Final cap on displacement per iteration. (default: 10)')
    group.add_argument('--prefer-orig-order', action='store_true', default=True,
                       help='Prefer original section ordering in optimization. (default: True)')


def handle_device_args(args, parser):
    """Handle device listing and selection.

    Args:
        args: Parsed arguments
        parser: ArgumentParser (for error messages)

    Returns:
        True if should continue, exits if --list-devices
    """
    import sys
    import jax

    if args.list_devices:
        print("Available JAX devices:")
        for i, d in enumerate(jax.devices()):
            default_marker = " (default)" if i == 0 else ""
            print(f"  [{i}] {d.device_kind} ({d.platform}){default_marker}")
        sys.exit(0)

    if args.device:
        try:
            jax.config.update("jax_default_device", args.device)
        except Exception as e:
            print(f"Warning: Could not set device '{args.device}': {e}")
            print("Continuing with default device...")

    return True


def create_mesh_config(args, stride):
    """Create mesh IntegrationConfig from arguments.

    Args:
        args: Parsed arguments with mesh parameters
        stride: Stride value for mesh

    Returns:
        mesh.IntegrationConfig
    """
    from sofima import mesh

    return mesh.IntegrationConfig(
        dt=args.dt,
        gamma=args.gamma,
        k0=args.k0,
        k=args.k,
        stride=(stride, stride),
        num_iters=args.num_iters,
        max_iters=args.max_iters,
        stop_v_max=args.stop_v_max,
        dt_max=args.dt_max,
        start_cap=args.start_cap,
        final_cap=args.final_cap,
        prefer_orig_order=args.prefer_orig_order
    )
