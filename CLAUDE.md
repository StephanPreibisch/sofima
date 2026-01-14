# SOFIMA Pairwise Alignment Tools

This directory contains command-line tools for 2D pairwise image alignment using SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment).

## Setup

Uses Pixi package manager:
```bash
pixi install
pixi run python align_pair.py --help
```

For GPU support on Linux (CUDA), ensure JAX with CUDA is installed. Check devices:
```bash
pixi run python align_pair.py --list-devices
```

**Note (2025-01)**: `jax-metal` for Apple Silicon GPUs does not work with SOFIMA yet.
It fails with "UNIMPLEMENTED: default_memory_space is not supported". Use CPU on Macs for now.

## Main Tool: align_pair.py

Aligns two 2D images using optical flow and elastic mesh relaxation.

### Basic Usage
```bash
pixi run python align_pair.py reference.zarr moving.zarr --output aligned.zarr
```

### Key Outputs
- `--output` / `-o`: Aligned image (ZARR or TIFF)
- `--output-flow-raw`: Raw flow field before cleaning (for debugging)
- `--output-flow`: Cleaned flow field (NaN = invalid vectors)
- `--output-map`: Inverse coordinate map (what actually deforms the image)
- `--visualize` / `-v`: Save PNG visualizations

### Device Selection
```bash
--list-devices          # Show available JAX devices
--device cpu|gpu|gpu:0  # Select specific device (default: auto)
```

## Parameters

### Flow Computation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--patch-size` | 160 | Correlation patch size (pixels). Keep similar across scales. |
| `--stride` | 40 | Spacing between flow vectors. Typically patch_size/4. |
| `--batch-size` | 256 | GPU batch size. Reduce if out of memory. |

### Flow Cleaning
| Parameter | Default | Scale-dependent? | Description |
|-----------|---------|------------------|-------------|
| `--min-peak-ratio` | 1.6 | No | Min ratio of best/second-best peak. Lower (1.2-1.4) for well-aligned images. |
| `--min-peak-sharpness` | 1.6 | No | Min peak sharpness. Lower for smooth regions. |
| `--max-magnitude` | 80 | **YES** | Max flow magnitude. Divide by 2 per downsample level. |
| `--max-deviation` | 20 | **YES** | Max deviation from median. Divide by 2 per downsample level. |

### Mesh Relaxation (FIRE optimizer)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k0` | 0.01 | Inter-section spring constant. Higher = follow flow more closely. |
| `--k` | 0.1 | Intra-section spring constant. Higher = stiffer/more rigid mesh. |
| `--stop-v-max` | 0.005 | Convergence threshold (max velocity). |

## Scale Considerations

When running on downsampled images:
- **max-magnitude, max-deviation**: Divide by 2 for each 2x downsample
- **patch-size, stride**: Usually keep similar across scales

Example for 32x downsampled (scale 5):
```bash
--max-magnitude 2.5 --max-deviation 0.625  # 80/32 and 20/32
```

## SOFIMA Pipeline

1. **Flow computation**: Cross-correlation between patches
2. **Flow cleaning**: Filter unreliable vectors (→ NaN)
3. **Mesh relaxation**: Elastic optimization with spring model
4. **Map inversion**: Create inverse map for warping
5. **Warping**: Apply inverse map to deform moving image

## Output Shapes

- **Flow fields**: `[2, 1, y, x]` - channels are [x_displacement, y_displacement]
- **Inverse map**: `[2, 1, y, x]` - at stride resolution, not pixel resolution

## Real-World Example: Multi-SEM Slab Alignment

Aligning consecutive slab faces (bot of slab 79 to top of slab 80) at scale 5 (32x downsampled):

```bash
pixi run python align_pair.py \
    ./pass03-scale5/flat-w61_serial_070_to_079-w61_s079_r00-bot-face.zarr \
    ./pass03-scale5/flat-w61_serial_080_to_089-w61_s080_r00-top-face.zarr \
    --output ./pass03-scale5/aligned.zarr \
    --output-map ./pass03-scale5/inverse_map.zarr \
    --output-flow ./pass03-scale5/flow_cleaned.zarr \
    --output-flow-raw ./pass03-scale5/flow_raw.zarr \
    --visualize \
    --max-magnitude 2.5 \
    --max-deviation 0.625
```

Note: `--max-magnitude 2.5` and `--max-deviation 0.625` are scaled for 32x downsample (80/32 and 20/32).

## Example Scripts

- `em_alignment_example.py`: Full pipeline on Google Cloud sample data (200 sections)
- `em_alignment_test.py`: Test version with 20 sections and visualizations

## Notes

- ZARR output uses v2 format for compatibility
- Reference image defines target coordinate space
- Moving image is warped to match reference
- For multi-SEM slab alignment: bot(N) = reference, top(N+1) = moving
