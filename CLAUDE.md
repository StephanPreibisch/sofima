# SOFIMA Alignment Tools

Command-line tools for image alignment using SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment).

**Scripts:**
- `align_pair.py` - Pairwise 2D alignment (two images)
- `align_stack.py` - Multi-slice stack alignment (N consecutive sections)
- `align_slabs.py` - Multi-SEM slab alignment (bot-top face pairs)
- `sofima_utils.py` - Shared utilities

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

## Multi-Slice Tool: align_stack.py

Aligns multiple consecutive slices with cumulative map composition and optional multi-resolution flow reconciliation.

### Input Options
```bash
# Single 3D file
--input stack.zarr

# Glob pattern for 2D files
--input-pattern "slices/z*.zarr"

# Explicit list
--input-list z0.zarr z1.zarr z2.zarr

# HDF5 files with multiresolution pyramid
--input-pattern "fibsem/*.h5" --mipmap 3
```

### HDF5 Support
For HDF5 files containing multiresolution pyramids (e.g., FIB-SEM data):

```bash
# List available mipmap levels
pixi run python align_stack.py --input-pattern "fibsem/*.h5" --list-mipmaps

# Load at specific mipmap level (0=full, 3=8x downsampled)
pixi run python align_stack.py --input-pattern "fibsem/*.h5" --mipmap 3 --output aligned.zarr
```

HDF5 format expected:
- Group: `0-0-0` (configurable via `--h5-group`)
- Datasets: `mipmap.0`, `mipmap.1`, `mipmap.2`, ... (2^N downsampling)
- Shape: `[1, height, width]` or `[height, width]`

### Basic Usage
```bash
pixi run python align_stack.py --input stack.zarr --output aligned.zarr
```

### Multi-Resolution (more robust)
```bash
pixi run python align_stack.py --input stack.zarr --output aligned.zarr --downsample-factor 2
```

### Skip Multi-Resolution (faster)
```bash
pixi run python align_stack.py --input stack.zarr --output aligned.zarr --skip-multi-res
```

### Key Differences from align_pair.py
1. **Cumulative map composition**: Each section's deformation builds on all previous sections
2. **Multi-resolution reconciliation**: Computes flows at multiple resolutions for robustness
3. **Stack input/output**: Handles 3D volumes or sequences of 2D files

### Additional Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--downsample-factor` | 2 | Factor for second resolution level |
| `--skip-multi-res` | False | Skip multi-resolution reconciliation |
| `--reconcile-max-gradient` | 0 | Max gradient for reconciliation |
| `--reconcile-min-patch-size` | 400 | Min patch size for reconciliation |

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

## Multi-SEM Slab Alignment: align_slabs.py

Aligns consecutive slab face pairs for multi-SEM data. For each pair: slab N's bot-face aligns with slab (N+1)'s top-face.

### Pattern Syntax
```bash
--pattern "path/to/flat-*-w61_s{slab}_r00-{face}-face.zarr"
```

Placeholders:
- `{slab}` - Slab number (e.g., 079, 080, 081)
- `{face}` - Face type (automatically replaced with `bot` or `top`)
- `*` - Standard glob wildcard (matches any characters)

### Basic Usage
```bash
# Align all discovered pairs, save aligned images
pixi run python align_slabs.py \
    --pattern "./pass03-scale5/flat-*-w61_s{slab}_r00-{face}-face.zarr" \
    --output-dir ./results \
    --output-aligned

# Process specific slab range
pixi run python align_slabs.py \
    --pattern "./data/flat-*-w61_s{slab}_r00-{face}-face.zarr" \
    --slabs 079-083 \
    --output-dir ./results \
    --output-aligned
```

### Output Options
| Flag | Description |
|------|-------------|
| `--output-aligned` | Save warped (aligned) images per pair |
| `--output-maps` | Save inverse coordinate maps per pair |
| `--output-flow` | Save cleaned flow fields per pair |
| `--output-stack` | Save all faces as single 3D stack |
| `--visualize` / `-v` | Save overlay comparison PNGs |

### Output Files
For each pair (e.g., 079-bot → 080-top):
- `slab_079_bot_to_080_top_aligned.zarr` - Warped top face
- `slab_079_bot_to_080_top_inv_map.zarr` - Inverse map
- `slab_079_bot_to_080_top_flow.zarr` - Cleaned flow
- `slab_079_bot_to_080_top_overlay.png` - Visualization

If `--output-stack`:
- `aligned_stack.zarr` - All faces combined (3D)
- `stack_order.txt` - Slice ordering metadata

### Alignment Pairs
The script automatically discovers and aligns consecutive pairs:
```
079-bot → 080-top
080-bot → 081-top
081-bot → 082-top
...
```
Bot-face is the reference (unchanged), top-face is warped to match.

## Notes

- ZARR output uses v2 format for compatibility
- Reference image defines target coordinate space
- Moving image is warped to match reference
- For multi-SEM slab alignment: bot(N) = reference, top(N+1) = moving
