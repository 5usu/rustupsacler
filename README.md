# Rust Upscaler (Rust + ONNX Runtime)

High-quality 4? image upscaling powered by ONNX Runtime, written in Rust. Supports batched tile inference for speed, automatic layout detection (NCHW/NHWC/HWC/CHW), and optional CUDA acceleration.

---

## Features
- **4? super-resolution**: Works out-of-the-box with Real-ESRGAN x4-style models
- **Fast batched tiling**: Packs tiles into a single batch; gracefully falls back to sequential
- **CPU by default, optional GPU**: Enable CUDA with a Cargo feature flag
- **Robust output handling**: Auto-detects output layout and scale factor
- **Simple CLI**: Sensible defaults, progress + timing logs
- **Model helper**: Python tool to make batch dims dynamic for ONNX models

---

## Quick Start

```bash
# Build (CPU)
cargo build --release

# Build (GPU via CUDA)
cargo build --release --features gpu

# Run (defaults shown)
./target/release/upscaler input.jpg \
  -o output_x4.png \
  -m models/realesrgan_x4.onnx \
  --tile 128 \
  --warmup 1
```

- If `-o` is omitted, output becomes `<stem>_x4.png` next to the input.
- If the GPU build fails to start CUDA, re-run with `--cpu` to force CPU.

---

## Installation

### Prerequisites
- **Rust** (edition 2024; install via `rustup`)
- **ONNX Runtime binaries** are auto-downloaded by the `ort` crate
- For GPU builds: **CUDA** runtime compatible with ONNX Runtime CUDA EP

### Build
```bash
# CPU (default)
cargo build --release

# GPU (strict CUDA only; will error if CUDA cannot initialize)
cargo build --release --features gpu
```

Binary is produced at `target/release/upscaler`.

---

## CLI Usage

```bash
upscaler <INPUT> [OPTIONS]
```

**Options**
- `-o, --output <PATH>`: Output path (PNG). Default: `<stem>_x4.png`
- `-m, --model <PATH>`: ONNX model path. Default: `models/realesrgan_x4.onnx`
- `--tile <PX>`: Tile size (both width and height). Default: `128`
- `--warmup <N>`: Number of warmup runs before timing. Default: `1`
- `--cpu`: Force CPU even if built with `gpu` feature
- `--no-batch`: Disable batched inference, run tiles sequentially

The program prints input/output sizes and a timing summary:
- prep | warmup | infer | stitch | save | mode

---

## Models

The default expects a 4? RGB super-resolution ONNX model. Real-ESRGAN x4 models are a good fit.

- Example: Real-ESRGAN x4 ONNX (community sources available; ensure NCHW/NHWC compatibility)
- Place your model at `models/realesrgan_x4.onnx` or pass `-m /path/to/model.onnx`

### Dynamic batch helper (optional)
Some models use fixed batch sizes or omit a batch dimension. Use the included helper to make batch dims dynamic and add a batch dimension to rank-3 outputs when needed.

```bash
# Requires Python and onnx
pip install onnx

# Make batch dimension dynamic (adds N to rank-3 outputs)
python make_dynamic_batch.py input.onnx output_dynamic.onnx
```

---

## Tips for Quality & Performance
- **Tile size**: Larger tiles reduce stitching overhead, but require more memory. Start with `--tile 128` and increase as memory permits.
- **Warmup**: Increase `--warmup` to stabilize timings on some backends.
- **Batch vs sequential**: Batched mode is on by default. If your model does not accept dynamic batching, the tool will automatically fall back to sequential or you can force `--no-batch`.
- **CPU vs GPU**: CPU is robust and portable. For maximum speed, build with `--features gpu` and ensure a working CUDA setup.

---

## Examples

```bash
# Basic (CPU)
upscaler input.jpg

# Explicit output and model
upscaler input.jpg -o results/cat_x4.png -m models/realesrgan_x4.onnx

# Larger tiles, more stable timings
upscaler input.png --tile 192 --warmup 2

# GPU build, but force CPU fallback
upscaler input.jpg --cpu

# Disable batching if your model insists on single input
upscaler input.jpg --no-batch
```

---

## How it works (brief)
- Pads the image to a multiple of `--tile`, then splits into tiles
- Packs tiles into a single 4D tensor for batched inference when possible
- Detects output layout (NCHW/NHWC/HWC/CHW) and scale from output size
- Stitches tiles back together and crops to the exact upscaled size

---

## Troubleshooting
- "CUDA initialization failed" ? Rebuild without `gpu` or run with `--cpu`.
- Model shape mismatch ? Run the dynamic batch helper or disable batching via `--no-batch`.
- Out-of-memory ? Reduce `--tile`.

---

## License
This repository did not declare a license text at the time this README was generated. Add one if you intend to distribute binaries or source.
