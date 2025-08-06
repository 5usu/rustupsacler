use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageFormat, RgbImage};
use ndarray::Array;
use ort::{session::Session, value::Tensor};
use ort::session::builder::GraphOptimizationLevel;

#[derive(Parser, Debug)]
#[command(version, about = "4x upscaler (batched) with ONNX Runtime (strict CUDA if built with --features gpu)")]
struct Cli {
    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(short='m', long, env = "ESRGAN_MODEL", default_value = "models/realesrgan_x4.onnx")]
    model: PathBuf,

    #[arg(long, default_value_t = 128)]
    tile: u32,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    no_batch: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let in_path = &args.input;
    let out_path = args.output.clone().unwrap_or_else(|| {
        let stem = in_path.file_stem().unwrap_or_default().to_string_lossy();
        in_path.with_file_name(format!("{stem}_x4.png"))
    });

    let t0 = Instant::now();
    let img = image::open(&args.input)
        .with_context(|| format!("failed to read {:?}", &args.input))?;
    let (w, h) = img.dimensions();
    println!("Input: {:?}  [{}x{}]", &args.input, w, h);

    let mut session = build_session(&args.model, args.cpu)?;

    let tile = args.tile;
    let pw = ((w + tile - 1) / tile) * tile;
    let ph = ((h + tile - 1) / tile) * tile;

    let mut padded = RgbImage::new(pw, ph);
    let rgb = img.to_rgb8();
    for y in 0..h {
        for x in 0..w {
            padded.put_pixel(x, y, *rgb.get_pixel(x, y));
        }
    }

    let tiles_x = (pw / tile) as usize;
    let tiles_y = (ph / tile) as usize;
    let tiles_t = tiles_x * tiles_y;

    let mut batch_vec = Vec::<f32>::with_capacity(tiles_t * 3 * tile as usize * tile as usize);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = (tx as u32) * tile;
            let y0 = (ty as u32) * tile;
            // R plane
            for y in 0..tile {
                for x in 0..tile {
                    batch_vec.push(padded.get_pixel(x0 + x, y0 + y)[0] as f32 / 255.0);
                }
            }
            // G plane
            for y in 0..tile {
                for x in 0..tile {
                    batch_vec.push(padded.get_pixel(x0 + x, y0 + y)[1] as f32 / 255.0);
                }
            }
            // B plane
            for y in 0..tile {
                for x in 0..tile {
                    batch_vec.push(padded.get_pixel(x0 + x, y0 + y)[2] as f32 / 255.0);
                }
            }
        }
    }
    let prep_done = Instant::now();

    let batched_arr =
        Array::from_shape_vec((tiles_t, 3usize, tile as usize, tile as usize), batch_vec)?;
    let batched_tensor = map_ort(Tensor::from_array(batched_arr))?;

    for _ in 0..args.warmup {
        let _ = map_ort(session.run(ort::inputs![batched_tensor.clone()]))?;
    }

let infer_start = Instant::now();

let attempted_batched: Option<(Vec<RgbImage>, u32, String)> = if args.no_batch {
    None
} else {
    match session.run(ort::inputs![batched_tensor.clone()]) {
        Ok(outs) => Some(decode_batched_output(outs, tiles_t, tile)?),
        Err(_) => None,
    }
};

let (tile_outputs, scale, layout) = if let Some(res) = attempted_batched {
    res
} else {
    sequential_tiles(&mut session, &padded, tile, tiles_x, tiles_y)?
};
let infer_done = Instant::now();

    let ow_tile = tile * scale;
    let oh_tile = tile * scale;
    let mut out_full = RgbImage::new(pw * scale, ph * scale);
    let mut idx = 0usize;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            blit(
                &mut out_full,
                &tile_outputs[idx],
                (tx as u32) * ow_tile,
                (ty as u32) * oh_tile,
            );
            idx += 1;
        }
    }
    let stitch_done = Instant::now();

    let final_w = w * scale;
    let final_h = h * scale;
    let out = crop_rgb(&out_full, 0, 0, final_w, final_h);

    let dyn_img = DynamicImage::ImageRgb8(out);
    let f = File::create(&out_path).with_context(|| format!("open {}", out_path.display()))?;
    let mut writer = BufWriter::new(f);
    dyn_img
        .write_to(&mut writer, ImageFormat::Png)
        .with_context(|| format!("write PNG to {}", out_path.display()))?;
    let save_done = Instant::now();

    println!("Model: {:?}", &args.model);
    println!(
        "Timing: prep {:.3?} | warmup x{} | infer {:.3?} | stitch {:.3?} | save {:.3?} | mode {}",
        prep_done - t0,
        args.warmup,
        infer_done - infer_start,
        stitch_done - infer_done,
        save_done - stitch_done,
        layout
    );
    println!("Output: {:?}  [{}x{}]", &out_path, final_w, final_h);
    Ok(())
}


fn map_ort<T>(r: ort::Result<T>) -> anyhow::Result<T> {
    r.map_err(|e| anyhow!(e.to_string()))
}

#[cfg(feature = "gpu")]
fn build_session(model_path: &Path, force_cpu: bool) -> Result<Session> {
    use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

    let mut builder = map_ort(Session::builder())?;
    builder = map_ort(builder.with_intra_threads(num_cpus::get()))?;
    builder = map_ort(builder.with_optimization_level(GraphOptimizationLevel::Level3))?;

    if force_cpu {
        builder = map_ort(builder.with_execution_providers([CPUExecutionProvider::default().build()]))?;
    } else {
        // STRICT CUDA only: will error if CUDA cannot initialize
        builder = map_ort(builder.with_execution_providers([
            CUDAExecutionProvider::default().with_device_id(0).build(),
        ]))?;
    }

    let model_bytes =
        std::fs::read(model_path).with_context(|| format!("reading model at {}", model_path.display()))?;
    map_ort(builder.commit_from_memory(&model_bytes))
}

#[cfg(not(feature = "gpu"))]
fn build_session(model_path: &Path, _force_cpu: bool) -> Result<Session> {
    use ort::execution_providers::CPUExecutionProvider;

    let mut builder = map_ort(Session::builder())?;
    builder = map_ort(builder.with_intra_threads(num_cpus::get()))?;
    builder = map_ort(builder.with_optimization_level(GraphOptimizationLevel::Level3))?;
    builder = map_ort(builder.with_execution_providers([CPUExecutionProvider::default().build()]))?;

    let model_bytes =
        std::fs::read(model_path).with_context(|| format!("reading model at {}", model_path.display()))?;
    map_ort(builder.commit_from_memory(&model_bytes))
}

fn decode_batched_output(
    outs: ort::session::SessionOutputs<'_>,
    tiles_t: usize,
    tile: u32,
) -> Result<(Vec<RgbImage>, u32, String)> {
    let view = map_ort(outs[0].try_extract_array::<f32>())?;
    let shape = view.shape();

    let (t, oh, ow, lay) = match shape.len() {
        4 if shape[0] == tiles_t && shape[1] == 3 => (shape[0], shape[2] as u32, shape[3] as u32, "NCHW"),
        4 if shape[0] == tiles_t && shape[3] == 3 => (shape[0], shape[1] as u32, shape[2] as u32, "NHWC"),
        3 if tiles_t == 1 && shape[0] == 3 => (1, shape[1] as u32, shape[2] as u32, "CHW"),
        3 if tiles_t == 1 && shape[2] == 3 => (1, shape[0] as u32, shape[1] as u32, "HWC"),
        _ => return Err(anyhow!("unexpected batched output shape: {:?}", shape)),
    };

    let s = (ow / tile).max(oh / tile).max(1);
    let data = view.as_slice().ok_or_else(|| anyhow!("non-contiguous output"))?;
    let mut outs_imgs = Vec::<RgbImage>::with_capacity(t);

    match lay {
        "NCHW" => {
            let plane = (oh * ow) as usize;
            let stride_t = 3 * plane;
            for ti in 0..t {
                let base = ti * stride_t;
                let mut img = RgbImage::new(ow, oh);
                for y in 0..oh {
                    for x in 0..ow {
                        let idx = (y * ow + x) as usize;
                        let r = (data[base + 0 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                        let g = (data[base + 1 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (data[base + 2 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                        img.put_pixel(x, y, image::Rgb([r, g, b]));
                    }
                }
                outs_imgs.push(img);
            }
        }
        "NHWC" => {
            let stride_t = (oh * ow * 3) as usize;
            for ti in 0..t {
                let base = ti * stride_t;
                let mut img = RgbImage::new(ow, oh);
                for y in 0..oh {
                    for x in 0..ow {
                        let b0 = base + ((y * ow + x) * 3) as usize;
                        let r = (data[b0 + 0].clamp(0.0, 1.0) * 255.0) as u8;
                        let g = (data[b0 + 1].clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (data[b0 + 2].clamp(0.0, 1.0) * 255.0) as u8;
                        img.put_pixel(x, y, image::Rgb([r, g, b]));
                    }
                }
                outs_imgs.push(img);
            }
        }
        "CHW" | "HWC" => {
            // Single-tile squeezed; T==1
            let img = decode_single(layout_from_3d(lay), data, oh, ow)?;
            outs_imgs.push(img);
        }
        _ => unreachable!(),
    }

    Ok((outs_imgs, s, lay.to_string()))
}

fn sequential_tiles(
    session: &mut Session,
    padded: &RgbImage,
    tile: u32,
    tiles_x: usize,
    tiles_y: usize,
) -> Result<(Vec<RgbImage>, u32, String)> {
    let mut outs = Vec::<RgbImage>::with_capacity(tiles_x * tiles_y);
    let mut detected_scale = 0u32;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = (tx as u32) * tile;
            let y0 = (ty as u32) * tile;
            let tile_img = crop_rgb(padded, x0, y0, tile, tile);
            let (img_out, s) = run_one_tile(session, &tile_img)?;
            if detected_scale == 0 {
                detected_scale = s;
            }
            outs.push(img_out);
        }
    }
    Ok((outs, detected_scale, "sequential".to_string()))
}

fn crop_rgb(src: &RgbImage, x: u32, y: u32, w: u32, h: u32) -> RgbImage {
    let mut out = RgbImage::new(w, h);
    for yy in 0..h {
        for xx in 0..w {
            out.put_pixel(xx, yy, *src.get_pixel(x + xx, y + yy));
        }
    }
    out
}

fn blit(dst: &mut RgbImage, patch: &RgbImage, x: u32, y: u32) {
    for yy in 0..patch.height() {
        for xx in 0..patch.width() {
            dst.put_pixel(x + xx, y + yy, *patch.get_pixel(xx, yy));
        }
    }
}

fn run_one_tile(session: &mut Session, tile_rgb: &RgbImage) -> Result<(RgbImage, u32)> {
    let (tw, th) = (tile_rgb.width(), tile_rgb.height());

    let mut input_vec = Vec::<f32>::with_capacity((3 * tw * th) as usize);
    for c in 0..3 {
        for y in 0..th {
            for x in 0..tw {
                input_vec.push(tile_rgb.get_pixel(x, y)[c] as f32 / 255.0);
            }
        }
    }
    let input_arr = Array::from_shape_vec((1usize, 3usize, th as usize, tw as usize), input_vec)?;
    let input_tensor = map_ort(Tensor::from_array(input_arr))?;

    let outs = map_ort(session.run(ort::inputs![input_tensor]))?;

    let view = map_ort(outs[0].try_extract_array::<f32>())?;
    let shape = view.shape();
    let data = view.as_slice().ok_or_else(|| anyhow!("non-contiguous output"))?;

    let (oh, ow, lay) = match shape.len() {
        4 if shape[1] == 3 => (shape[2] as u32, shape[3] as u32, "NCHW"),
        4 if shape[3] == 3 => (shape[1] as u32, shape[2] as u32, "NHWC"),
        3 if shape[0] == 3 => (shape[1] as u32, shape[2] as u32, "CHW"),
        3 if shape[2] == 3 => (shape[0] as u32, shape[1] as u32, "HWC"),
        _ => return Err(anyhow!("unexpected per-tile output shape: {:?}", shape)),
    };
    let scale = (ow / tw).max(oh / th).max(1);

    let mut out_img = RgbImage::new(ow, oh);
    match lay {
        "NCHW" | "CHW" => {
            let plane = (oh * ow) as usize;
            for y in 0..oh {
                for x in 0..ow {
                    let idx = (y * ow + x) as usize;
                    let r = (data[0 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (data[1 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (data[2 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                    out_img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
        }
        "NHWC" | "HWC" => {
            for y in 0..oh {
                for x in 0..ow {
                    let base = ((y * ow + x) * 3) as usize;
                    let r = (data[base + 0].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (data[base + 1].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (data[base + 2].clamp(0.0, 1.0) * 255.0) as u8;
                    out_img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
        }
        _ => unreachable!(),
    }

    Ok((out_img, scale))
}

fn layout_from_3d(l: &str) -> &str {
    match l {
        "CHW" => "NCHW",
        "HWC" => "NHWC",
        _ => l,
    }
}

fn decode_single(layout: &str, data: &[f32], oh: u32, ow: u32) -> Result<RgbImage> {
    let mut img = RgbImage::new(ow, oh);
    match layout {
        "NCHW" => {
            let plane = (oh * ow) as usize;
            for y in 0..oh {
                for x in 0..ow {
                    let idx = (y * ow + x) as usize;
                    let r = (data[0 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (data[1 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (data[2 * plane + idx].clamp(0.0, 1.0) * 255.0) as u8;
                    img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
        }
        "NHWC" => {
            for y in 0..oh {
                for x in 0..ow {
                    let base = ((y * ow + x) * 3) as usize;
                    let r = (data[base + 0].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (data[base + 1].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (data[base + 2].clamp(0.0, 1.0) * 255.0) as u8;
                    img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
        }
        _ => return Err(anyhow!("unexpected layout: {}", layout)),
    }
    Ok(img)
}
