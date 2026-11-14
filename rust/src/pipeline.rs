//! The whole engine for one burst: RAW (Bayer) or YUV in, JPEG bytes out.

use crate::align::{align, build_pyramid, pick_reference, LEVELS};
use crate::color::{apply_matrix, camera_to_srgb};
use crate::demosaic::{bin_quads, demosaic_mhc, Rgb};
use crate::encode::encode_jpeg;
use crate::finish::{finish, quick_srgb8, FinishInfo, FinishParams};
use crate::merge::{Merger, NoiseModel};
use crate::meta::Meta;
use crate::plane::{box_downsample, resize_bilinear, Plane};
use crate::raw::{apply_lsc, apply_wb_highlights, bin_grey, clip_weights, normalize_raw, refine_wb, wb_gains, Lsc};
use crate::tone::srgb_decode;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Processed {
    pub jpeg: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    /// Index of the frame used as the alignment reference (and saved as DNG).
    pub ref_index: usize,
    pub info: FinishInfo,
}

/// Above this many sensor pixels the image is quad-binned instead of demosaiced.
pub const DEFAULT_MAX_PIXELS: usize = 17_000_000;

/// Align and merge a burst of already normalised planes. `load(i)` returns frame i as a
/// Bayer/luma plane; `grey(i)` its binned grey (computed once per frame by the caller).
/// Returns the merged plane and the reference index.
pub fn merge_burst<F>(n: usize, load: F, greys: &[Plane], w: usize, h: usize, noise: &NoiseModel) -> Result<(Vec<f32>, usize), String>
where
    F: Fn(usize) -> Result<Vec<f32>, String>,
{
    let ref_idx = pick_reference(greys, 3);
    let refp = load(ref_idx)?;
    if n == 1 {
        return Ok((refp, ref_idx));
    }
    let ref_pyr = build_pyramid(&greys[ref_idx], LEVELS);
    let mut merger = Merger::new(w, h, 32);
    for k in (0..n).filter(|k| *k != ref_idx) {
        let alt = load(k)?;
        let offs = align(&ref_pyr, &build_pyramid(&greys[k], LEVELS));
        merger.add(&refp, &alt, &offs, 2, noise);
    }
    Ok((merger.finish(&refp), ref_idx))
}

fn check_dims(w: usize, h: usize) -> Result<(), String> {
    if w < 64 || h < 64 || w % 2 != 0 || h % 2 != 0 {
        return Err(format!("unsupported frame size {}x{}", w, h));
    }
    Ok(())
}

/// Merged, normalised Bayer to linear sRGB (stage 3).
pub fn develop_bayer(mut bayer: Vec<f32>, w: usize, h: usize, meta: &Meta, lsc: Option<&Lsc>, max_pixels: usize) -> Rgb {
    // Clipping is judged on the sensor values, before lens shading lifts the corners.
    let clip = clip_weights(&bayer, w, h);
    if let Some(l) = lsc {
        apply_lsc(&mut bayer, w, h, meta.cfa, l);
    }
    let base = wb_gains(meta.neutral);
    let gains = refine_wb(&bayer, w, h, meta.cfa, base);
    apply_wb_highlights(&mut bayer, w, meta.cfa, gains, &clip);
    let mut rgb = if w * h > max_pixels { bin_quads(&bayer, w, h, meta.cfa) } else { demosaic_mhc(&bayer, w, h, meta.cfa) };
    drop(bayer);
    apply_matrix(&mut rgb, &camera_to_srgb(meta));
    rgb
}

pub fn process_raw(frames: &[&[u8]], w: usize, h: usize, stride: usize, meta: &Meta, lsc: Option<&Lsc>, quality: u8, max_pixels: usize) -> Result<Processed, String> {
    check_dims(w, h)?;
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let n = frames.len();
    let mut greys = Vec::with_capacity(n);
    for f in frames {
        let norm = normalize_raw(f, w, h, stride, meta)?;
        greys.push(bin_grey(&norm, w, h));
    }
    let (merged, ref_index) = merge_burst(n, |i| normalize_raw(frames[i], w, h, stride, meta), &greys, w, h, &meta.noise)?;
    drop(greys);
    let rgb = develop_bayer(merged, w, h, meta, lsc, max_pixels);
    let (ow, oh) = (rgb.w, rgb.h);
    let params = FinishParams { scene: meta.scene, ev_gain: meta.ev_gain, noise: meta.noise, frames: n, display_referred: false };
    let (rgb8, info) = finish(rgb, &params);
    let jpeg = encode_jpeg(&rgb8, ow, oh, quality)?;
    Ok(Processed { jpeg, width: ow, height: oh, frames: n, ref_index, info })
}

/// A quick half-resolution render of a single RAW frame for the before/after view.
pub fn reference_raw(frame: &[u8], w: usize, h: usize, stride: usize, meta: &Meta, lsc: Option<&Lsc>, quality: u8) -> Result<(Vec<u8>, usize, usize), String> {
    check_dims(w, h)?;
    let mut bayer = normalize_raw(frame, w, h, stride, meta)?;
    let clip = clip_weights(&bayer, w, h);
    if let Some(l) = lsc {
        apply_lsc(&mut bayer, w, h, meta.cfa, l);
    }
    apply_wb_highlights(&mut bayer, w, meta.cfa, wb_gains(meta.neutral), &clip);
    let mut rgb = bin_quads(&bayer, w, h, meta.cfa);
    drop(bayer);
    apply_matrix(&mut rgb, &camera_to_srgb(meta));
    // The reference is what the sensor saw: show it at its own exposure, not the lifted one.
    let px = quick_srgb8(&rgb, 1.0);
    let jpeg = encode_jpeg(&px, rgb.w, rgb.h, quality)?;
    Ok((jpeg, rgb.w, rgb.h))
}

/// One YUV_420_888 frame as three borrowed planes.
pub struct YuvFrame<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct YuvLayout {
    pub w: usize,
    pub h: usize,
    pub y_row: usize,
    pub uv_row: usize,
    pub uv_pixel: usize,
}

fn read_y(f: &YuvFrame, l: &YuvLayout) -> Result<Vec<f32>, String> {
    if f.y.len() < l.y_row * (l.h - 1) + l.w {
        return Err("Y plane too small".into());
    }
    let mut out = vec![0.0f32; l.w * l.h];
    out.par_chunks_mut(l.w).enumerate().for_each(|(y, row)| {
        let src = &f.y[y * l.y_row..y * l.y_row + l.w];
        for (o, s) in row.iter_mut().zip(src) {
            *o = *s as f32 / 255.0;
        }
    });
    Ok(out)
}

fn read_chroma(plane: &[u8], l: &YuvLayout) -> Result<Vec<f32>, String> {
    let (cw, ch) = (l.w / 2, l.h / 2);
    if plane.len() < l.uv_row * (ch - 1) + l.uv_pixel * (cw - 1) + 1 {
        return Err("chroma plane too small".into());
    }
    let mut out = vec![0.0f32; cw * ch];
    out.par_chunks_mut(cw).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = (plane[y * l.uv_row + x * l.uv_pixel] as f32 - 128.0) / 255.0;
        }
    });
    Ok(out)
}

fn yuv_to_linear(y: &[f32], u: Plane, v: Plane, w: usize, h: usize) -> Rgb {
    let up = resize_bilinear(&u, w, h);
    let vp = resize_bilinear(&v, w, h);
    let mut rgb = Rgb::new(w, h);
    let [r, g, b] = &mut rgb.c;
    r.par_iter_mut()
        .zip(g.par_iter_mut().zip(b.par_iter_mut()))
        .enumerate()
        .for_each(|(i, (r, (g, b)))| {
            let (yy, cb, cr) = (y[i], up.d[i], vp.d[i]);
            *r = srgb_decode(yy + 1.402 * cr);
            *g = srgb_decode(yy - 0.344136 * cb - 0.714136 * cr);
            *b = srgb_decode(yy + 1.772 * cb);
        });
    rgb
}

pub fn process_yuv(frames: &[YuvFrame], layout: &YuvLayout, meta: &Meta, quality: u8) -> Result<Processed, String> {
    let (w, h) = (layout.w, layout.h);
    check_dims(w, h)?;
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let n = frames.len();
    let mut greys = Vec::with_capacity(n);
    for f in frames {
        let y = read_y(f, layout)?;
        greys.push(bin_grey(&y, w, h));
    }
    let noise = meta.noise;
    let chroma_noise = NoiseModel { s: noise.s * 2.0, o: noise.o * 2.0 };
    let (y_merged, ref_idx) = merge_burst(n, |i| read_y(&frames[i], layout), &greys, w, h, &noise)?;
    // Chroma planes use the same offsets as luma (scale 1 at half resolution).
    let (cw, ch) = (w / 2, h / 2);
    let ref_pyr = build_pyramid(&greys[ref_idx], LEVELS);
    let (u_ref, v_ref) = (read_chroma(frames[ref_idx].u, layout)?, read_chroma(frames[ref_idx].v, layout)?);
    let (mut mu, mut mv) = (Merger::new(cw, ch, 16), Merger::new(cw, ch, 16));
    for k in (0..n).filter(|k| *k != ref_idx) {
        let offs = align(&ref_pyr, &build_pyramid(&greys[k], LEVELS));
        mu.add(&u_ref, &read_chroma(frames[k].u, layout)?, &offs, 1, &chroma_noise);
        mv.add(&v_ref, &read_chroma(frames[k].v, layout)?, &offs, 1, &chroma_noise);
    }
    drop(greys);
    let (u, v) = (mu.finish(&u_ref), mv.finish(&v_ref));
    let rgb = yuv_to_linear(&y_merged, Plane::from_vec(cw, ch, u), Plane::from_vec(cw, ch, v), w, h);
    let params = FinishParams { scene: meta.scene, ev_gain: 1.0, noise, frames: n, display_referred: true };
    let (rgb8, info) = finish(rgb, &params);
    let jpeg = encode_jpeg(&rgb8, w, h, quality)?;
    Ok(Processed { jpeg, width: w, height: h, frames: n, ref_index: ref_idx, info })
}

pub fn reference_yuv(frame: &YuvFrame, layout: &YuvLayout, quality: u8) -> Result<(Vec<u8>, usize, usize), String> {
    check_dims(layout.w, layout.h)?;
    let y = read_y(frame, layout)?;
    let (cw, ch) = (layout.w / 2, layout.h / 2);
    let u = Plane::from_vec(cw, ch, read_chroma(frame.u, layout)?);
    let v = Plane::from_vec(cw, ch, read_chroma(frame.v, layout)?);
    let full = yuv_to_linear(&y, u, v, layout.w, layout.h);
    let small = Rgb {
        w: full.w.div_ceil(2),
        h: full.h.div_ceil(2),
        c: [0, 1, 2].map(|c| box_downsample(&Plane { w: full.w, h: full.h, d: full.c[c].clone() }, 2).d),
    };
    let px = quick_srgb8(&small, 1.0);
    let jpeg = encode_jpeg(&px, small.w, small.h, quality)?;
    Ok((jpeg, small.w, small.h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::cfa_color;
    use crate::testutil::{textured, Rng};

    /// Ground-truth scene as linear RGB planes (mildly coloured texture).
    fn truth(w: usize, h: usize) -> [Plane; 3] {
        let t = textured(w, h);
        let k = [0.9f32, 1.0, 0.7];
        [0, 1, 2].map(|c| Plane { w, h, d: t.d.iter().map(|v| v * 0.5 * k[c]).collect() })
    }

    fn mosaic_of(t: &[Plane; 3], x0: usize, y0: usize, w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                t[cfa_color(0, x, y)].d[(y0 + y) * t[0].w + x0 + x]
            })
            .collect()
    }

    fn to_raw(norm: &[f32], w: usize, h: usize, sigma: f32, rng: &mut Rng) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 2);
        for v in norm {
            let n = v + sigma * rng.gauss();
            let raw = (64.0 + n * (1023.0 - 64.0)).round().clamp(0.0, 1023.0) as u16;
            out.extend_from_slice(&raw.to_le_bytes());
        }
        out
    }

    fn meta() -> Meta {
        Meta { white: 1023.0, black: [64.0; 4], neutral: [0.9, 1.0, 0.7], noise: NoiseModel { s: 0.0, o: 0.02 * 0.02 }, ..Meta::default() }
    }

    #[test]
    fn merged_bayer_is_closer_to_truth_than_one_frame() {
        let (w, h) = (192, 128);
        let t = truth(w + 32, h + 32);
        let sigma = 0.02;
        let shifts = [(16usize, 16usize), (18, 14), (14, 18), (20, 16), (16, 12), (12, 16)];
        let mut rng = Rng::new(42);
        let frames: Vec<Vec<f32>> = shifts
            .iter()
            .map(|(x0, y0)| mosaic_of(&t, *x0, *y0, w, h).iter().map(|v| v + sigma * rng.gauss()).collect())
            .collect();
        let greys: Vec<Plane> = frames.iter().map(|f| bin_grey(f, w, h)).collect();
        let noise = NoiseModel { s: 0.0, o: sigma * sigma };
        let (merged, ref_idx) = merge_burst(frames.len(), |i| Ok(frames[i].clone()), &greys, w, h, &noise).unwrap();
        // Truth in the reference frame's coordinates.
        let (rx, ry) = shifts[ref_idx];
        let want = mosaic_of(&t, rx, ry, w, h);
        let mse = |a: &[f32]| {
            let mut s = 0.0;
            let mut n = 0;
            for y in 24..h - 24 {
                for x in 24..w - 24 {
                    let d = a[y * w + x] - want[y * w + x];
                    s += d * d;
                    n += 1;
                }
            }
            s / n as f32
        };
        let (single, merged_mse) = (mse(&frames[ref_idx]), mse(&merged));
        assert!(merged_mse < single / 2.5, "single {} merged {}", single, merged_mse);
    }

    #[test]
    fn raw_burst_produces_decodable_jpeg() {
        let (w, h) = (128, 96);
        let t = truth(w + 16, h + 16);
        let mut rng = Rng::new(7);
        let bufs: Vec<Vec<u8>> = [(8usize, 8usize), (10, 8), (8, 6), (6, 8)]
            .iter()
            .map(|(x, y)| to_raw(&mosaic_of(&t, *x, *y, w, h), w, h, 0.02, &mut rng))
            .collect();
        let frames: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        let out = process_raw(&frames, w, h, w * 2, &meta(), None, 95, DEFAULT_MAX_PIXELS).unwrap();
        assert_eq!((out.width, out.height, out.frames), (w, h, 4));
        let mut dec = jpeg_decoder::Decoder::new(&out.jpeg[..]);
        let px = dec.decode().unwrap();
        let info = dec.info().unwrap();
        assert_eq!((info.width as usize, info.height as usize), (w, h));
        let mean = px.iter().map(|v| *v as f32).sum::<f32>() / px.len() as f32;
        assert!(mean > 40.0 && mean < 230.0, "mean {}", mean);
    }

    #[test]
    fn big_sensors_are_quad_binned() {
        let (w, h) = (128, 96);
        let t = truth(w, h);
        let mut rng = Rng::new(1);
        let buf = to_raw(&mosaic_of(&t, 0, 0, w, h), w, h, 0.0, &mut rng);
        let out = process_raw(&[&buf], w, h, w * 2, &meta(), None, 90, 1000).unwrap();
        assert_eq!((out.width, out.height), (w / 2, h / 2));
    }

    #[test]
    fn reference_render_is_half_size() {
        let (w, h) = (128, 96);
        let t = truth(w, h);
        let mut rng = Rng::new(2);
        let buf = to_raw(&mosaic_of(&t, 0, 0, w, h), w, h, 0.0, &mut rng);
        let (jpg, ow, oh) = reference_raw(&buf, w, h, w * 2, &meta(), None, 85).unwrap();
        assert_eq!((ow, oh), (w / 2, h / 2));
        assert!(jpg.len() > 100);
    }

    #[test]
    fn rejects_bad_input() {
        let m = meta();
        assert!(process_raw(&[], 128, 96, 256, &m, None, 90, 1000).is_err());
        assert!(process_raw(&[&[0u8; 10][..]], 128, 96, 256, &m, None, 90, 1000).is_err());
        assert!(process_raw(&[&[0u8; 10][..]], 33, 33, 66, &m, None, 90, 1000).is_err());
    }

    fn yuv_planes(t: &[Plane; 3], x0: usize, y0: usize, w: usize, h: usize, sigma: f32, rng: &mut Rng) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; w * h / 4];
        let mut v = vec![0u8; w * h / 4];
        for j in 0..h {
            for i in 0..w {
                let lum = t[1].d[(y0 + j) * t[1].w + x0 + i] + sigma * rng.gauss();
                y[j * w + i] = (lum * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        for j in 0..h / 2 {
            for i in 0..w / 2 {
                u[j * (w / 2) + i] = (128.0 + 20.0 + 3.0 * rng.gauss()).round() as u8;
                v[j * (w / 2) + i] = (128.0 - 10.0 + 3.0 * rng.gauss()).round() as u8;
            }
        }
        (y, u, v)
    }

    #[test]
    fn yuv_fallback_burst_works() {
        let (w, h) = (128, 96);
        let t = truth(w + 16, h + 16);
        let mut rng = Rng::new(11);
        let bufs: Vec<_> = [(8usize, 8usize), (10, 8), (8, 10)].iter().map(|(x, y)| yuv_planes(&t, *x, *y, w, h, 0.02, &mut rng)).collect();
        let frames: Vec<YuvFrame> = bufs.iter().map(|(y, u, v)| YuvFrame { y, u, v }).collect();
        let layout = YuvLayout { w, h, y_row: w, uv_row: w / 2, uv_pixel: 1 };
        let out = process_yuv(&frames, &layout, &meta(), 92).unwrap();
        assert_eq!((out.width, out.height, out.frames), (w, h, 3));
        let mut dec = jpeg_decoder::Decoder::new(&out.jpeg[..]);
        assert_eq!(dec.decode().unwrap().len(), w * h * 3);
        let (jpg, ow, oh) = reference_yuv(&frames[0], &layout, 85).unwrap();
        assert_eq!((ow, oh), (w / 2, h / 2));
        assert!(!jpg.is_empty());
    }
}
