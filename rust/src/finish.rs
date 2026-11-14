//! Stages 4 to 5 glued together: linear sRGB in, display-referred 8-bit RGB out.

use crate::demosaic::Rgb;
use crate::denoise::{strengths, wavelet_denoise};
use crate::merge::NoiseModel;
use crate::plane::{box_downsample, resize_bilinear, Plane};
use crate::sharpen::{local_contrast, unsharp, vibrance};
use crate::tone::{auto_exposure_gain, black_point, gain_map, luma_stats, luma_thumbnail, plan_fusion, rolloff_scale, s_curve, srgb_encode};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct FinishParams {
    pub scene: u8,
    /// Digital gain that restores the deliberate under-exposure of the burst.
    pub ev_gain: f32,
    pub noise: NoiseModel,
    pub frames: usize,
    /// True when the pixels already went through a camera ISP tone curve (the YUV path).
    pub display_referred: bool,
}

#[derive(Clone, Debug, Default)]
pub struct FinishInfo {
    pub total_gain: f32,
    pub fusion_evs: Vec<f32>,
    pub fusion_strength: f32,
    pub luma_sigma: f32,
}

/// Luma wavelet thresholds (noise sigmas per level, finest first): gentle, so fine grain stays.
const LUMA_K: [f32; 5] = [1.1, 1.0, 0.8, 0.6, 0.4];
/// Chroma thresholds at half resolution: strong at every scale, so no colour blotches remain.
const CHROMA_K: [f32; 6] = [3.0, 3.0, 3.0, 3.0, 2.5, 2.0];

/// Display-domain noise standard deviation on a 1/4-resolution grid. Follows the sensor noise model
/// through the frame merge, the local gain of the tone pass, the black point, sRGB encoding and the
/// contrast curve, so the denoise strength matches the noise each region really has (lifted
/// shadows are noisier than the bright sky).
#[allow(clippy::too_many_arguments)]
fn noise_map(rgb: &Rgb, gm: &Plane, total_gain: f32, bp: f32, contrast: f32, noise: &NoiseModel, frames: usize) -> Plane {
    const B: usize = 4;
    let (w, h) = (rgb.w, rgb.h);
    let (bw, bh) = (w.div_ceil(B), h.div_ceil(B));
    let (sx, sy) = (gm.w as f32 / w as f32, gm.h as f32 / h as f32);
    let eff = 1.0 + 0.7 * (frames.max(1) as f32 - 1.0);
    let lift = 1.0 / (1.0 - bp);
    let [r, g, b] = &rgb.c;
    let mut d = vec![0.0f32; bw * bh];
    d.par_chunks_mut(bw).enumerate().for_each(|(by, row)| {
        for (bx, o) in row.iter_mut().enumerate() {
            let (x0, y0) = (bx * B, by * B);
            let (x1, y1) = ((x0 + B).min(w), (y0 + B).min(h));
            let mut sum = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * w + x;
                    sum += crate::tone::luma(r[i], g[i], b[i]);
                }
            }
            let lum = (sum / ((x1 - x0) * (y1 - y0)) as f32).max(0.0);
            let (cx, cy) = ((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5);
            let k = total_gain * gm.sample(cx * sx - 0.5, cy * sy - 0.5);
            let sd_lin = (noise.var(lum) / eff).sqrt() * k * lift;
            let out = ((lum * k - bp).max(0.0) * lift).min(1.0);
            let dx = (out * 0.02).max(1e-5);
            let slope = (srgb_encode(out + dx) - srgb_encode((out - dx).max(0.0))) / (out + dx - (out - dx).max(0.0));
            let yv = srgb_encode(out);
            let curve = 1.0 - contrast * (std::f32::consts::TAU * yv).cos();
            *o = (sd_lin * slope * curve).clamp(0.0015, 0.15);
        }
    });
    Plane { w: bw, h: bh, d }
}

pub fn finish(rgb: Rgb, p: &FinishParams) -> (Vec<u8>, FinishInfo) {
    let (w, h) = (rgb.w, rgb.h);
    let [r, g, b] = &rgb.c;

    // Stage 4: exposure and HDR on a thumbnail.
    let mut thumb = luma_thumbnail(r, g, b, w, h, 768);
    let pre = if p.display_referred { 1.0 } else { p.ev_gain.max(1.0) };
    thumb.d.iter_mut().for_each(|v| *v *= pre);
    let ae = if p.display_referred { 1.0 } else { auto_exposure_gain(&luma_stats(&thumb), p.scene) };
    thumb.d.iter_mut().for_each(|v| *v *= ae);
    let total_gain = pre * ae;
    let stats = luma_stats(&thumb);
    // Black point and contrast: the ISP already did this for display-referred (YUV) input.
    let bp = if p.display_referred { 0.0 } else { black_point(stats.p005) };
    let contrast = match (p.display_referred, p.scene) {
        (true, _) => 0.0,
        (_, 2) => 0.22, // night: keep it moody but not muddy
        (_, 1) => 0.25,
        _ => 0.3,
    };
    let mut plan = plan_fusion(&stats, p.scene);
    if p.display_referred {
        plan.strength *= 0.5;
    }
    let gm = gain_map(&thumb, &plan);
    let (sx, sy) = (gm.w as f32 / w as f32, gm.h as f32 / h as f32);
    let mut sd = noise_map(&rgb, &gm, total_gain, bp, contrast, &p.noise, p.frames);
    if p.scene == 2 {
        sd.d.iter_mut().for_each(|v| *v *= 1.2);
    }

    // Tone, roll off highlights, encode gamma and split into Y / Cb / Cr in one pass.
    let mut yp = Plane::new(w, h);
    let mut cbp = Plane::new(w, h);
    let mut crp = Plane::new(w, h);
    yp.d.par_chunks_mut(w)
        .zip(cbp.d.par_chunks_mut(w).zip(crp.d.par_chunks_mut(w)))
        .enumerate()
        .for_each(|(row, (yr, (cbr, crr)))| {
            let fy = (row as f32 + 0.5) * sy - 0.5;
            for x in 0..w {
                let k = total_gain * gm.sample((x as f32 + 0.5) * sx - 0.5, fy);
                let i = row * w + x;
                let lift = 1.0 / (1.0 - bp);
                let (mut rv, mut gv, mut bv) = (((r[i] * k) - bp).max(0.0) * lift, ((g[i] * k) - bp).max(0.0) * lift, ((b[i] * k) - bp).max(0.0) * lift);
                let s = rolloff_scale(rv.max(gv).max(bv));
                rv *= s;
                gv *= s;
                bv *= s;
                let (re, ge, be) = (srgb_encode(rv), srgb_encode(gv), srgb_encode(bv));
                let y = 0.299 * re + 0.587 * ge + 0.114 * be;
                yr[x] = if contrast > 0.0 { s_curve(y, contrast) } else { y };
                cbr[x] = (be - y) * 0.564;
                crr[x] = (re - y) * 0.713;
            }
        });
    drop(rgb);

    // Stage 5: denoise, sharpen, local contrast, vibrance.
    let (luma_sigma, chroma_sigma) = strengths(&p.noise, p.frames, total_gain, p.scene);
    let _ = chroma_sigma;
    let mut y = wavelet_denoise(&yp, &sd, &LUMA_K);
    drop(yp);
    // Chroma at half resolution (JPEG stores it that way anyway): cheaper, and coarser blotches
    // fall into the six wavelet levels. Averaging 2x2 halves the noise, hence sd / 2.
    let half_sd = Plane { w: sd.w, h: sd.h, d: sd.d.iter().map(|v| v * 0.5).collect() };
    let mut cb = resize_bilinear(&wavelet_denoise(&box_downsample(&cbp, 2), &half_sd, &CHROMA_K), w, h);
    drop(cbp);
    let mut cr = resize_bilinear(&wavelet_denoise(&box_downsample(&crp, 2), &half_sd, &CHROMA_K), w, h);
    drop(crp);
    let (amount, lc) = match p.scene {
        2 => (0.5, 0.05),
        1 => (0.6, 0.1),
        _ => (0.65, 0.08),
    };
    // Capture sharpening: small radius, cored at the local noise level so grain is not amplified.
    let core = 0.6 * sd.d.iter().sum::<f32>() / sd.d.len().max(1) as f32;
    unsharp(&mut y, 0.75, amount, core, 0.02);
    local_contrast(&mut y, lc);
    vibrance(&mut cb.d, &mut cr.d, 0.15, 1.0);

    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(row, px)| {
        for x in 0..w {
            let i = row * w + x;
            let yv = y.d[i];
            let r = yv + cr.d[i] / 0.713;
            let b = yv + cb.d[i] / 0.564;
            let g = (yv - 0.299 * r - 0.114 * b) / 0.587;
            px[3 * x] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[3 * x + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[3 * x + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    });
    (out, FinishInfo { total_gain, fusion_evs: plan.evs, fusion_strength: plan.strength, luma_sigma })
}

/// Cheap conversion of linear RGB to 8-bit sRGB for previews (no tone or detail processing).
pub fn quick_srgb8(rgb: &Rgb, gain: f32) -> Vec<u8> {
    let (w, h) = (rgb.w, rgb.h);
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(row, px)| {
        for x in 0..w {
            let i = row * w + x;
            let (mut r, mut g, mut b) = (rgb.c[0][i] * gain, rgb.c[1][i] * gain, rgb.c[2][i] * gain);
            let s = rolloff_scale(r.max(g).max(b));
            r *= s;
            g *= s;
            b *= s;
            px[3 * x] = (srgb_encode(r) * 255.0 + 0.5) as u8;
            px[3 * x + 1] = (srgb_encode(g) * 255.0 + 0.5) as u8;
            px[3 * x + 2] = (srgb_encode(b) * 255.0 + 0.5) as u8;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{textured, Rng};

    fn scene(w: usize, h: usize, scale: f32) -> Rgb {
        let t = textured(w, h);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            img.c[0][i] = t.d[i] * scale * 0.9;
            img.c[1][i] = t.d[i] * scale;
            img.c[2][i] = t.d[i] * scale * 0.7;
        }
        img
    }

    fn params(scene: u8, gain: f32) -> FinishParams {
        FinishParams { scene, ev_gain: gain, noise: NoiseModel { s: 1e-3, o: 1e-5 }, frames: 6, display_referred: false }
    }

    #[test]
    fn output_has_right_size_and_uses_the_range() {
        let (out, info) = finish(scene(96, 64, 0.5), &params(0, 1.0));
        assert_eq!(out.len(), 96 * 64 * 3);
        assert!(out.iter().any(|v| *v > 150) && out.iter().any(|v| *v < 100));
        assert!(info.total_gain > 0.5 && info.total_gain < 3.0);
    }

    #[test]
    fn ev_gain_brightens_underexposed_input() {
        let dark = scene(96, 64, 0.04);
        let mean = |v: &[u8]| v.iter().map(|x| *x as f32).sum::<f32>() / v.len() as f32;
        let (a, _) = finish(dark.clone(), &params(2, 1.0));
        let (b, _) = finish(dark, &params(2, 4.0));
        assert!(mean(&b) > mean(&a) + 10.0, "{} vs {}", mean(&a), mean(&b));
    }

    #[test]
    fn hdr_scene_lifts_shadows() {
        let (w, h) = (128, 64);
        let t = textured(w, h);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            let k = if i % w < w / 2 { 0.01 } else { 1.6 };
            for c in 0..3 {
                img.c[c][i] = t.d[i] * k;
            }
        }
        let mut flat = params(1, 1.0);
        flat.frames = 8;
        let (out, info) = finish(img, &flat);
        assert!(info.fusion_strength > 0.0 && info.fusion_evs.len() >= 2);
        let dark_mean: f32 = (0..h).flat_map(|y| (8..24).map(move |x| (y, x))).map(|(y, x)| out[(y * w + x) * 3 + 1] as f32).sum::<f32>() / (h * 16) as f32;
        assert!(dark_mean > 20.0, "shadows stayed black: {}", dark_mean);
    }

    #[test]
    fn deep_shadows_stay_dark_and_contrast_is_kept() {
        // A dark room with a bright screen: most pixels near black, a bright block.
        let (w, h) = (128, 64);
        let t = textured(w, h);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            let k = if i % w < 80 { 0.004 } else { 0.5 };
            for c in 0..3 {
                img.c[c][i] = t.d[i] * k;
            }
        }
        let (out, _) = finish(img, &params(1, 1.0));
        let mean = |x0: usize, x1: usize| -> f32 {
            (0..h).flat_map(|y| (x0..x1).map(move |x| (y, x))).map(|(y, x)| out[(y * w + x) * 3 + 1] as f32).sum::<f32>() / (h * (x1 - x0)) as f32
        };
        let (dark, bright) = (mean(8, 64), mean(96, 120));
        assert!(dark < 75.0, "blacks lifted to grey: {}", dark);
        assert!(bright - dark > 80.0, "flat: dark {} bright {}", dark, bright);
    }

    #[test]
    fn s_curve_is_monotonic_with_fixed_ends() {
        assert!(s_curve(0.0, 0.3).abs() < 1e-6 && (s_curve(1.0, 0.3) - 1.0).abs() < 1e-6);
        let mut last = -1.0;
        for i in 0..=100 {
            let v = s_curve(i as f32 / 100.0, 0.3);
            assert!(v >= last);
            last = v;
        }
        assert!(s_curve(0.25, 0.3) < 0.25 && s_curve(0.75, 0.3) > 0.75);
    }

    #[test]
    fn noisy_input_comes_out_smoother() {
        let (w, h) = (96, 64);
        let mut r = Rng::new(9);
        let mut img = Rgb::new(w, h);
        // Noise as the model says a single frame has at this level (var = s * x + o).
        let mut p = params(2, 1.0);
        p.frames = 1;
        let sd = p.noise.var(0.2).sqrt();
        for i in 0..w * h {
            for c in 0..3 {
                img.c[c][i] = 0.2 + sd * r.gauss();
            }
        }
        let var_in = crate::testutil::variance(&img.c[1]);
        let (out, _) = finish(img, &p);
        let g: Vec<f32> = (0..w * h).map(|i| out[3 * i + 1] as f32 / 255.0).collect();
        // Compare in the display domain: slope of sRGB near 0.2 is about 1.2. Luma denoise is
        // deliberately gentle (fine grain stays), so ask for a clear reduction, not a wipe-out.
        let var_out = crate::testutil::variance(&g);
        assert!(var_out < var_in * 1.44 * 0.7, "in {} out {}", var_in * 1.44, var_out);
    }
}
