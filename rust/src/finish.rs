//! Stages 4 to 5 glued together: linear sRGB in, display-referred 8-bit RGB out.

use crate::demosaic::Rgb;
use crate::denoise::{denoise_chroma, denoise_luma, strengths};
use crate::merge::NoiseModel;
use crate::plane::Plane;
use crate::sharpen::{local_contrast, unsharp, vibrance};
use crate::tone::{auto_exposure_gain, gain_map, luma_stats, luma_thumbnail, plan_fusion, rolloff_scale, srgb_encode};
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
    let mut plan = plan_fusion(&luma_stats(&thumb), p.scene);
    if p.display_referred {
        plan.strength *= 0.5;
    }
    let gm = gain_map(&thumb, &plan);
    let (sx, sy) = (gm.w as f32 / w as f32, gm.h as f32 / h as f32);

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
                let (mut rv, mut gv, mut bv) = ((r[i] * k).max(0.0), (g[i] * k).max(0.0), (b[i] * k).max(0.0));
                let s = rolloff_scale(rv.max(gv).max(bv));
                rv *= s;
                gv *= s;
                bv *= s;
                let (re, ge, be) = (srgb_encode(rv), srgb_encode(gv), srgb_encode(bv));
                let y = 0.299 * re + 0.587 * ge + 0.114 * be;
                yr[x] = y;
                cbr[x] = (be - y) * 0.564;
                crr[x] = (re - y) * 0.713;
            }
        });
    drop(rgb);

    // Stage 5: denoise, sharpen, local contrast, vibrance.
    let (luma_sigma, chroma_sigma) = strengths(&p.noise, p.frames, total_gain, p.scene);
    let mut y = denoise_luma(&yp, luma_sigma);
    let (mut cb, mut cr) = denoise_chroma(&cbp, &crp, &y, luma_sigma, chroma_sigma);
    drop((yp, cbp, crp));
    let (amount, lc) = match p.scene {
        2 => (0.45, 0.08),
        1 => (0.65, 0.18),
        _ => (0.7, 0.12),
    };
    unsharp(&mut y, 1.0, amount, luma_sigma * 0.6, 0.03);
    local_contrast(&mut y, lc);
    vibrance(&mut cb.d, &mut cr.d, 0.25, 1.05);

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
    fn noisy_input_comes_out_smoother() {
        let (w, h) = (96, 64);
        let mut r = Rng::new(9);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            for c in 0..3 {
                img.c[c][i] = 0.2 + 0.02 * r.gauss();
            }
        }
        let var_in = crate::testutil::variance(&img.c[1]);
        let (out, _) = finish(img, &params(2, 1.0));
        let g: Vec<f32> = (0..w * h).map(|i| out[3 * i + 1] as f32 / 255.0).collect();
        // Compare in the display domain: slope of sRGB near 0.2 is about 1.2.
        assert!(crate::testutil::variance(&g) < var_in * 1.44 / 2.0);
    }
}
