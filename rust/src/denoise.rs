//! Stage 5a: denoising. Edge-preserving luma (bilateral) and stronger chroma (cross-bilateral
//! at half resolution). Strength follows the sensor noise model, the number of merged frames
//! and the digital gain that was applied.

use crate::merge::NoiseModel;
use crate::plane::{box_downsample, resize_bilinear, Plane};
use crate::tone::srgb_encode;
use rayon::prelude::*;

fn lut(sigma: f32) -> ([f32; 256], f32) {
    let step = 3.0 * sigma / 255.0;
    let mut t = [0.0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        let d = i as f32 * step;
        *v = (-d * d / (2.0 * sigma * sigma)).exp();
    }
    (t, 1.0 / step)
}

#[inline]
fn look(t: &[f32; 256], inv_step: f32, d: f32) -> f32 {
    let i = (d * inv_step) as usize;
    if i > 255 {
        0.0
    } else {
        t[i]
    }
}

/// Bilateral filter of `src`. Range weights come from `guide` (sigma `sg`) and, when `sself` is set,
/// also from `src` itself.
pub fn bilateral(src: &Plane, guide: &Plane, radius: isize, ss: f32, sg: f32, sself: Option<f32>) -> Plane {
    let (w, h) = (src.w, src.h);
    let (tg, ig) = lut(sg);
    let ts = sself.map(lut);
    let side = (2 * radius + 1) as usize;
    let mut spatial = vec![0.0f32; side * side];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            spatial[(dy + radius) as usize * side + (dx + radius) as usize] = (-((dx * dx + dy * dy) as f32) / (2.0 * ss * ss)).exp();
        }
    }
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (xi, yi) = (x as isize, y as isize);
            let cg = guide.d[y * w + x];
            let cs = src.d[y * w + x];
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let g = guide.at(xi + dx, yi + dy);
                    let s = src.at(xi + dx, yi + dy);
                    let mut wt = spatial[(dy + radius) as usize * side + (dx + radius) as usize] * look(&tg, ig, (g - cg).abs());
                    if let Some((t, inv)) = &ts {
                        wt *= look(t, *inv, (s - cs).abs());
                    }
                    acc += wt * s;
                    wsum += wt;
                }
            }
            *o = if wsum > 0.0 { acc / wsum } else { cs };
        }
    });
    Plane { w, h, d: out }
}

/// Noise standard deviation in the gamma-encoded domain (around linear 0.1) after merging.
pub fn display_noise_sd(noise: &NoiseModel, frames: usize, gain: f32) -> f32 {
    let eff = 1.0 + 0.7 * (frames.max(1) as f32 - 1.0);
    let sd_lin = (noise.var(0.1) / eff).sqrt() * gain;
    let slope = (srgb_encode(0.105) - srgb_encode(0.095)) / 0.01;
    sd_lin * slope
}

/// (luma sigma, chroma sigma) for the range kernels.
pub fn strengths(noise: &NoiseModel, frames: usize, gain: f32, scene: u8) -> (f32, f32) {
    let sd = display_noise_sd(noise, frames, gain);
    let boost = if scene == 2 { 1.25 } else { 1.0 };
    ((2.2 * sd * boost).clamp(0.004, 0.09), (6.0 * sd * boost).clamp(0.012, 0.25))
}

pub fn denoise_luma(y: &Plane, sigma: f32) -> Plane {
    bilateral(y, y, 2, 1.5, sigma, None)
}

/// Denoise chroma at half resolution, guided by luma edges.
pub fn denoise_chroma(cb: &Plane, cr: &Plane, y: &Plane, luma_sigma: f32, chroma_sigma: f32) -> (Plane, Plane) {
    let (w, h) = (cb.w, cb.h);
    let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
    let ys = box_downsample(y, 2);
    let cbs = box_downsample(cb, 2);
    let crs = box_downsample(cr, 2);
    let gs = (luma_sigma * 2.5).max(0.03);
    let a = bilateral(&cbs, &ys, 3, 2.5, gs, Some(chroma_sigma));
    let b = bilateral(&crs, &ys, 3, 2.5, gs, Some(chroma_sigma));
    debug_assert_eq!((a.w, a.h), (hw, hh));
    (resize_bilinear(&a, w, h), resize_bilinear(&b, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{variance, Rng};

    #[test]
    fn bilateral_removes_noise_and_keeps_edges() {
        let (w, h) = (64, 64);
        let mut r = Rng::new(3);
        let clean: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.2 } else { 0.8 }).collect();
        let noisy = Plane::from_vec(w, h, clean.iter().map(|v| v + 0.02 * r.gauss()).collect());
        let out = denoise_luma(&noisy, 0.06);
        let err = |p: &Plane| variance(&p.d.iter().zip(&clean).map(|(a, b)| a - b).collect::<Vec<_>>());
        assert!(err(&out) < err(&noisy) / 3.0);
        // The step edge is still sharp.
        let (l, rr) = (out.d[10 * w + 29], out.d[10 * w + 34]);
        assert!(rr - l > 0.55, "edge flattened: {} {}", l, rr);
    }

    #[test]
    fn chroma_denoise_is_stronger_than_luma_and_flat_stays_flat() {
        let (w, h) = (64, 64);
        let mut r = Rng::new(5);
        let cb = Plane::from_vec(w, h, (0..w * h).map(|_| 0.05 * r.gauss()).collect());
        let flat = Plane::from_vec(w, h, vec![0.5; w * h]);
        let (a, _) = denoise_chroma(&cb, &cb, &flat, 0.02, 0.08);
        assert!(variance(&a.d) < variance(&cb.d) / 6.0);
        let z = Plane::new(w, h);
        let (c, _) = denoise_chroma(&z, &z, &flat, 0.02, 0.08);
        assert!(c.d.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn strength_scales_with_iso_frames_and_gain() {
        let low = NoiseModel { s: 1e-4, o: 1e-6 };
        let high = NoiseModel { s: 1e-2, o: 1e-4 };
        let (l1, c1) = strengths(&low, 4, 1.0, 0);
        let (l2, c2) = strengths(&high, 4, 1.0, 0);
        assert!(l2 > l1 && c2 > c1 && c1 > l1);
        let (l3, _) = strengths(&high, 12, 1.0, 0);
        assert!(l3 < l2);
        let (l4, _) = strengths(&high, 4, 2.0, 0);
        assert!(l4 >= l2);
    }
}
