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

/// Noise standard deviation of each à trous level for unit white noise (B3-spline kernel).
const LEVEL_SD: [f32; 7] = [0.8907, 0.2007, 0.0856, 0.0413, 0.0205, 0.0103, 0.0052];

/// One à trous smoothing step: separable B3 spline [1 4 6 4 1]/16 with holes of `hole` pixels.
fn atrous_step(p: &Plane, hole: usize) -> Plane {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (w, h) = (p.w, p.h);
    let hole = hole as isize;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &p.d[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, kv) in K.iter().enumerate() {
                let xx = (x as isize + (k as isize - 2) * hole).clamp(0, w as isize - 1) as usize;
                s += kv * src[xx];
            }
            *o = s;
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let r: [&[f32]; 5] = std::array::from_fn(|k| {
            let yy = (y as isize + (k as isize - 2) * hole).clamp(0, h as isize - 1) as usize;
            &tmp[yy * w..(yy + 1) * w]
        });
        for (x, o) in row.iter_mut().enumerate() {
            *o = K[0] * r[0][x] + K[1] * r[1][x] + K[2] * r[2][x] + K[3] * r[3][x] + K[4] * r[4][x];
        }
    });
    Plane { w, h, d: out }
}

/// Multi-scale wavelet (à trous) denoise, the approach of raw converters such as RawTherapee and
/// darktable. `sd` is a low-resolution map of the local noise standard deviation (sampled
/// bilinearly, so the strength follows the real noise of each region); `k[j]` is the threshold
/// in noise sigmas for level j, finest first. The shrinkage d·d²/(d²+t²) leaves real detail alone
/// and fades noise out smoothly, so fine grain survives instead of the flat, painted look of a
/// bilateral filter, and coarse levels remove the low-frequency blotches a small filter misses.
pub fn wavelet_denoise(p: &Plane, sd: &Plane, k: &[f32]) -> Plane {
    let (w, h) = (p.w, p.h);
    let (sx, sy) = (sd.w as f32 / w as f32, sd.h as f32 / h as f32);
    let mut out = vec![0.0f32; w * h];
    let mut cur = p.clone();
    for (j, kj) in k.iter().enumerate() {
        let next = atrous_step(&cur, 1 << j);
        let f = kj * LEVEL_SD[j.min(LEVEL_SD.len() - 1)];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            for (x, o) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let d = cur.d[i] - next.d[i];
                let t = f * sd.sample((x as f32 + 0.5) * sx - 0.5, fy);
                let d2 = d * d;
                *o += d * d2 / (d2 + t * t + 1e-12);
            }
        });
        cur = next;
    }
    out.par_iter_mut().zip(cur.d.par_iter()).for_each(|(o, c)| *o += c);
    Plane { w, h, d: out }
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
    fn wavelet_removes_noise_and_low_frequency_blotches_but_keeps_edges() {
        let (w, h) = (128, 128);
        let mut r = Rng::new(11);
        let clean: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.3 } else { 0.7 }).collect();
        // White noise plus a smooth blotch pattern (what demosaiced chroma noise looks like).
        let blotch = |i: usize| 0.01 * ((i % w) as f32 * 0.15).sin() * ((i / w) as f32 * 0.17).cos();
        let noisy = Plane::from_vec(w, h, (0..w * h).map(|i| clean[i] + 0.02 * r.gauss() + blotch(i)).collect());
        let sd = Plane::from_vec(4, 4, vec![0.02; 16]);
        let out = wavelet_denoise(&noisy, &sd, &[3.0; 6]);
        let err = |p: &Plane| variance(&p.d.iter().zip(&clean).map(|(a, b)| a - b).collect::<Vec<_>>());
        assert!(err(&out) < err(&noisy) / 4.0, "{} vs {}", err(&out), err(&noisy));
        let (l, rr) = (out.d[60 * w + 60], out.d[60 * w + 67]);
        assert!(rr - l > 0.3, "edge flattened: {} {}", l, rr);
        // A flat image stays exactly flat.
        let flat = Plane::from_vec(w, h, vec![0.4; w * h]);
        assert!(wavelet_denoise(&flat, &sd, &[1.0; 5]).d.iter().all(|v| (v - 0.4).abs() < 1e-5));
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
