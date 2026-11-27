//! Stage 5b: Halo-free Guided Clarity (edge-preserving midtone depth), per-pixel noise-masked
//! dual-scale optical & micro-texture sharpening, and perceptual vibrance helpers.

use crate::plane::{box_downsample, fast_guided_filter, gaussian_blur, Plane};
use rayon::prelude::*;

fn minmax(p: &Plane, r: isize) -> (Plane, Plane) {
    let (w, h) = (p.w, p.h);
    let pass = |src: &[f32], horizontal: bool, is_min: bool| -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut v = if is_min { f32::MAX } else { f32::MIN };
                for k in -r..=r {
                    let (xx, yy) = if horizontal {
                        ((x as isize + k).clamp(0, w as isize - 1) as usize, y)
                    } else {
                        (x, (y as isize + k).clamp(0, h as isize - 1) as usize)
                    };
                    let s = src[yy * w + xx];
                    v = if is_min { v.min(s) } else { v.max(s) };
                }
                *o = v;
            }
        });
        out
    };
    let lo = pass(&pass(&p.d, true, true), false, true);
    let hi = pass(&pass(&p.d, true, false), false, false);
    (Plane { w, h, d: lo }, Plane { w, h, d: hi })
}

/// Unsharp mask on luma. Small details below `core` are left alone (they are mostly noise);
/// the result may overshoot the local min/max by at most `halo`.
pub fn unsharp(y: &mut Plane, sigma: f32, amount: f32, core: f32, halo: f32) {
    let blur = gaussian_blur(y, sigma);
    let (lo, hi) = minmax(y, 2);
    let c2 = core * core;
    y.d.par_iter_mut().enumerate().for_each(|(i, v)| {
        let d = *v - blur.d[i];
        let d = d * (d * d) / (d * d + c2 + 1e-12);
        let o = *v + amount * d;
        *v = o.clamp(lo.d[i] - halo, hi.d[i] + halo).clamp(0.0, 1.0);
    });
}

/// Spatially-adaptive dual-scale pro sharpening using the per-pixel local noise map `sd`:
/// - Scale 1 (`σ = 0.72`): optical diffraction/AA capture sharpening on crisp edges.
/// - Scale 2 (`σ = 1.65`): subtle fine micro-texture pop weighted toward midtones.
/// Both scales are cored by the exact local noise level (so lifted shadows & smooth skies are never
/// turned gritty while well-lit subjects get tack-sharp prime-lens bite) and tightly halo-clamped.
pub fn pro_sharpen(y: &mut Plane, sd: &Plane, fine_amount: f32, texture_amount: f32, halo: f32) {
    if fine_amount <= 0.0 && texture_amount <= 0.0 {
        return;
    }
    let (w, h) = (y.w, y.h);
    let blur_fine = gaussian_blur(y, 0.72);
    let blur_tex = if texture_amount > 0.0 { Some(gaussian_blur(y, 1.65)) } else { None };
    let (lo, hi) = minmax(y, 2);
    let (sx, sy) = (sd.w as f32 / w as f32, sd.h as f32 / h as f32);

    y.d.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        let fy = (py as f32 + 0.5) * sy - 0.5;
        let off = py * w;
        for (px, v) in row.iter_mut().enumerate() {
            let i = off + px;
            let val = (*v).clamp(0.0, 1.0);
            let local_sd = sd.sample((px as f32 + 0.5) * sx - 0.5, fy);
            // Scale 1: optical capture sharpening cored tightly to local noise.
            let c1 = (0.75 * local_sd).max(0.0025);
            let d1 = val - blur_fine.d[i];
            let d1_cored = d1 * (d1 * d1) / (d1 * d1 + c1 * c1 + 1e-12);
            // Scale 2: fine micro-texture (midtone-weighted, stronger noise gate so skies stay silky).
            let mut delta = fine_amount * d1_cored;
            if let Some(bt) = &blur_tex {
                let c2 = (1.8 * local_sd).max(0.006);
                let d2 = blur_fine.d[i] - bt.d[i];
                let d2_cored = d2 * (d2 * d2) / (d2 * d2 + c2 * c2 + 1e-12);
                let mid = (4.0 * val * (1.0 - val)).max(0.0);
                delta += texture_amount * mid * d2_cored;
            }
            let o = val + delta;
            *v = o.clamp(lo.d[i] - halo, hi.d[i] + halo).clamp(0.0, 1.0);
        }
    });
}

/// Halo-free edge-preserving midtone Clarity using the Fast Guided Filter (Lightroom Clarity style).
/// Separates the image into an edge-preserving base layer and medium-scale local texture, boosting
/// tactile 3D structure in midtones while suppressing noise in shadows and preventing sky halos.
pub fn guided_clarity(y: &mut Plane, sd: &Plane, amount: f32) {
    if amount <= 0.0 || y.w < 16 || y.h < 16 {
        return;
    }
    let short_side = y.w.min(y.h);
    let radius = (short_side / 42).clamp(6, 36);
    let scale = (short_side / 360).clamp(1, 4);
    let base = fast_guided_filter(y, y, radius, 0.032, scale);
    let (w, sx, sy) = (y.w, sd.w as f32 / y.w as f32, sd.h as f32 / y.h as f32);
    y.d.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        let fy = (py as f32 + 0.5) * sy - 0.5;
        let off = py * w;
        for (px, v) in row.iter_mut().enumerate() {
            let val = (*v).clamp(0.0, 1.0);
            let d = val - base.d[off + px];
            let local_sd = sd.sample((px as f32 + 0.5) * sx - 0.5, fy);
            let gate = (2.5 * local_sd).max(0.008);
            let d_clean = d * (d * d) / (d * d + gate * gate + 1e-12);
            // Smooth midtone bell protects deepest shadows and brightest speculars from clipping.
            let mid = (4.0 * val * (1.0 - val)).max(0.0).powf(0.85);
            *v = (val + amount * mid * d_clean).clamp(0.0, 1.0);
        }
    });
}

/// Large-radius contrast boost weighted towards mid-tones.
pub fn local_contrast(y: &mut Plane, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let k = (y.w.max(y.h) / 96).max(2);
    let low = gaussian_blur(&box_downsample(y, k), 2.0);
    let (w, kf) = (y.w, k as f32);
    y.d.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        let fy = (py as f32 + 0.5) / kf - 0.5;
        for (px, v) in row.iter_mut().enumerate() {
            let l = low.sample((px as f32 + 0.5) / kf - 0.5, fy);
            let mid = 4.0 * *v * (1.0 - *v);
            *v = (*v + amount * mid * (*v - l)).clamp(0.0, 1.0);
        }
    });
}

/// Boost low-saturation colours more than saturated ones. `cb`/`cr` are in -0.5..0.5.
pub fn vibrance(cb: &mut [f32], cr: &mut [f32], vib: f32, sat: f32) {
    cb.par_iter_mut().zip(cr.par_iter_mut()).for_each(|(b, r)| {
        let s = ((*b * *b + *r * *r).sqrt() / 0.5).min(1.0);
        let f = sat * (1.0 + vib * (1.0 - s));
        *b *= f;
        *r *= f;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge() -> Plane {
        let (w, h) = (48, 32);
        Plane::from_vec(w, h, (0..w * h).map(|i| if i % w < w / 2 { 0.3 } else { 0.6 }).collect())
    }

    #[test]
    fn unsharp_steepens_edges_without_big_halos() {
        let mut p = edge();
        let w = p.w;
        let before = p.clone();
        unsharp(&mut p, 1.2, 1.0, 0.002, 0.02);
        assert!(p.d.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        assert!(p.d[10 * w + 25] >= 0.3 - 0.0201 && p.d[10 * w + 22] >= 0.3 - 0.0201);
        assert!(p.d[10 * w + 26] <= 0.6 + 0.0201);
        assert!(p.d[10 * w + 23] < before.d[10 * w + 23] - 1e-4);
        assert!(p.d[10 * w + 24] > before.d[10 * w + 24] + 1e-4);
    }

    #[test]
    fn pro_sharpen_respects_local_noise_map() {
        let mut clean_y = edge();
        let mut noisy_y = edge();
        let sd_lo = Plane::from_vec(4, 4, vec![0.002; 16]);
        let sd_hi = Plane::from_vec(4, 4, vec![0.20; 16]);
        let before = clean_y.d[10 * clean_y.w + 24];
        pro_sharpen(&mut clean_y, &sd_lo, 0.8, 0.2, 0.02);
        pro_sharpen(&mut noisy_y, &sd_hi, 0.8, 0.2, 0.02);
        assert!(clean_y.d[10 * clean_y.w + 24] - before > noisy_y.d[10 * noisy_y.w + 24] - before);
    }

    #[test]
    fn guided_clarity_keeps_flat_and_range() {
        let mut flat = Plane::from_vec(40, 40, vec![0.4; 1600]);
        let sd = Plane::from_vec(4, 4, vec![0.005; 16]);
        guided_clarity(&mut flat, &sd, 0.3);
        assert!(flat.d.iter().all(|v| (v - 0.4).abs() < 1e-4));
        let mut e = edge();
        e.d[0] = -0.002; // slight wavelet undershoot must never produce NaN
        e.d[1] = 1.003;
        guided_clarity(&mut e, &sd, 0.35);
        assert!(e.d.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn coring_leaves_tiny_detail_alone() {
        let (w, h) = (32, 32);
        let mut p = Plane::from_vec(w, h, (0..w * h).map(|i| 0.5 + if (i + i / w) % 2 == 0 { 0.001 } else { -0.001 }).collect());
        let before = p.clone();
        unsharp(&mut p, 1.0, 1.0, 0.02, 0.02);
        let dev: f32 = p.d.iter().zip(&before.d).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(dev < 0.0005, "{}", dev);
    }

    #[test]
    fn local_contrast_keeps_range_and_flat() {
        let mut flat = Plane::from_vec(40, 40, vec![0.4; 1600]);
        local_contrast(&mut flat, 0.3);
        assert!(flat.d.iter().all(|v| (v - 0.4).abs() < 1e-4));
        let mut e = edge();
        local_contrast(&mut e, 0.5);
        assert!(e.d.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn vibrance_boosts_dull_more_than_vivid() {
        let (mut b, mut r) = (vec![0.02, 0.4], vec![0.0, 0.0]);
        vibrance(&mut b, &mut r, 0.4, 1.0);
        assert!(b[0] / 0.02 > b[1] / 0.4);
        assert!(b[0] > 0.02);
    }
}
