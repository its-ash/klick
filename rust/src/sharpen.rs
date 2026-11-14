//! Stage 5b: finishing touches on the display-referred image: unsharp mask with noise
//! coring and halo control, mild local contrast, and vibrance.

use crate::plane::{box_downsample, gaussian_blur, Plane};
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
        // Overshoot is capped by the halo limit.
        assert!(p.d[10 * w + 25] >= 0.3 - 0.0201 && p.d[10 * w + 22] >= 0.3 - 0.0201);
        assert!(p.d[10 * w + 26] <= 0.6 + 0.0201);
        // Steeper than before at the edge: dark side darker, bright side brighter.
        assert!(p.d[10 * w + 23] < before.d[10 * w + 23] - 1e-4);
        assert!(p.d[10 * w + 24] > before.d[10 * w + 24] + 1e-4);
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
