//! Stage 5a: Inter-scale BiShrink à trous wavelet denoising (Sendur & Selesnick) +
//! Guided edge-locked chroma upsampling (zero color bleeding across sharp edges).
//! Strength follows the calibrated sensor noise map at every pixel.

use crate::merge::NoiseModel;
use crate::plane::{box_downsample, guided_coeffs, resize_bilinear, Plane};
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

/// Multi-scale à trous wavelet denoise with Inter-Scale BiShrink (Sendur & Selesnick):
/// real edges & fine textures persist across adjacent wavelet scales (joint parent-child energy is
/// large -> preserved crisply), whereas sensor noise is uncorrelated across scales (shrunk cleanly).
/// Leaves fine natural texture intact instead of the waxy, painted look of plain bilateral filters.
pub fn wavelet_denoise(p: &Plane, sd: &Plane, k: &[f32]) -> Plane {
    if k.is_empty() {
        return p.clone();
    }
    let (w, h) = (p.w, p.h);
    let (sx, sy) = (sd.w as f32 / w as f32, sd.h as f32 / h as f32);
    // Stream parent/child bands without storing all scales in RAM at once (only 2 detail bands live!).
    let mut out = vec![0.0f32; w * h];
    let mut cur = p.clone();
    let mut next = atrous_step(&cur, 1);
    let mut cur_detail: Vec<f32> = cur.d.par_iter().zip(&next.d).map(|(a, b)| a - b).collect();
    cur = next;

    for j in 0..k.len() {
        let parent_detail: Option<Vec<f32>> = if j + 1 < k.len() {
            next = atrous_step(&cur, 1 << (j + 1));
            let d: Vec<f32> = cur.d.par_iter().zip(&next.d).map(|(a, b)| a - b).collect();
            cur = next;
            Some(d)
        } else {
            None
        };

        let f = k[j] * LEVEL_SD[j.min(LEVEL_SD.len() - 1)];
        let f_parent = LEVEL_SD[(j + 1).min(LEVEL_SD.len() - 1)];
        let scale_ratio = (LEVEL_SD[j.min(LEVEL_SD.len() - 1)] / f_parent.max(1e-4)).min(3.0);
        let p_ref = parent_detail.as_deref();
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            let row_off = y * w;
            for (x, o) in row.iter_mut().enumerate() {
                let i = row_off + x;
                let d = cur_detail[i];
                let t = f * sd.sample((x as f32 + 0.5) * sx - 0.5, fy);
                // Joint parent-child energy: correlated edges resist shrinkage; uncorrelated noise shrinks.
                let e2 = match p_ref {
                    Some(pd) => {
                        let dp = pd[i] * scale_ratio;
                        d * d + 0.45 * dp * dp
                    }
                    None => d * d,
                };
                // 4th-order BiShrink Wiener weight: crushes sub-threshold noise more strongly while
                // preserving >=99% of real edge contrast (eliminates multi-scale edge bleed/halos).
                let e4 = e2 * e2;
                let t4 = (t * t) * (t * t);
                *o += d * e4 / (e4 + t4 + 1e-18);
            }
        });

        if let Some(pd) = parent_detail {
            cur_detail = pd;
        }
    }
    out.par_iter_mut().zip(cur.d.par_iter()).for_each(|(o, c)| *o += c);
    Plane { w, h, d: out }
}

/// Joint `(L, a, b)` Bilateral Upsampling of half-resolution denoised chroma `(a_half, b_half)`
/// back to full resolution using full-resolution `(l_full, a_raw, b_raw)` as edge guidance.
/// Unlike pure luma-guided filters (which blur across isoluminant red/grey or blue/grey boundaries
/// where `ΔL ≈ 0`), weighting the 2x2 nearest half-res chroma samples by BOTH lightness and chroma
/// similarity keeps every colored edge razor-sharp with zero color bleed while smoothing flat noise.
pub fn jbu_upsample_chroma(
    a_half: &Plane,
    b_half: &Plane,
    l_full: &Plane,
    a_raw: &Plane,
    b_raw: &Plane,
    sd: &Plane,
) -> (Plane, Plane) {
    let (w, h) = (l_full.w, l_full.h);
    let (hw, hh) = (a_half.w, a_half.h);
    let l_half = box_downsample(l_full, 2);
    let (sx, sy) = (sd.w as f32 / w as f32, sd.h as f32 / h as f32);
    let mut oa = vec![0.0f32; w * h];
    let mut ob = vec![0.0f32; w * h];
    oa.par_chunks_mut(w)
        .zip(ob.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (ra, rb))| {
            let fy = (y as f32 + 0.5) * 0.5 - 0.5;
            let y0 = fy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(hh - 1);
            let wy1 = (fy - y0 as f32).clamp(0.0, 1.0);
            let wy0 = 1.0 - wy1;
            let sdy = (y as f32 + 0.5) * sy - 0.5;
            let off = y * w;
            for x in 0..w {
                let fx = (x as f32 + 0.5) * 0.5 - 0.5;
                let x0 = fx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(hw - 1);
                let wx1 = (fx - x0 as f32).clamp(0.0, 1.0);
                let wx0 = 1.0 - wx1;

                let i = off + x;
                let lc = l_full.d[i];
                let ac = a_raw.d[i];
                let bc = b_raw.d[i];
                let local_sd = sd.sample((x as f32 + 0.5) * sx - 0.5, sdy);
                let inv_2sl2 = 0.5 / (local_sd * 2.5).max(0.030).powi(2);
                let inv_2sc2 = 0.5 / (local_sd * 2.2).max(0.028).powi(2);

                let pts = [
                    (x0, y0, wx0 * wy0),
                    (x1, y0, wx1 * wy0),
                    (x0, y1, wx0 * wy1),
                    (x1, y1, wx1 * wy1),
                ];
                let mut sa = 0.0f32;
                let mut sb = 0.0f32;
                let mut sw = 0.0f32;
                for (hx, hy, ws) in pts {
                    let hi = hy * hw + hx;
                    let dl = l_half.d[hi] - lc;
                    let ah = a_half.d[hi];
                    let bh = b_half.d[hi];
                    let da = ah - ac;
                    let db = bh - bc;
                    let wr = (-(dl * dl) * inv_2sl2 - (da * da + db * db) * inv_2sc2).exp();
                    let w_tot = ws * wr + 1e-7 * ws;
                    sa += w_tot * ah;
                    sb += w_tot * bh;
                    sw += w_tot;
                }
                ra[x] = sa / sw;
                rb[x] = sb / sw;
            }
        });
    (Plane { w, h, d: oa }, Plane { w, h, d: ob })
}

/// Edge-guided upsampling of a half-resolution denoised chroma plane back to full resolution,
/// guided by full-resolution lightness `guide_full`. Snaps color transitions onto crisp lightness
/// boundaries so denoised chroma never bleeds across high-contrast edges.
pub fn guided_upsample_chroma(chroma_half: &Plane, guide_full: &Plane) -> Plane {
    let guide_half = box_downsample(guide_full, 2);
    debug_assert_eq!((guide_half.w, guide_half.h), (chroma_half.w, chroma_half.h));
    let (ma, mb) = guided_coeffs(&guide_half, chroma_half, 2, 0.0025);
    let (w, h) = (guide_full.w, guide_full.h);
    let (sx, sy) = (ma.w as f32 / w as f32, ma.h as f32 / h as f32);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let gr = &guide_full.d[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            *o = ma.sample(fx, fy) * gr[x] + mb.sample(fx, fy);
        }
    });
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
        let blotch = |i: usize| 0.01 * ((i % w) as f32 * 0.15).sin() * ((i / w) as f32 * 0.17).cos();
        let noisy = Plane::from_vec(w, h, (0..w * h).map(|i| clean[i] + 0.02 * r.gauss() + blotch(i)).collect());
        let sd = Plane::from_vec(4, 4, vec![0.02; 16]);
        let out = wavelet_denoise(&noisy, &sd, &[3.0; 6]);
        let err = |p: &Plane| variance(&p.d.iter().zip(&clean).map(|(a, b)| a - b).collect::<Vec<_>>());
        assert!(err(&out) < err(&noisy) / 4.0, "{} vs {}", err(&out), err(&noisy));
        let (l, rr) = (out.d[60 * w + 60], out.d[60 * w + 67]);
        assert!(rr - l > 0.3, "edge flattened: {} {}", l, rr);
        let flat = Plane::from_vec(w, h, vec![0.4; w * h]);
        assert!(wavelet_denoise(&flat, &sd, &[1.0; 5]).d.iter().all(|v| (v - 0.4).abs() < 1e-5));
    }

    #[test]
    fn guided_chroma_upsample_keeps_edge_aligned() {
        let (w, h) = (64, 64);
        let luma = Plane::from_vec(w, h, (0..w * h).map(|i| if i % w < 32 { 0.2 } else { 0.8 }).collect());
        let chroma_h = Plane::from_vec(w / 2, h / 2, (0..(w / 2) * (h / 2)).map(|i| if i % (w / 2) < 16 { -0.1 } else { 0.1 }).collect());
        let up = guided_upsample_chroma(&chroma_h, &luma);
        assert!(up.d[32 * w + 31] < -0.05 && up.d[32 * w + 32] > 0.05);
    }

    #[test]
    fn jbu_chroma_upsample_stops_isoluminant_color_bleed() {
        let (w, h) = (64, 64);
        let luma = Plane::from_vec(w, h, vec![0.5; w * h]); // completely flat L!
        let a_raw = Plane::from_vec(w, h, (0..w * h).map(|i| if i % w < 32 { 0.22 } else { 0.0 }).collect());
        let b_raw = Plane::new(w, h);
        let ah = box_downsample(&a_raw, 2);
        let bh = box_downsample(&b_raw, 2);
        let sd = Plane::from_vec(4, 4, vec![0.008; 16]);
        let (au, _) = jbu_upsample_chroma(&ah, &bh, &luma, &a_raw, &b_raw, &sd);
        assert!(au.d[32 * w + 31] > 0.20 && au.d[32 * w + 32] < 0.02, "bleed: {} {}", au.d[32 * w + 31], au.d[32 * w + 32]);
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
