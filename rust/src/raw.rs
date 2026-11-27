//! Stage 3a: sensor-domain preparation. Black/white level normalisation, hot/dead pixel
//! removal, Gr/Gb green equilibration, 2x2 binned grey, lens shading correction, neutral-anchor
//! white balance refinement, and color-preserving highlight reconstruction.

use crate::merge::NoiseModel;
use crate::meta::{cfa_color, lsc_channel, Meta, B, G, R};
use crate::plane::Plane;
use rayon::prelude::*;

/// Read one RAW16 frame (little-endian u16, `stride` bytes per row) into black-subtracted,
/// 0..1 normalised floats. Values are not clamped so noise stays unbiased for the merge.
pub fn normalize_raw(src: &[u8], w: usize, h: usize, stride: usize, meta: &Meta) -> Result<Vec<f32>, String> {
    if stride < w * 2 || src.len() < stride * (h - 1) + w * 2 {
        return Err(format!("raw buffer too small: {} bytes for {}x{} stride {}", src.len(), w, h, stride));
    }
    let mut out = vec![0.0f32; w * h];
    let scale: Vec<f32> = (0..4).map(|i| 1.0 / (meta.white - meta.black[i])).collect();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let line = &src[y * stride..y * stride + w * 2];
        for (x, o) in row.iter_mut().enumerate() {
            let v = u16::from_le_bytes([line[2 * x], line[2 * x + 1]]) as f32;
            let p = (y & 1) * 2 + (x & 1);
            *o = (v - meta.black[p]) * scale[p];
        }
    });
    Ok(out)
}

/// Remove isolated hot and dead sensor pixels in Bayer space before demosaicing.
/// Compares each site against its 4 cardinal same-channel neighbors (`±2` pixels);
/// only true single-pixel impulses exceeding `max(0.06, 6 * local_sigma)` are replaced.
pub fn despeckle_raw(bayer: &mut [f32], w: usize, h: usize, noise: &NoiseModel) {
    if w < 6 || h < 6 {
        return;
    }
    let src = bayer.to_vec();
    bayer.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < 2 || y >= h - 2 {
            return;
        }
        let r_up = &src[(y - 2) * w..(y - 1) * w];
        let r_cur = &src[y * w..(y + 1) * w];
        let r_dn = &src[(y + 2) * w..(y + 3) * w];
        for x in 2..w - 2 {
            let c = r_cur[x];
            let mut n = [r_up[x], r_dn[x], r_cur[x - 2], r_cur[x + 2]];
            n.sort_unstable_by(|a, b| a.total_cmp(b));
            let med = 0.5 * (n[1] + n[2]);
            let spread = n[3] - n[0];
            let thr = (6.0 * noise.var(med).sqrt() + 0.5 * spread).max(0.06);
            if c > n[3] + thr || c < n[0] - thr {
                row[x] = med;
            }
        }
    });
}

/// Balance systematic Gr/Gb green channel crosstalk differences so smooth skies and walls
/// never show 2x2 Bayer maze artifacts after demosaicing and sharpening.
pub fn equilibrate_green(bayer: &mut [f32], w: usize, h: usize, cfa: u8) {
    let (sum_gr, sum_gb, cnt) = bayer
        .par_chunks(2 * w)
        .enumerate()
        .map(|(qy, rows)| {
            if rows.len() < 2 * w {
                return (0.0f64, 0.0f64, 0u32);
            }
            let (mut s_gr, mut s_gb, mut n) = (0.0f64, 0.0f64, 0u32);
            let mut gr_i = 0usize;
            let mut gb_i = 0usize;
            for dy in 0..2 {
                for dx in 0..2 {
                    match lsc_channel(cfa, dx, dy) {
                        1 => gr_i = dy * w + dx,
                        2 => gb_i = dy * w + dx,
                        _ => {}
                    }
                }
            }
            for qx in 0..w / 2 {
                let o = 2 * qx;
                let gr = rows[o + gr_i];
                let gb = rows[o + gb_i];
                if gr > 0.06 && gb > 0.06 && gr < 0.85 && gb < 0.85 && (gr - gb).abs() < 0.05 * (gr + gb) {
                    s_gr += gr as f64;
                    s_gb += gb as f64;
                    n += 1;
                }
            }
            let _ = qy;
            (s_gr, s_gb, n)
        })
        .reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    if cnt < ((w * h) / 400) as u32 + 32 || sum_gb <= 0.0 {
        return;
    }
    let ratio = (sum_gr / sum_gb) as f32;
    if (ratio - 1.0).abs() < 0.004 || ratio < 0.88 || ratio > 1.14 {
        return;
    }
    let gain = ratio.clamp(0.92, 1.08);
    bayer.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            if lsc_channel(cfa, x, y) == 2 {
                *v *= gain;
            }
        }
    });
}

/// 2x2 binned grey (mean of one RGGB quad) used for alignment.
pub fn bin_grey(bayer: &[f32], w: usize, h: usize) -> Plane {
    let (gw, gh) = (w / 2, h / 2);
    let mut out = vec![0.0f32; gw * gh];
    out.par_chunks_mut(gw).enumerate().for_each(|(y, row)| {
        let r0 = &bayer[(2 * y) * w..(2 * y + 1) * w];
        let r1 = &bayer[(2 * y + 1) * w..(2 * y + 2) * w];
        for (x, o) in row.iter_mut().enumerate() {
            *o = 0.25 * (r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1]);
        }
    });
    Plane { w: gw, h: gh, d: out }
}

/// Lens shading gain map, 4 channels (R, Gr, Gb, B) per cell, row-major.
#[derive(Clone, Debug)]
pub struct Lsc {
    pub w: usize,
    pub h: usize,
    pub gains: Vec<f32>,
}

impl Lsc {
    pub fn from_slice(w: usize, h: usize, data: &[f32]) -> Option<Lsc> {
        if w < 2 || h < 2 || data.len() < w * h * 4 {
            return None;
        }
        if data[..w * h * 4].iter().any(|v| !v.is_finite() || *v < 0.5 || *v > 16.0) {
            return None;
        }
        Some(Lsc { w, h, gains: data[..w * h * 4].to_vec() })
    }

    #[inline]
    fn gain(&self, fx: f32, fy: f32, ch: usize) -> f32 {
        let x0 = (fx.floor() as usize).min(self.w - 2);
        let y0 = (fy.floor() as usize).min(self.h - 2);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let g = |x: usize, y: usize| self.gains[(y * self.w + x) * 4 + ch];
        let top = g(x0, y0) + (g(x0 + 1, y0) - g(x0, y0)) * tx;
        let bot = g(x0, y0 + 1) + (g(x0 + 1, y0 + 1) - g(x0, y0 + 1)) * tx;
        top + (bot - top) * ty
    }
}

pub fn apply_lsc(bayer: &mut [f32], w: usize, h: usize, cfa: u8, lsc: &Lsc) {
    let sx = (lsc.w - 1) as f32 / (w - 1).max(1) as f32;
    let sy = (lsc.h - 1) as f32 / (h - 1).max(1) as f32;
    bayer.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = y as f32 * sy;
        for (x, v) in row.iter_mut().enumerate() {
            *v *= lsc.gain(x as f32 * sx, fy, lsc_channel(cfa, x, y));
        }
    });
}

/// Multipliers (R, G, B) that turn the as-shot neutral into grey, with G fixed at 1.
pub fn wb_gains(neutral: [f32; 3]) -> [f32; 3] {
    [neutral[1] / neutral[0], 1.0, neutral[1] / neutral[2]]
}

/// Neutral-anchor white balance refinement. Only samples 2x2 quads that are already near-neutral
/// under the camera's as-shot WB (`max/min < 1.38`), so scenes dominated by green foliage, golden
/// sunsets, or close-up portraits are never shifted off-tint by naive grey-world averaging.
pub fn refine_wb(bayer: &[f32], w: usize, h: usize, cfa: u8, base: [f32; 3]) -> [f32; 3] {
    const ALPHA: f32 = 0.22;
    let (sum, cnt) = bayer
        .par_chunks(2 * w)
        .enumerate()
        .map(|(qy, rows)| {
            if rows.len() < 2 * w {
                return ([0.0f64; 3], 0u32);
            }
            let (mut s, mut n) = ([0.0f64; 3], 0u32);
            for qx in 0..w / 2 {
                let mut acc = [0.0f32; 3];
                let mut raw_max = 0.0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let v = rows[dy * w + 2 * qx + dx];
                        raw_max = raw_max.max(v);
                        let ch = cfa_color(cfa, 2 * qx + dx, 2 * qy + dy);
                        acc[ch] += if ch == G { 0.5 * v } else { v };
                    }
                }
                if raw_max >= 0.90 {
                    continue;
                }
                let r = acc[R] * base[R];
                let g = acc[G] * base[G];
                let b = acc[B] * base[B];
                let lo = r.min(g).min(b);
                let hi = r.max(g).max(b);
                if lo > 0.05 && hi < 0.90 && hi < lo * 1.38 {
                    // Weight slightly brighter neutral surfaces (white paper, clouds, concrete) higher.
                    let wt = (g / (0.1 + g)) as f64;
                    s[R] += r as f64 * wt;
                    s[G] += g as f64 * wt;
                    s[B] += b as f64 * wt;
                    n += 1;
                }
            }
            (s, n)
        })
        .reduce(
            || ([0.0; 3], 0),
            |a, b| ([a.0[0] + b.0[0], a.0[1] + b.0[1], a.0[2] + b.0[2]], a.1 + b.1),
        );
    let min_quads = ((w * h / 4) / 80).max(16) as u32;
    if cnt < min_quads || sum[G] <= 0.0 {
        return base;
    }
    let mut out = base;
    for c in [R, B] {
        if sum[c] > 0.0 {
            let correction = ((sum[G] / sum[c]) as f32).clamp(0.88, 1.12);
            out[c] = base[c] * correction.powf(ALPHA);
        }
    }
    out
}

/// Normalised level where a sample starts to count as clipped, and where it is fully clipped.
const CLIP_LO: f32 = 0.88;
const CLIP_HI: f32 = 0.98;

#[inline]
fn smoothstep_clip(v: f32) -> f32 {
    let t = ((v - CLIP_LO) / (CLIP_HI - CLIP_LO)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Per 2x2 quad, how clipped it is: 0 when every sample is well below saturation, rising smoothly to 1
/// when the brightest one is saturated. Must run on normalised RAW, before lens shading gains.
pub fn clip_weights(bayer: &[f32], w: usize, h: usize) -> Plane {
    let (qw, qh) = (w / 2, h / 2);
    let mut d = vec![0.0f32; qw * qh];
    d.par_chunks_mut(qw).enumerate().for_each(|(y, row)| {
        let r0 = &bayer[(2 * y) * w..(2 * y + 1) * w];
        let r1 = &bayer[(2 * y + 1) * w..(2 * y + 2) * w];
        for (x, o) in row.iter_mut().enumerate() {
            let m = r0[2 * x].max(r0[2 * x + 1]).max(r1[2 * x]).max(r1[2 * x + 1]);
            *o = smoothstep_clip(m);
        }
    });
    Plane { w: qw, h: qh, d }
}

/// Per-channel raw saturation mask (3 planes R/G/B at quad resolution, values in `0..1`)
/// captured BEFORE lens shading correction so highlight reconstruction knows exactly which
/// color channels actually clipped on the sensor.
pub fn channel_clip_masks(bayer: &[f32], w: usize, h: usize, cfa: u8) -> [Plane; 3] {
    let (qw, qh) = (w / 2, h / 2);
    let mut mr = vec![0.0f32; qw * qh];
    let mut mg = vec![0.0f32; qw * qh];
    let mut mb = vec![0.0f32; qw * qh];
    mr.par_chunks_mut(qw)
        .zip(mg.par_chunks_mut(qw).zip(mb.par_chunks_mut(qw)))
        .enumerate()
        .for_each(|(qy, (rr, (gr, br)))| {
            let r0 = &bayer[(2 * qy) * w..(2 * qy + 1) * w];
            let r1 = &bayer[(2 * qy + 1) * w..(2 * qy + 2) * w];
            for qx in 0..qw {
                let mut m = [0.0f32; 3];
                let pts = [(2 * qx, 0, r0[2 * qx]), (2 * qx + 1, 0, r0[2 * qx + 1]), (2 * qx, 1, r1[2 * qx]), (2 * qx + 1, 1, r1[2 * qx + 1])];
                for (x, dy, v) in pts {
                    let ch = cfa_color(cfa, x, 2 * qy + dy);
                    m[ch] = m[ch].max(v);
                }
                rr[qx] = smoothstep_clip(m[R]);
                gr[qx] = smoothstep_clip(m[G]);
                br[qx] = smoothstep_clip(m[B]);
            }
        });
    [Plane { w: qw, h: qh, d: mr }, Plane { w: qw, h: qh, d: mg }, Plane { w: qw, h: qh, d: mb }]
}

/// White balance with color-preserving highlight reconstruction.
/// - When all 3 sensor channels clip, pulls smoothly to neutral white (no magenta speculars).
/// - When only 1 or 2 channels clip (most commonly Green clips first under daylight because R/B WB
///   gains are > 1, or Blue clips in bright sky), lifts ONLY the clipped channel(s) toward the
///   brightest white-balanced channel so magenta casts vanish while sky blues and warm sunsets stay
///   luminous and colored rather than turning chalky grey.
pub fn apply_wb_reconstruct(bayer: &mut [f32], w: usize, cfa: u8, gains: [f32; 3], masks: &[Plane; 3]) {
    let qw = (w / 2).min(masks[0].w);
    bayer.par_chunks_mut(2 * w).enumerate().for_each(|(qy, rows)| {
        if rows.len() < 2 * w || qy >= masks[0].h {
            return;
        }
        for qx in 0..qw {
            let idx = [2 * qx, 2 * qx + 1, w + 2 * qx, w + 2 * qx + 1];
            let mut vals = [0.0f32; 4];
            let mut chs = [0usize; 4];
            let mut top = 0.0f32;
            for (k, &i) in idx.iter().enumerate() {
                let (x, y) = (i % w, 2 * qy + i / w);
                let ch = cfa_color(cfa, x, y);
                chs[k] = ch;
                vals[k] = rows[i].max(0.0) * gains[ch];
                top = top.max(vals[k]);
            }
            let q = qy * masks[0].w + qx;
            let cr = masks[R].d[q];
            let cg = masks[G].d[q];
            let cb = masks[B].d[q];
            let all_clipped = cr.min(cg).min(cb);
            let any_clipped = cr.max(cg).max(cb);
            if any_clipped <= 1e-4 {
                for (k, &i) in idx.iter().enumerate() {
                    rows[i] = vals[k];
                }
                continue;
            }
            let c_arr = [cr, cg, cb];
            for (k, &i) in idx.iter().enumerate() {
                let ch = chs[k];
                // Clipped channels are lifted up to `top` (kills pink/cyan casts while leaving
                // unclipped channels untouched so partial highlights keep natural hue!).
                // Once all 3 channels clip, smoothly converge all channels to neutral white.
                let t = c_arr[ch].max(all_clipped);
                rows[i] = vals[k] + (top - vals[k]) * t;
            }
        }
    });
}

/// Legacy single-mask highlight WB kept for reference renders and unit tests.
pub fn apply_wb_highlights(bayer: &mut [f32], w: usize, cfa: u8, gains: [f32; 3], clip: &Plane) {
    bayer.par_chunks_mut(2 * w).enumerate().for_each(|(qy, rows)| {
        if rows.len() < 2 * w || qy >= clip.h {
            return;
        }
        for qx in 0..(w / 2).min(clip.w) {
            let idx = [2 * qx, 2 * qx + 1, w + 2 * qx, w + 2 * qx + 1];
            let mut vals = [0.0f32; 4];
            let mut top = 0.0f32;
            for (k, &i) in idx.iter().enumerate() {
                let (x, y) = (i % w, 2 * qy + i / w);
                vals[k] = rows[i].max(0.0) * gains[cfa_color(cfa, x, y)];
                top = top.max(vals[k]);
            }
            let t = clip.d[qy * clip.w + qx];
            for (k, &i) in idx.iter().enumerate() {
                rows[i] = vals[k] + (top - vals[k]) * t;
            }
        }
    });
}

pub fn apply_wb(bayer: &mut [f32], w: usize, cfa: u8, gains: [f32; 3]) {
    bayer.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = (*v).clamp(0.0, 1.0) * gains[cfa_color(cfa, x, y)];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_highlights_come_out_neutral_not_pink() {
        let (w, h) = (8, 8);
        let mut b: Vec<f32> = (0..w * h).map(|i| if i % w < 4 { 1.0 } else { 0.3 }).collect();
        let gains = [2.0, 1.0, 1.6];
        let clip = clip_weights(&b, w, h);
        apply_wb_highlights(&mut b, w, 0, gains, &clip);
        let clipped: Vec<f32> = (0..h).flat_map(|y| (0..4).map(move |x| (y, x))).map(|(y, x)| b[y * w + x]).collect();
        assert!(clipped.iter().all(|v| (v - 2.0).abs() < 1e-5), "{:?}", clipped);
        for y in 0..h {
            for x in 4..w {
                let want = 0.3 * gains[cfa_color(0, x, y)];
                assert!((b[y * w + x] - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn reconstruct_keeps_sky_color_and_kills_magenta_clip() {
        let (w, h) = (8, 8);
        // Bright blue sky: Blue clipped at 1.0, Green 0.6, Red 0.25.
        let mut b: Vec<f32> = (0..w * h)
            .map(|i| match cfa_color(0, i % w, i / w) {
                R => 0.25,
                G => 0.60,
                _ => 1.0,
            })
            .collect();
        let gains = [2.0, 1.0, 1.5];
        let masks = channel_clip_masks(&b, w, h, 0);
        apply_wb_reconstruct(&mut b, w, 0, gains, &masks);
        // Unclipped R (0.50) and G (0.60) stay colored instead of turning flat 1.5 grey!
        assert!((b[0] - 0.50).abs() < 1e-4, "R {}", b[0]);
        assert!((b[1] - 0.60).abs() < 1e-4, "G {}", b[1]);
        assert!((b[w + 1] - 1.50).abs() < 1e-4, "B {}", b[w + 1]);
    }

    #[test]
    fn despeckle_fixes_isolated_hot_pixel() {
        let (w, h) = (16, 16);
        let mut b = vec![0.2f32; w * h];
        b[8 * w + 8] = 0.95; // hot pixel
        despeckle_raw(&mut b, w, h, &NoiseModel { s: 1e-4, o: 1e-5 });
        assert!((b[8 * w + 8] - 0.2).abs() < 1e-4, "{}", b[8 * w + 8]);
    }

    #[test]
    fn equilibrate_green_balances_gr_gb() {
        let (w, h) = (32, 32);
        let mut b = vec![0.4f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if lsc_channel(0, x, y) == 2 {
                    b[y * w + x] = 0.38; // 5% Gb crosstalk dip
                }
            }
        }
        equilibrate_green(&mut b, w, h, 0);
        assert!((b[1 * w + 0] - 0.4).abs() < 0.005, "{}", b[1 * w]);
    }

    fn meta() -> Meta {
        Meta { white: 1023.0, black: [63.0, 64.0, 64.0, 65.0], ..Meta::default() }
    }

    #[test]
    fn normalizes_levels() {
        let m = meta();
        let (w, h) = (4, 2);
        let mut bytes = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v: u16 = if (y & 1, x & 1) == (0, 0) { 63 } else { 1023 };
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&[0, 0]);
        }
        let n = normalize_raw(&bytes, w, h, w * 2 + 2, &m).unwrap();
        assert!((n[0]).abs() < 1e-6);
        assert!((n[1] - 1.0).abs() < 1e-6);
        assert!(normalize_raw(&bytes[..10], w, h, w * 2 + 2, &m).is_err());
    }

    #[test]
    fn bin_grey_averages_quads() {
        let b: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0];
        let g = bin_grey(&b, 4, 2);
        assert_eq!((g.w, g.h), (2, 1));
        assert!((g.d[0] - 1.25).abs() < 1e-6 && (g.d[1] - 1.25).abs() < 1e-6);
    }

    #[test]
    fn lsc_flattens_vignette() {
        let (w, h) = (64, 48);
        let (lw, lh) = (9usize, 7usize);
        let fall = |u: f32, v: f32| 1.0 - 0.2 * (((u - 0.5).powi(2) + (v - 0.5).powi(2)) / 0.5);
        let mut gains = Vec::new();
        for j in 0..lh {
            for i in 0..lw {
                let g = 1.0 / fall(i as f32 / (lw - 1) as f32, j as f32 / (lh - 1) as f32);
                gains.extend_from_slice(&[g; 4]);
            }
        }
        let lsc = Lsc::from_slice(lw, lh, &gains).unwrap();
        let mut img: Vec<f32> = (0..w * h)
            .map(|i| 0.5 * fall((i % w) as f32 / (w - 1) as f32, (i / w) as f32 / (h - 1) as f32))
            .collect();
        apply_lsc(&mut img, w, h, 0, &lsc);
        assert!(img.iter().all(|v| (v - 0.5).abs() < 0.01), "not flat");
        assert!(Lsc::from_slice(2, 2, &[0.1; 16]).is_none());
    }

    #[test]
    fn wb_makes_neutral_grey() {
        let g = wb_gains([0.5, 1.0, 0.4]);
        assert!((0.5 * g[0] - 1.0).abs() < 1e-6 && (0.4 * g[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn refine_wb_is_bounded_and_preserves_colored_scenes() {
        let (w, h) = (64, 64);
        let mut img = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = match cfa_color(0, x, y) {
                    0 => 0.25,
                    1 => 0.5,
                    _ => 0.2,
                };
            }
        }
        let base = [2.0, 1.0, 2.5];
        let r = refine_wb(&img, w, h, 0, base);
        assert!((r[0] - 2.0).abs() < 1e-4 && (r[2] - 2.5).abs() < 1e-4);
        // A pure bright-green forest scene has no neutral quads -> as-shot WB stays untouched!
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = match cfa_color(0, x, y) {
                    1 => 0.5,
                    _ => 0.08,
                };
            }
        }
        let forest = refine_wb(&img, w, h, 0, [1.5, 1.0, 1.5]);
        assert!((forest[0] - 1.5).abs() < 1e-5 && (forest[2] - 1.5).abs() < 1e-5);
    }
}
