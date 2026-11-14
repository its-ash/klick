//! Stage 3a: sensor-domain preparation. Black/white level normalisation, 2x2 binned grey,
//! lens shading correction and white balance (as-shot neutral with a gentle grey-world refinement).

use crate::meta::{cfa_color, lsc_channel, Meta};
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

/// Nudge the as-shot gains towards a grey-world estimate, only when it is close.
/// Mid-tone pixels (not near black or clipping) are averaged per channel.
pub fn refine_wb(bayer: &[f32], w: usize, h: usize, cfa: u8, base: [f32; 3]) -> [f32; 3] {
    const ALPHA: f32 = 0.2;
    let (sum, cnt) = bayer
        .par_chunks(w)
        .enumerate()
        .map(|(y, row)| {
            let mut s = [0.0f64; 3];
            let mut c = [0u32; 3];
            for (x, v) in row.iter().enumerate() {
                let ch = cfa_color(cfa, x, y);
                let lin = *v * base[ch];
                if lin > 0.03 && *v < 0.92 && lin < 0.92 {
                    s[ch] += lin as f64;
                    c[ch] += 1;
                }
            }
            (s, c)
        })
        .reduce(
            || ([0.0; 3], [0; 3]),
            |a, b| ([a.0[0] + b.0[0], a.0[1] + b.0[1], a.0[2] + b.0[2]], [a.1[0] + b.1[0], a.1[1] + b.1[1], a.1[2] + b.1[2]]),
        );
    if cnt.iter().any(|&c| c < (w * h / 400) as u32 + 16) {
        return base;
    }
    let m: Vec<f32> = (0..3).map(|c| (sum[c] / cnt[c] as f64) as f32).collect();
    let mut out = base;
    for c in [0usize, 2] {
        let correction = (m[1] / m[c]).clamp(0.85, 1.15);
        out[c] = base[c] * correction.powf(ALPHA);
    }
    out
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
            bytes.extend_from_slice(&[0, 0]); // row padding
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
        // Gain map that exactly inverts a radial falloff sampled on a 9x7 grid.
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
    fn refine_wb_is_bounded_and_noop_on_grey() {
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
        // With gains that make this exactly grey, refinement must not move them.
        let base = [2.0, 1.0, 2.5];
        let r = refine_wb(&img, w, h, 0, base);
        assert!((r[0] - 2.0).abs() < 1e-4 && (r[2] - 2.5).abs() < 1e-4);
        // A strongly mismatched estimate moves by at most 15%^0.2 and never wildly.
        let r2 = refine_wb(&img, w, h, 0, [1.0, 1.0, 1.0]);
        assert!(r2[0] > 1.0 && r2[0] < 1.05);
    }
}
