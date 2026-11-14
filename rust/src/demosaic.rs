//! Stage 3b: Malvar-He-Cutler 5x5 gradient-corrected linear demosaic, plus a 2x2 quad
//! binning path used when the sensor is too large to demosaic within a phone's memory.

use crate::meta::{cfa_color, B, G, R};
use rayon::prelude::*;

/// Planar float RGB image.
#[derive(Clone, Debug)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub c: [Vec<f32>; 3],
}

impl Rgb {
    pub fn new(w: usize, h: usize) -> Rgb {
        Rgb { w, h, c: [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]] }
    }
}

pub fn demosaic_mhc(bayer: &[f32], w: usize, h: usize, cfa: u8) -> Rgb {
    assert_eq!(bayer.len(), w * h);
    let px = |x: isize, y: isize| -> f32 {
        // Mirror at the borders so the CFA phase is preserved (offset of 2 keeps the colour).
        let m = |v: isize, n: isize| -> isize {
            if v < 0 {
                (-v).min(n - 1)
            } else if v >= n {
                (2 * (n - 1) - v).max(0)
            } else {
                v
            }
        };
        bayer[m(y, h as isize) as usize * w + m(x, w as isize) as usize]
    };
    let mut r = vec![0.0f32; w * h];
    let mut g = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    r.par_chunks_mut(w)
        .zip(g.par_chunks_mut(w).zip(b.par_chunks_mut(w)))
        .enumerate()
        .for_each(|(y, (rr, (gr, br)))| {
            let yi = y as isize;
            for x in 0..w {
                let xi = x as isize;
                let c = px(xi, yi);
                let n1 = px(xi, yi - 1);
                let s1 = px(xi, yi + 1);
                let w1 = px(xi - 1, yi);
                let e1 = px(xi + 1, yi);
                let n2 = px(xi, yi - 2);
                let s2 = px(xi, yi + 2);
                let w2 = px(xi - 2, yi);
                let e2 = px(xi + 2, yi);
                let diag = px(xi - 1, yi - 1) + px(xi + 1, yi - 1) + px(xi - 1, yi + 1) + px(xi + 1, yi + 1);
                let here = cfa_color(cfa, x, y);
                let mut out = [0.0f32; 3];
                out[here] = c;
                // Green at a red/blue site.
                let g_at_chroma = (4.0 * c + 2.0 * (n1 + s1 + w1 + e1) - (n2 + s2 + w2 + e2)) / 8.0;
                // Red/blue at a green site; horizontal if the wanted colour sits left/right.
                let h_interp = (5.0 * c + 4.0 * (w1 + e1) - (w2 + e2) - diag + 0.5 * (n2 + s2)) / 8.0;
                let v_interp = (5.0 * c + 4.0 * (n1 + s1) - (n2 + s2) - diag + 0.5 * (w2 + e2)) / 8.0;
                // Red at blue (or blue at red).
                let d_interp = (6.0 * c + 2.0 * diag - 1.5 * (n2 + s2 + w2 + e2)) / 8.0;
                if here == G {
                    let left = cfa_color(cfa, x + 1, y);
                    // Colour of the horizontal neighbours; the vertical ones are the other chroma.
                    let (horiz, vert) = if left == R { (R, B) } else { (B, R) };
                    out[horiz] = h_interp;
                    out[vert] = v_interp;
                } else {
                    out[G] = g_at_chroma;
                    let other = if here == R { B } else { R };
                    out[other] = d_interp;
                }
                rr[x] = out[R];
                gr[x] = out[G];
                br[x] = out[B];
            }
        });
    Rgb { w, h, c: [r, g, b] }
}

/// Collapse every 2x2 quad into one RGB pixel (R, mean of both greens, B). Half resolution.
pub fn bin_quads(bayer: &[f32], w: usize, h: usize, cfa: u8) -> Rgb {
    let (ow, oh) = (w / 2, h / 2);
    let mut out = Rgb::new(ow, oh);
    let [r, g, b] = &mut out.c;
    r.par_chunks_mut(ow)
        .zip(g.par_chunks_mut(ow).zip(b.par_chunks_mut(ow)))
        .enumerate()
        .for_each(|(y, (rr, (gr, br)))| {
            for x in 0..ow {
                let mut acc = [0.0f32; 3];
                let mut n = [0.0f32; 3];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (px, py) = (2 * x + dx, 2 * y + dy);
                        let ch = cfa_color(cfa, px, py);
                        acc[ch] += bayer[py * w + px];
                        n[ch] += 1.0;
                    }
                }
                rr[x] = acc[R] / n[R];
                gr[x] = acc[G] / n[G];
                br[x] = acc[B] / n[B];
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mosaic(w: usize, h: usize, cfa: u8, rgb: [f32; 3]) -> Vec<f32> {
        (0..w * h).map(|i| rgb[cfa_color(cfa, i % w, i / w)]).collect()
    }

    #[test]
    fn recovers_flat_colour_for_every_pattern() {
        for cfa in 0..4u8 {
            let (w, h) = (32, 24);
            let want = [0.7, 0.4, 0.2];
            let out = demosaic_mhc(&mosaic(w, h, cfa, want), w, h, cfa);
            for c in 0..3 {
                for (i, v) in out.c[c].iter().enumerate() {
                    assert!((v - want[c]).abs() < 1e-4, "cfa {} ch {} px {} = {}", cfa, c, i, v);
                }
            }
        }
    }

    #[test]
    fn follows_a_smooth_grey_ramp() {
        let (w, h) = (48, 32);
        let bayer: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let out = demosaic_mhc(&bayer, w, h, 0);
        // Interior pixels: all channels equal the ramp (kernels reproduce linear signals).
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let want = x as f32 / w as f32;
                for c in 0..3 {
                    assert!((out.c[c][y * w + x] - want).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn quad_binning_keeps_colour_and_halves_size() {
        let (w, h) = (16, 8);
        let want = [0.6, 0.3, 0.1];
        let out = bin_quads(&mosaic(w, h, 2, want), w, h, 2);
        assert_eq!((out.w, out.h), (8, 4));
        for c in 0..3 {
            assert!(out.c[c].iter().all(|v| (v - want[c]).abs() < 1e-6));
        }
    }
}
