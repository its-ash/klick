//! Stage 3b: Edge-adaptive directional Hamilton-Adams + Constant-Hue Color-Difference demosaic
//! with chromatic-median false-color suppression, plus co-registered 2x2 quad binning for
//! >17 MP sensors and Malvar-He-Cutler kept for comparison.

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

#[inline]
fn mirror(v: isize, n: isize) -> usize {
    if v < 0 {
        (-v).min(n - 1) as usize
    } else if v >= n {
        (2 * (n - 1) - v).max(0) as usize
    } else {
        v as usize
    }
}

/// 3-pass Pro Directional Edge-Adaptive Demosaic:
/// 1. Directional Hamilton-Adams Green interpolation with inverse-gradient-squared blending
///    (eliminates horizontal/vertical zippering along high-contrast edges).
/// 2. Constant-hue color-difference (`R - G`, `B - G`) chroma interpolation with diagonal
///    edge-adaptive weighting at opposite-chroma sites (eliminates false color fringes).
/// 3. Edge-gated 3x3 chromatic median on `R - G` and `B - G` at strong luminance edges to suppress
///    residual purple/green moire without softening flat colors.
pub fn demosaic_pro(bayer: &[f32], w: usize, h: usize, cfa: u8) -> Rgb {
    assert_eq!(bayer.len(), w * h);
    let wi = w as isize;
    let hi = h as isize;
    let px = |x: isize, y: isize| -> f32 { bayer[mirror(y, hi) * w + mirror(x, wi)] };

    // Pass 1: full-resolution edge-adaptive Green plane.
    let mut g = vec![0.0f32; w * h];
    g.par_chunks_mut(w).enumerate().for_each(|(y, gr)| {
        let yi = y as isize;
        for x in 0..w {
            if cfa_color(cfa, x, y) == G {
                gr[x] = bayer[y * w + x];
                continue;
            }
            let xi = x as isize;
            let c = px(xi, yi);
            let w1 = px(xi - 1, yi);
            let e1 = px(xi + 1, yi);
            let n1 = px(xi, yi - 1);
            let s1 = px(xi, yi + 1);
            let w2 = px(xi - 2, yi);
            let e2 = px(xi + 2, yi);
            let n2 = px(xi, yi - 2);
            let s2 = px(xi, yi + 2);
            let nw1 = px(xi - 1, yi - 1);
            let ne1 = px(xi + 1, yi - 1);
            let sw1 = px(xi - 1, yi + 1);
            let se1 = px(xi + 1, yi + 1);

            // Hamilton-Adams second-order corrected directional estimates.
            let gh = 0.5 * (w1 + e1) + 0.25 * (2.0 * c - w2 - e2);
            let gv = 0.5 * (n1 + s1) + 0.25 * (2.0 * c - n2 - s2);

            // Multi-row directional gradients for robust edge orientation detection.
            let dh = (w1 - e1).abs() + (2.0 * c - w2 - e2).abs()
                + 0.5 * ((nw1 - ne1).abs() + (sw1 - se1).abs());
            let dv = (n1 - s1).abs() + (2.0 * c - n2 - s2).abs()
                + 0.5 * ((nw1 - sw1).abs() + (ne1 - se1).abs());

            let eps = 1e-6;
            let wh = 1.0 / (dh * dh + eps);
            let wv = 1.0 / (dv * dv + eps);
            let dir_est = (wh * gh + wv * gv) / (wh + wv);
            // On near-Nyquist diagonal textures where both horizontal and vertical gradients are
            // simultaneously large and equal, blend gently toward isotropic 4-neighbor green to
            // suppress Bayer maze artifacts without softening horizontal/vertical edges.
            let g_iso = 0.25 * (w1 + e1 + n1 + s1);
            let d_min = dh.min(dv);
            let d_max = dh.max(dv);
            let diag_iso = if d_max > 0.04 {
                ((d_min / d_max - 0.65) / 0.30).clamp(0.0, 1.0) * 0.45
            } else {
                0.0
            };
            let est = dir_est + diag_iso * (g_iso - dir_est);
            // Soft overshoot guard prevents ringing on razor-sharp 1px edges.
            let lo = w1.min(e1).min(n1).min(s1);
            let hi_v = w1.max(e1).max(n1).max(s1);
            let margin = 0.20 * (hi_v - lo);
            gr[x] = est.clamp((lo - margin).max(0.0), hi_v + margin);
        }
    });

    // Pass 2: constant-hue color-difference planes (dr = R - G, db = B - G).
    let gp = |x: isize, y: isize| -> f32 { g[mirror(y, hi) * w + mirror(x, wi)] };
    let mut dr = vec![0.0f32; w * h];
    let mut db = vec![0.0f32; w * h];
    dr.par_chunks_mut(w)
        .zip(db.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (drr, dbr))| {
            let yi = y as isize;
            for x in 0..w {
                let xi = x as isize;
                let here = cfa_color(cfa, x, y);
                let gc = g[y * w + x];
                if here == G {
                    let h_col = cfa_color(cfa, x + 1, y);
                    let dh_val = 0.5 * ((px(xi - 1, yi) - gp(xi - 1, yi)) + (px(xi + 1, yi) - gp(xi + 1, yi)));
                    let dv_val = 0.5 * ((px(xi, yi - 1) - gp(xi, yi - 1)) + (px(xi, yi + 1) - gp(xi, yi + 1)));
                    if h_col == R {
                        drr[x] = dh_val;
                        dbr[x] = dv_val;
                    } else {
                        dbr[x] = dh_val;
                        drr[x] = dv_val;
                    }
                } else {
                    let own = bayer[y * w + x] - gc;
                    // Diagonal edge-adaptive interpolation of the opposite chroma difference.
                    let nw = px(xi - 1, yi - 1);
                    let ne = px(xi + 1, yi - 1);
                    let sw = px(xi - 1, yi + 1);
                    let se = px(xi + 1, yi + 1);
                    let d_nw = nw - gp(xi - 1, yi - 1);
                    let d_ne = ne - gp(xi + 1, yi - 1);
                    let d_sw = sw - gp(xi - 1, yi + 1);
                    let d_se = se - gp(xi + 1, yi + 1);
                    let grad_a = (nw - se).abs() + (gp(xi - 1, yi - 1) - gp(xi + 1, yi + 1)).abs();
                    let grad_b = (ne - sw).abs() + (gp(xi + 1, yi - 1) - gp(xi - 1, yi + 1)).abs();
                    let wa = 1.0 / (grad_a * grad_a + 1e-6);
                    let wb = 1.0 / (grad_b * grad_b + 1e-6);
                    let opp = (wa * 0.5 * (d_nw + d_se) + wb * 0.5 * (d_ne + d_sw)) / (wa + wb);
                    if here == R {
                        drr[x] = own;
                        dbr[x] = opp;
                    } else {
                        dbr[x] = own;
                        drr[x] = opp;
                    }
                }
            }
        });

    // Pass 3: edge-gated 3x3 chromatic median suppression of residual demosaic false color.
    let mut r = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    r.par_chunks_mut(w)
        .zip(b.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (rr, br))| {
            let yi = y as isize;
            for x in 0..w {
                let xi = x as isize;
                let idx = y * w + x;
                let gc = g[idx];
                // Measure local luma edge strength; flat interiors keep exact color differences.
                let edge = (gp(xi - 1, yi) - gp(xi + 1, yi)).abs().max((gp(xi, yi - 1) - gp(xi, yi + 1)).abs());
                let alpha = ((edge / (gc + 0.04) - 0.12) * 2.5).clamp(0.0, 0.55);
                if alpha <= 1e-3 {
                    rr[x] = (gc + dr[idx]).max(0.0);
                    br[x] = (gc + db[idx]).max(0.0);
                    continue;
                }
                let mut nr = [0.0f32; 9];
                let mut nb = [0.0f32; 9];
                let mut k = 0;
                for dy in -1isize..=1 {
                    let yy = mirror(yi + dy, hi) * w;
                    for dx in -1isize..=1 {
                        let ii = yy + mirror(xi + dx, wi);
                        nr[k] = dr[ii];
                        nb[k] = db[ii];
                        k += 1;
                    }
                }
                nr.sort_unstable_by(|a, b| a.total_cmp(b));
                nb.sort_unstable_by(|a, b| a.total_cmp(b));
                let fr = dr[idx] + alpha * (nr[4] - dr[idx]);
                let fb = db[idx] + alpha * (nb[4] - db[idx]);
                rr[x] = (gc + fr).max(0.0);
                br[x] = (gc + fb).max(0.0);
            }
        });
    Rgb { w, h, c: [r, g, b] }
}

/// Malvar-He-Cutler 5x5 gradient-corrected linear demosaic.
pub fn demosaic_mhc(bayer: &[f32], w: usize, h: usize, cfa: u8) -> Rgb {
    assert_eq!(bayer.len(), w * h);
    let px = |x: isize, y: isize| -> f32 { bayer[mirror(y, h as isize) * w + mirror(x, w as isize)] };
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
                let g_at_chroma = (4.0 * c + 2.0 * (n1 + s1 + w1 + e1) - (n2 + s2 + w2 + e2)) / 8.0;
                let h_interp = (5.0 * c + 4.0 * (w1 + e1) - (w2 + e2) - diag + 0.5 * (n2 + s2)) / 8.0;
                let v_interp = (5.0 * c + 4.0 * (n1 + s1) - (n2 + s2) - diag + 0.5 * (w2 + e2)) / 8.0;
                let d_interp = (6.0 * c + 2.0 * diag - 1.5 * (n2 + s2 + w2 + e2)) / 8.0;
                if here == G {
                    let left = cfa_color(cfa, x + 1, y);
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

/// Co-registered 2x2 quad binning to half resolution (used for >17 MP sensors).
/// Evaluates R, G, and B centered at the exact `(2*x + 0.5, 2*y + 0.5)` quad midpoint via
/// symmetric 4-neighbor bilinear weights for R/B so high-contrast edges have zero color shift.
pub fn bin_quads(bayer: &[f32], w: usize, h: usize, cfa: u8) -> Rgb {
    let (ow, oh) = (w / 2, h / 2);
    let mut out = Rgb::new(ow, oh);
    let [r, g, b] = &mut out.c;
    // Locate (dx, dy) of R and B inside each 2x2 quad (0 or 1).
    let (mut rx, mut ry, mut bx, mut by) = (0usize, 0usize, 1usize, 1usize);
    for dy in 0..2 {
        for dx in 0..2 {
            match cfa_color(cfa, dx, dy) {
                R => {
                    rx = dx;
                    ry = dy;
                }
                B => {
                    bx = dx;
                    by = dy;
                }
                _ => {}
            }
        }
    }
    let wi = w as isize;
    let hi = h as isize;
    let sample_at_center = |qx: usize, qy: usize, ox: usize, oy: usize| -> f32 {
        // Primary site is at (2*qx + ox, 2*qy + oy); step ±2 toward quad center (2*qx + 0.5, 2*qy + 0.5).
        let x0 = (2 * qx + ox) as isize;
        let y0 = (2 * qy + oy) as isize;
        let sx = if ox == 0 { 2isize } else { -2isize };
        let sy = if oy == 0 { 2isize } else { -2isize };
        let x1 = if x0 + sx >= 0 && x0 + sx < wi { x0 + sx } else { x0 };
        let y1 = if y0 + sy >= 0 && y0 + sy < hi { y0 + sy } else { y0 };
        let at = |x: isize, y: isize| bayer[y as usize * w + x as usize];
        let a = at(x0, y0);
        let b = at(x1, y0);
        let c = at(x0, y1);
        let d = at(x1, y1);
        0.5625 * a + 0.1875 * (b + c) + 0.0625 * d
    };
    r.par_chunks_mut(ow)
        .zip(g.par_chunks_mut(ow).zip(b.par_chunks_mut(ow)))
        .enumerate()
        .for_each(|(y, (rr, (gr, br)))| {
            let r0 = &bayer[(2 * y) * w..(2 * y + 1) * w];
            let r1 = &bayer[(2 * y + 1) * w..(2 * y + 2) * w];
            for x in 0..ow {
                rr[x] = sample_at_center(x, y, rx, ry);
                br[x] = sample_at_center(x, y, bx, by);
                let g1 = if rx == 0 && ry == 0 || bx == 0 && by == 0 {
                    r0[2 * x + 1] + r1[2 * x]
                } else {
                    r0[2 * x] + r1[2 * x + 1]
                };
                gr[x] = 0.5 * g1;
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
            for out in [demosaic_mhc(&mosaic(w, h, cfa, want), w, h, cfa), demosaic_pro(&mosaic(w, h, cfa, want), w, h, cfa)] {
                for c in 0..3 {
                    for (i, v) in out.c[c].iter().enumerate() {
                        assert!((v - want[c]).abs() < 1e-4, "cfa {} ch {} px {} = {}", cfa, c, i, v);
                    }
                }
            }
        }
    }

    #[test]
    fn pro_demosaic_eliminates_false_color_on_sharp_edge() {
        let (w, h) = (48, 32);
        // Pure grey vertical high-contrast step edge (R = G = B everywhere).
        let bayer: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.1 } else { 0.9 }).collect();
        let pro = demosaic_pro(&bayer, w, h, 0);
        let mhc = demosaic_mhc(&bayer, w, h, 0);
        let chroma_err = |img: &Rgb| -> f32 {
            let mut s = 0.0;
            for y in 4..h - 4 {
                for x in 20..28 {
                    let i = y * w + x;
                    let (r, g, b) = (img.c[0][i], img.c[1][i], img.c[2][i]);
                    s += (r - g).abs() + (b - g).abs();
                }
            }
            s
        };
        assert!(chroma_err(&pro) < chroma_err(&mhc) * 0.5, "pro {} vs mhc {}", chroma_err(&pro), chroma_err(&mhc));
    }

    #[test]
    fn follows_a_smooth_grey_ramp() {
        let (w, h) = (48, 32);
        let bayer: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let out = demosaic_pro(&bayer, w, h, 0);
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let want = x as f32 / w as f32;
                for c in 0..3 {
                    assert!((out.c[c][y * w + x] - want).abs() < 2e-3);
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
