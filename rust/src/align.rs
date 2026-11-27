//! Stage 1: burst alignment. Coarse-to-fine tile search on a Gaussian pyramid of the
//! 2x2-binned grey image with 3-candidate coarse inheritance, neighbor median regularization,
//! and sub-pixel quadratic parabola refinement.
//!
//! Convention: `alt(x + dx + sub_x, y + dy + sub_y)` matches `ref(x, y)`.

use crate::plane::{box_blur, downsample2, Plane};
use rayon::prelude::*;

/// Tile edge and stride at every pyramid level (50% overlap).
pub const TILE: usize = 16;
pub const STRIDE: usize = 8;
pub const LEVELS: usize = 4;

/// Search radius per level, level 0 = full (binned) resolution.
const RADIUS: [i32; LEVELS] = [2, 2, 4, 4];

pub struct Pyramid {
    pub levels: Vec<Plane>,
}

pub fn build_pyramid(grey: &Plane, n: usize) -> Pyramid {
    let mut levels = vec![grey.clone()];
    while levels.len() < n {
        let last = levels.last().unwrap();
        if last.w < 2 * TILE || last.h < 2 * TILE {
            break;
        }
        levels.push(downsample2(last));
    }
    Pyramid { levels }
}

/// Per-tile integer + sub-pixel offsets in level-0 (binned) pixels.
#[derive(Clone, Debug)]
pub struct Offsets {
    pub gw: usize,
    pub gh: usize,
    pub v: Vec<(i32, i32)>,
    /// Sub-pixel refinement in `[-0.5, 0.5]` added to `v`.
    pub sub: Vec<(f32, f32)>,
}

/// Tiles start at -STRIDE so every pixel is covered by exactly two tiles per axis.
pub fn grid_dims(w: usize, h: usize) -> (usize, usize) {
    (w.div_ceil(STRIDE) + 1, h.div_ceil(STRIDE) + 1)
}

impl Offsets {
    pub fn zeros(w: usize, h: usize) -> Offsets {
        let (gw, gh) = grid_dims(w, h);
        Offsets { gw, gh, v: vec![(0, 0); gw * gh], sub: vec![(0.0, 0.0); gw * gh] }
    }

    #[inline]
    pub fn exact(&self, idx: usize) -> (f32, f32) {
        let (dx, dy) = self.v[idx];
        let (sx, sy) = self.sub[idx];
        (dx as f32 + sx, dy as f32 + sy)
    }
}

#[inline]
fn tile_cost(r: &Plane, a: &Plane, x0: isize, y0: isize, ox: isize, oy: isize, l1: bool) -> f32 {
    let t = TILE as isize;
    let inside = x0 >= 0
        && y0 >= 0
        && x0 + t <= r.w as isize
        && y0 + t <= r.h as isize
        && x0 + ox >= 0
        && y0 + oy >= 0
        && x0 + ox + t <= a.w as isize
        && y0 + oy + t <= a.h as isize;
    let mut s = 0.0f32;
    if inside {
        for v in 0..TILE {
            let ri = (y0 as usize + v) * r.w + x0 as usize;
            let ai = ((y0 + oy) as usize + v) * a.w + (x0 + ox) as usize;
            let rr = &r.d[ri..ri + TILE];
            let aa = &a.d[ai..ai + TILE];
            if l1 {
                for (p, q) in rr.iter().zip(aa) {
                    s += (p - q).abs();
                }
            } else {
                for (p, q) in rr.iter().zip(aa) {
                    let d = p - q;
                    s += d * d;
                }
            }
        }
    } else {
        for v in 0..t {
            for u in 0..t {
                let d = r.at(x0 + u, y0 + v) - a.at(x0 + u + ox, y0 + v + oy);
                s += if l1 { d.abs() } else { d * d };
            }
        }
    }
    s
}

/// 1D quadratic sub-pixel minimum from 3 equally spaced L2 costs `(c_minus, c_zero, c_plus)`.
#[inline]
fn quad_min(cm: f32, c0: f32, cp: f32) -> f32 {
    let denom = cm - 2.0 * c0 + cp;
    if denom <= 1e-9 {
        return 0.0;
    }
    (0.5 * (cm - cp) / denom).clamp(-0.5, 0.5)
}

/// Computes the global median tile offset across the frame to anchor flat skies, 1D edges,
/// and periodic repeating patterns against stray local minima.
fn global_median(v: &[(i32, i32)]) -> (i32, i32) {
    if v.is_empty() {
        return (0, 0);
    }
    let mut xs: Vec<i32> = v.iter().map(|p| p.0).collect();
    let mut ys: Vec<i32> = v.iter().map(|p| p.1).collect();
    let mid = v.len() / 2;
    let (_, mx, _) = xs.select_nth_unstable(mid);
    let (_, my, _) = ys.select_nth_unstable(mid);
    (*mx, *my)
}

/// Smooths stray outlier tile offsets (e.g. in flat sky/wall tiles or periodic textures) toward the
/// 3x3 neighbor median or global camera-shake median whenever either has nearly as good a cost as
/// the tile's noisy local minimum.
fn regularize_offsets(r: &Plane, a: &Plane, gw: usize, gh: usize, v: &[(i32, i32)], l1: bool) -> Vec<(i32, i32)> {
    let gmed = global_median(v);
    (0..gw * gh)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx % gw, idx / gw);
            let cur = v[idx];
            let mut xs = [0i32; 9];
            let mut ys = [0i32; 9];
            let mut n = 0usize;
            for dj in -1isize..=1 {
                let nj = (j as isize + dj).clamp(0, gh as isize - 1) as usize;
                for di in -1isize..=1 {
                    let ni = (i as isize + di).clamp(0, gw as isize - 1) as usize;
                    let p = v[nj * gw + ni];
                    xs[n] = p.0;
                    ys[n] = p.1;
                    n += 1;
                }
            }
            xs[..n].sort_unstable();
            ys[..n].sort_unstable();
            let med = (xs[n / 2], ys[n / 2]);
            let x0 = (i * STRIDE) as isize - STRIDE as isize;
            let y0 = (j * STRIDE) as isize - STRIDE as isize;
            let c_cur = tile_cost(r, a, x0, y0, cur.0 as isize, cur.1 as isize, l1);
            // First check global median (fixes periodic stripes & large flat sky/horizon patches).
            if (cur.0 - gmed.0).abs() + (cur.1 - gmed.1).abs() > 0 {
                let c_g = tile_cost(r, a, x0, y0, gmed.0 as isize, gmed.1 as isize, l1);
                if c_g <= c_cur * 1.12 + 1e-5 {
                    return gmed;
                }
            }
            if (cur.0 - med.0).abs() + (cur.1 - med.1).abs() <= 1 {
                return cur;
            }
            let c_med = tile_cost(r, a, x0, y0, med.0 as isize, med.1 as isize, l1);
            if c_med <= c_cur * 1.08 + 1e-5 {
                med
            } else {
                cur
            }
        })
        .collect()
}

pub fn align(refp: &Pyramid, alt: &Pyramid) -> Offsets {
    let n = refp.levels.len().min(alt.levels.len());
    let mut prev: Option<(usize, usize, Vec<(i32, i32)>, (i32, i32))> = None;
    for l in (0..n).rev() {
        let r = &refp.levels[l];
        let a = &alt.levels[l];
        let (gw, gh) = grid_dims(r.w, r.h);
        let rad = RADIUS[l];
        let l1 = l == 0;
        let raw: Vec<(i32, i32)> = (0..gw * gh)
            .into_par_iter()
            .map(|idx| {
                let (i, j) = (idx % gw, idx / gw);
                let x0 = (i * STRIDE) as isize - STRIDE as isize;
                let y0 = (j * STRIDE) as isize - STRIDE as isize;
                // Inherit up to 4 coarse candidates: parent, nearest horizontal/vertical coarse
                // neighbors (Hasinoff HDR+), plus the previous level's global median motion.
                let (px, py) = match &prev {
                    Some((pgw, pgh, pv, pgmed)) => {
                        let pi = (i / 2).min(*pgw - 1);
                        let pj = (j / 2).min(*pgh - 1);
                        let ni = if i % 2 == 0 { pi.saturating_sub(1) } else { (pi + 1).min(*pgw - 1) };
                        let nj = if j % 2 == 0 { pj.saturating_sub(1) } else { (pj + 1).min(*pgh - 1) };
                        let cands = [pv[pj * *pgw + pi], pv[pj * *pgw + ni], pv[nj * *pgw + pi], *pgmed];
                        let mut best_p = cands[0];
                        let mut best_c = f32::MAX;
                        for (k, &(ox, oy)) in cands.iter().enumerate() {
                            if k > 0 && cands[..k].contains(&(ox, oy)) {
                                continue;
                            }
                            let c = tile_cost(r, a, x0, y0, (ox * 2) as isize, (oy * 2) as isize, l1);
                            if c < best_c {
                                best_c = c;
                                best_p = (ox, oy);
                            }
                        }
                        (best_p.0 * 2, best_p.1 * 2)
                    }
                    None => (0, 0),
                };
                let mut best = (px, py);
                let mut best_cost = f32::MAX;
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        let (cx, cy) = (px + dx, py + dy);
                        let c = tile_cost(r, a, x0, y0, cx as isize, cy as isize, l1);
                        let c = c * (1.0 + 0.003 * (dx.abs() + dy.abs()) as f32);
                        if c < best_cost {
                            best_cost = c;
                            best = (cx, cy);
                        }
                    }
                }
                best
            })
            .collect();
        let v = if gw >= 3 && gh >= 3 { regularize_offsets(r, a, gw, gh, &raw, l1) } else { raw };
        let gmed = global_median(&v);
        prev = Some((gw, gh, v, gmed));
    }
    let (gw, gh, v, _) = prev.expect("pyramid has at least one level");
    let r0 = &refp.levels[0];
    let a0 = &alt.levels[0];
    // Sub-pixel quadratic refinement on the L2 cost surface around the winning integer offset.
    let sub: Vec<(f32, f32)> = (0..gw * gh)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx % gw, idx / gw);
            let x0 = (i * STRIDE) as isize - STRIDE as isize;
            let y0 = (j * STRIDE) as isize - STRIDE as isize;
            let (ox, oy) = v[idx];
            let (oxi, oyi) = (ox as isize, oy as isize);
            let c0 = tile_cost(r0, a0, x0, y0, oxi, oyi, false);
            if c0 < 1e-10 {
                return (0.0, 0.0);
            }
            let cxm = tile_cost(r0, a0, x0, y0, oxi - 1, oyi, false);
            let cxp = tile_cost(r0, a0, x0, y0, oxi + 1, oyi, false);
            let cym = tile_cost(r0, a0, x0, y0, oxi, oyi - 1, false);
            let cyp = tile_cost(r0, a0, x0, y0, oxi, oyi + 1, false);
            (quad_min(cxm, c0, cxp), quad_min(cym, c0, cyp))
        })
        .collect();
    debug_assert_eq!((gw, gh), grid_dims(r0.w, r0.h));
    Offsets { gw, gh, v, sub }
}

/// Pick the sharpest of the first `limit` frames using noise-robust center-weighted 2D Tenengrad
/// (horizontal + vertical gradient energy on a lightly 3x3-smoothed plane so sensor noise does not
/// masquerade as sharpness).
pub fn pick_reference(greys: &[Plane], limit: usize) -> usize {
    let n = greys.len().min(limit.max(1));
    if n <= 1 {
        return 0;
    }
    let mut best = (0usize, -1.0f64);
    for (i, g) in greys.iter().take(n).enumerate() {
        let sm = box_blur(g, 1);
        let (w, h) = (sm.w, sm.h);
        let (x0, x1) = (w / 6, w - w / 6);
        let (y0, y1) = (h / 6, h - h / 6);
        let score: f64 = (y0.max(1)..y1.min(h.saturating_sub(1)))
            .into_par_iter()
            .map(|y| {
                let prev = &sm.d[(y - 1) * w..y * w];
                let cur = &sm.d[y * w..(y + 1) * w];
                let next = &sm.d[(y + 1) * w..(y + 2) * w];
                let mut acc = 0.0f64;
                for x in x0.max(1)..x1.min(w.saturating_sub(1)) {
                    let gx = (cur[x + 1] - cur[x - 1]) as f64;
                    let gy = (next[x] - prev[x]) as f64;
                    acc += gx * gx + gy * gy;
                }
                acc
            })
            .sum();
        if score > best.1 * 1.02 || best.1 < 0.0 {
            best = (i, score);
        }
    }
    best.0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::testutil::{shifted, textured};

    #[test]
    fn aligns_known_shift() {
        let (w, h) = (192, 160);
        let base = textured(w + 64, h + 64);
        let refp = shifted(&base, w, h, 32, 32, 0.0, 1);
        // alt(x + dx, y + dy) == ref(x, y) with (dx, dy) = (5, -3).
        let alt = shifted(&base, w, h, 32 - 5, 32 + 3, 0.02, 2);
        let pr = build_pyramid(&refp, LEVELS);
        let pa = build_pyramid(&alt, LEVELS);
        let o = align(&pr, &pa);
        let interior: Vec<_> = (2..o.gh - 2)
            .flat_map(|j| (2..o.gw - 2).map(move |i| (i, j)))
            .map(|(i, j)| o.v[j * o.gw + i])
            .collect();
        let good = interior.iter().filter(|&&p| p == (5, -3)).count();
        assert!(good * 10 >= interior.len() * 9, "{} of {}", good, interior.len());
    }

    #[test]
    fn subpixel_alignment_recovers_fractional_shift() {
        let (w, h) = (160, 128);
        let base = textured(w + 32, h + 32);
        let refp = shifted(&base, w, h, 16, 16, 0.0, 1);
        let mut alt = Plane::new(w, h);
        // alt(x + 2.35, y - 1.40) == ref(x, y) -> alt(x, y) = base(16 + x - 2.35, 16 + y + 1.40)
        for y in 0..h {
            for x in 0..w {
                alt.d[y * w + x] = base.sample_catmull_rom(16.0 + x as f32 - 2.35, 16.0 + y as f32 + 1.40);
            }
        }
        let o = align(&build_pyramid(&refp, LEVELS), &build_pyramid(&alt, LEVELS));
        let mut errs = Vec::new();
        for j in 2..o.gh - 2 {
            for i in 2..o.gw - 2 {
                let (ex, ey) = o.exact(j * o.gw + i);
                errs.push((ex - 2.35).abs() + (ey + 1.40).abs());
            }
        }
        let mean_err = errs.iter().sum::<f32>() / errs.len() as f32;
        assert!(mean_err < 0.2, "subpixel mean err {}", mean_err);
    }

    #[test]
    fn identical_frames_align_to_zero() {
        let p = textured(128, 96);
        let pr = build_pyramid(&p, LEVELS);
        let o = align(&pr, &pr);
        assert!(o.v.iter().all(|&v| v == (0, 0)));
        assert!(o.sub.iter().all(|&(sx, sy)| sx.abs() < 1e-4 && sy.abs() < 1e-4));
    }

    #[test]
    fn reference_is_sharpest() {
        let sharp = textured(64, 64);
        let blur = crate::plane::gaussian_blur(&sharp, 2.0);
        assert_eq!(pick_reference(&[blur.clone(), sharp, blur], 3), 1);
    }
}
