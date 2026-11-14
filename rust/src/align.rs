//! Stage 1: burst alignment. Coarse-to-fine tile search on a Gaussian pyramid of the
//! 2x2-binned grey image. Offsets are whole pixels of the binned image, so they are
//! always even in Bayer pixels and keep the CFA phase.
//!
//! Convention: `alt(x + dx, y + dy)` matches `ref(x, y)`.

use crate::plane::{downsample2, Plane};
use rayon::prelude::*;

/// Tile edge and stride at every pyramid level (50% overlap).
pub const TILE: usize = 16;
pub const STRIDE: usize = 8;
pub const LEVELS: usize = 4;

/// Search radius per level, level 0 = full (binned) resolution.
const RADIUS: [i32; LEVELS] = [1, 2, 4, 4];

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

/// Per-tile offsets in level-0 (binned) pixels.
#[derive(Clone, Debug)]
pub struct Offsets {
    pub gw: usize,
    pub gh: usize,
    pub v: Vec<(i32, i32)>,
}

/// Tiles start at -STRIDE so every pixel is covered by exactly two tiles per axis.
pub fn grid_dims(w: usize, h: usize) -> (usize, usize) {
    (w.div_ceil(STRIDE) + 1, h.div_ceil(STRIDE) + 1)
}

impl Offsets {
    pub fn zeros(w: usize, h: usize) -> Offsets {
        let (gw, gh) = grid_dims(w, h);
        Offsets { gw, gh, v: vec![(0, 0); gw * gh] }
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

pub fn align(refp: &Pyramid, alt: &Pyramid) -> Offsets {
    let n = refp.levels.len().min(alt.levels.len());
    let mut prev: Option<Offsets> = None; // offsets of the coarser level, in that level's pixels
    for l in (0..n).rev() {
        let r = &refp.levels[l];
        let a = &alt.levels[l];
        let (gw, gh) = grid_dims(r.w, r.h);
        let rad = RADIUS[l];
        let l1 = l == 0;
        let v: Vec<(i32, i32)> = (0..gw * gh)
            .into_par_iter()
            .map(|idx| {
                let (i, j) = (idx % gw, idx / gw);
                let x0 = (i * STRIDE) as isize - STRIDE as isize;
                let y0 = (j * STRIDE) as isize - STRIDE as isize;
                let (px, py) = match &prev {
                    Some(p) => {
                        let pi = (i / 2).min(p.gw - 1);
                        let pj = (j / 2).min(p.gh - 1);
                        let (ox, oy) = p.v[pj * p.gw + pi];
                        (ox * 2, oy * 2)
                    }
                    None => (0, 0),
                };
                let mut best = (px, py);
                let mut best_cost = f32::MAX;
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        let (cx, cy) = (px + dx, py + dy);
                        let c = tile_cost(r, a, x0, y0, cx as isize, cy as isize, l1);
                        let c = c * (1.0 + 0.004 * (dx.abs() + dy.abs()) as f32);
                        if c < best_cost {
                            best_cost = c;
                            best = (cx, cy);
                        }
                    }
                }
                best
            })
            .collect();
        prev = Some(Offsets { gw, gh, v });
    }
    let mut out = prev.expect("pyramid has at least one level");
    // If the pyramid was shorter than LEVELS the finest level is still level 0.
    let (gw, gh) = grid_dims(refp.levels[0].w, refp.levels[0].h);
    debug_assert_eq!((out.gw, out.gh), (gw, gh));
    out.gw = gw;
    out.gh = gh;
    out
}

/// Pick the sharpest of the first `limit` frames (mean absolute gradient) as the reference.
pub fn pick_reference(greys: &[Plane], limit: usize) -> usize {
    let n = greys.len().min(limit.max(1));
    let mut best = (0usize, -1.0f32);
    for (i, g) in greys.iter().take(n).enumerate() {
        let s: f32 = g
            .d
            .par_chunks(g.w)
            .map(|row| row.windows(2).map(|p| (p[1] - p[0]).abs()).sum::<f32>())
            .sum();
        if s > best.1 * 1.02 || best.1 < 0.0 {
            best = (i, s);
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
    fn identical_frames_align_to_zero() {
        let p = textured(128, 96);
        let pr = build_pyramid(&p, LEVELS);
        let o = align(&pr, &pr);
        assert!(o.v.iter().all(|&v| v == (0, 0)));
    }

    #[test]
    fn reference_is_sharpest() {
        let sharp = textured(64, 64);
        let blur = crate::plane::gaussian_blur(&sharp, 2.0);
        assert_eq!(pick_reference(&[blur.clone(), sharp, blur], 3), 1);
    }
}
