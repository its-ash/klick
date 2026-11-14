//! Stage 2: robust temporal merge. Each aligned alternate frame is blended into the
//! reference tile by tile. The per-pixel weight follows the sensor noise model
//! (a Wiener-style shrinkage: differences of a few sigma are noise and get averaged,
//! larger differences are motion or ghosts and get rejected). Tiles are blended with a
//! raised-cosine (Hann) window at 50% overlap, so the window weights sum to exactly 1.

use crate::align::{grid_dims, Offsets};
use rayon::prelude::*;

/// Variance of a normalised linear sample `x` is `s * x + o`.
#[derive(Clone, Copy, Debug)]
pub struct NoiseModel {
    pub s: f32,
    pub o: f32,
}

impl NoiseModel {
    #[inline]
    pub fn var(&self, x: f32) -> f32 {
        (self.s * x.max(0.0) + self.o).max(1e-9)
    }
}

/// Difference-to-noise ratio (in variance units) at which an alternate pixel is down-weighted to 50%.
const REJECT_Q: f32 = 8.0;

pub struct Merger {
    w: usize,
    h: usize,
    tile: usize,
    stride: usize,
    acc_a: Vec<f32>,
    acc_w: Vec<f32>,
}

fn hann(tile: usize) -> Vec<f32> {
    (0..tile)
        .map(|t| 0.5 - 0.5 * (std::f32::consts::TAU * (t as f32 + 0.5) / tile as f32).cos())
        .collect()
}

impl Merger {
    /// `tile` is the tile edge in plane pixels (even); stride is tile / 2.
    pub fn new(w: usize, h: usize, tile: usize) -> Merger {
        assert!(tile >= 4 && tile % 2 == 0);
        Merger { w, h, tile, stride: tile / 2, acc_a: vec![0.0; w * h], acc_w: vec![0.0; w * h] }
    }

    /// Blend `alt` into the accumulators. `scale` converts offset units (binned pixels)
    /// to plane pixels (2 for Bayer or full-res luma, 1 for half-res chroma).
    pub fn add(&mut self, refp: &[f32], alt: &[f32], offs: &Offsets, scale: i32, noise: &NoiseModel) {
        let (w, h, tile, stride) = (self.w, self.h, self.tile, self.stride);
        assert_eq!(refp.len(), w * h);
        assert_eq!(alt.len(), w * h);
        let gw = w.div_ceil(stride) + 1;
        let gh = h.div_ceil(stride) + 1;
        assert!(offs.gw == gw && offs.gh == gh, "offset grid {}x{} does not match plane grid {}x{}", offs.gw, offs.gh, gw, gh);
        let win = hann(tile);
        let at = |x: isize, y: isize| -> f32 {
            alt[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize]
        };
        let off = |i: usize, j: usize| -> (isize, isize) {
            let (dx, dy) = offs.v[j * gw + i];
            ((dx * scale) as isize, (dy * scale) as isize)
        };

        // Pass 1: mean difference-to-noise ratio per tile.
        let ratios: Vec<f32> = (0..gw * gh)
            .into_par_iter()
            .map(|idx| {
                let (i, j) = (idx % gw, idx / gw);
                let x0 = (i * stride) as isize - stride as isize;
                let y0 = (j * stride) as isize - stride as isize;
                let (ox, oy) = off(i, j);
                let mut s = 0.0f32;
                let mut n = 0u32;
                for v in 0..tile as isize {
                    let y = y0 + v;
                    if y < 0 || y >= h as isize {
                        continue;
                    }
                    for u in 0..tile as isize {
                        let x = x0 + u;
                        if x < 0 || x >= w as isize {
                            continue;
                        }
                        let r = refp[y as usize * w + x as usize];
                        let d = at(x + ox, y + oy) - r;
                        s += d * d / noise.var(r);
                        n += 1;
                    }
                }
                if n == 0 {
                    0.0
                } else {
                    s / n as f32
                }
            })
            .collect();

        // Pass 2: accumulate by horizontal bands (band b is covered by tile rows b and b + 1).
        let band_px = stride * w;
        self.acc_a
            .par_chunks_mut(band_px)
            .zip(self.acc_w.par_chunks_mut(band_px))
            .enumerate()
            .for_each(|(b, (aa, aw))| {
                let ybase = b * stride;
                for j in [b, b + 1] {
                    if j >= gh {
                        continue;
                    }
                    let y0 = (j * stride) as isize - stride as isize;
                    for i in 0..gw {
                        let x0 = (i * stride) as isize - stride as isize;
                        let (ox, oy) = off(i, j);
                        let tile_q = ratios[j * gw + i];
                        let xs = x0.max(0) as usize;
                        let xe = ((x0 + tile as isize).min(w as isize)).max(0) as usize;
                        for yy in 0..stride {
                            let y = ybase + yy;
                            if y >= h {
                                break;
                            }
                            let wy = win[(y as isize - y0) as usize];
                            for x in xs..xe {
                                let wx = win[(x as isize - x0) as usize];
                                let r = refp[y * w + x];
                                let a = at(x as isize + ox, y as isize + oy);
                                let d = a - r;
                                let q = 0.5 * d * d / noise.var(r) + 0.5 * tile_q;
                                let t = q / REJECT_Q;
                                let wk = 1.0 / (1.0 + t * t);
                                let hw = wx * wy * wk;
                                aa[yy * w + x] += hw * a;
                                aw[yy * w + x] += hw;
                            }
                        }
                    }
                }
            });
    }

    pub fn finish(self, refp: &[f32]) -> Vec<f32> {
        refp.par_iter()
            .zip(self.acc_a.par_iter().zip(self.acc_w.par_iter()))
            .map(|(r, (a, wt))| (r + a) / (1.0 + wt))
            .collect()
    }
}

/// Convenience: merge whole burst planes given precomputed offsets (used by tests and the pipeline).
pub fn merge_planes(refp: &[f32], alts: &[(&[f32], &Offsets)], w: usize, h: usize, tile: usize, scale: i32, noise: &NoiseModel) -> Vec<f32> {
    let mut m = Merger::new(w, h, tile);
    for (a, o) in alts {
        m.add(refp, a, o, scale, noise);
    }
    m.finish(refp)
}

pub fn grid_for(w: usize, h: usize) -> (usize, usize) {
    grid_dims(w, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{variance, Rng};

    fn noisy(w: usize, h: usize, level: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..w * h).map(|_| level + sigma * r.gauss()).collect()
    }

    #[test]
    fn merge_reduces_noise_variance() {
        let (w, h) = (128, 96);
        let sigma = 0.03;
        let noise = NoiseModel { s: 0.0, o: sigma * sigma };
        let refp = noisy(w, h, 0.4, sigma, 1);
        let alts: Vec<Vec<f32>> = (0..7).map(|k| noisy(w, h, 0.4, sigma, 10 + k)).collect();
        let offs = Offsets::zeros(w / 2, h / 2);
        let list: Vec<(&[f32], &Offsets)> = alts.iter().map(|a| (a.as_slice(), &offs)).collect();
        let out = merge_planes(&refp, &list, w, h, 32, 2, &noise);
        let (v0, v1) = (variance(&refp), variance(&out));
        assert!(v1 < v0 / 3.5, "variance {} -> {}", v0, v1);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - 0.4).abs() < 0.005);
    }

    #[test]
    fn merge_rejects_ghosts() {
        let (w, h) = (128, 96);
        let sigma = 0.01;
        let noise = NoiseModel { s: 0.0, o: sigma * sigma };
        let refp = noisy(w, h, 0.3, sigma, 1);
        let mut alt = noisy(w, h, 0.3, sigma, 2);
        // A bright moving object only in the alternate frame.
        for y in 30..60 {
            for x in 40..80 {
                alt[y * w + x] = 0.9;
            }
        }
        let offs = Offsets::zeros(w / 2, h / 2);
        let out = merge_planes(&refp, &[(&alt, &offs)], w, h, 32, 2, &noise);
        let centre = out[45 * w + 60];
        assert!(centre < 0.35, "ghost leaked: {}", centre);
    }

    #[test]
    fn identical_frames_are_unchanged() {
        let (w, h) = (64, 64);
        let refp = noisy(w, h, 0.5, 0.0, 1);
        let offs = Offsets::zeros(w / 2, h / 2);
        let noise = NoiseModel { s: 0.0, o: 1e-4 };
        let out = merge_planes(&refp, &[(&refp, &offs)], w, h, 32, 2, &noise);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-5));
    }
}
