//! Stage 2: HDR+ frequency-domain robust temporal merge (Hasinoff et al., SIGGRAPH 2016).
//!
//! Each aligned alternate frame is resampled with sub-pixel Catmull-Rom accuracy per channel
//! (keeping Bayer CFA phase intact by operating on the 4 R/Gr/Gb/B sub-planes for RAW, or directly
//! on luma/chroma planes for YUV), transformed via an orthonormal 16x16 2D-DCT-II, and merged
//! coefficient-by-coefficient with a Wiener shrinkage weight from the calibrated sensor noise model.
//! Low-frequency noise averages cleanly across all frames while moving objects / ghosts are rejected
//! per frequency and per tile without blurring fine detail. Tiles are blended with a raised-cosine
//! Hann synthesis window at 50% overlap (exact partition of unity).

use crate::align::{grid_dims, Offsets};
use crate::plane::catmull_weights;
use rayon::prelude::*;
use std::sync::OnceLock;

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

/// Orthonormal DCT-II Wiener temporal rejection factor (Hasinoff et al. c ≈ 8).
const WIENER_C: f32 = 8.0;

/// Cached orthonormal 16x16 DCT-II basis matrix `D[k][n]`, where `D^T` is the exact inverse IDCT-II.
fn dct16_basis() -> &'static [[f32; 16]; 16] {
    static BASIS: OnceLock<[[f32; 16]; 16]> = OnceLock::new();
    BASIS.get_or_init(|| {
        let mut m = [[0.0f32; 16]; 16];
        let s0 = (1.0f32 / 16.0).sqrt();
        let sk = (2.0f32 / 16.0).sqrt();
        for k in 0..16 {
            let scale = if k == 0 { s0 } else { sk };
            for n in 0..16 {
                let angle = std::f32::consts::PI * (n as f32 + 0.5) * k as f32 / 16.0;
                m[k][n] = scale * angle.cos();
            }
        }
        m
    })
}

/// Forward orthonormal 16x16 2D-DCT-II: `out = D * tile * D^T`.
#[inline]
fn dct2_16(src: &[f32; 256]) -> [f32; 256] {
    let d = dct16_basis();
    let mut tmp = [0.0f32; 256];
    for y in 0..16 {
        let row = &src[y * 16..(y + 1) * 16];
        for u in 0..16 {
            let bu = &d[u];
            let mut s = 0.0f32;
            for x in 0..16 {
                s += row[x] * bu[x];
            }
            tmp[y * 16 + u] = s;
        }
    }
    let mut out = [0.0f32; 256];
    for v in 0..16 {
        let bv = &d[v];
        for u in 0..16 {
            let mut s = 0.0f32;
            for y in 0..16 {
                s += tmp[y * 16 + u] * bv[y];
            }
            out[v * 16 + u] = s;
        }
    }
    out
}

/// Inverse orthonormal 16x16 2D-DCT-II: `out = D^T * coeff * D`.
#[inline]
fn idct2_16(src: &[f32; 256]) -> [f32; 256] {
    let d = dct16_basis();
    let mut tmp = [0.0f32; 256];
    for v in 0..16 {
        let row = &src[v * 16..(v + 1) * 16];
        for x in 0..16 {
            let mut s = 0.0f32;
            for u in 0..16 {
                s += row[u] * d[u][x];
            }
            tmp[v * 16 + x] = s;
        }
    }
    let mut out = [0.0f32; 256];
    for y in 0..16 {
        for x in 0..16 {
            let mut s = 0.0f32;
            for v in 0..16 {
                s += tmp[v * 16 + x] * d[v][y];
            }
            out[y * 16 + x] = s;
        }
    }
    out
}

fn hann16() -> &'static [f32; 16] {
    static WIN: OnceLock<[f32; 16]> = OnceLock::new();
    WIN.get_or_init(|| {
        std::array::from_fn(|t| 0.5 - 0.5 * (std::f32::consts::TAU * (t as f32 + 0.5) / 16.0).cos())
    })
}

#[inline]
fn sample_subplane(buf: &[f32], w: usize, h: usize, fx: f32, fy: f32) -> f32 {
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let xi = x0 as isize;
    let yi = y0 as isize;
    let at = |x: isize, y: isize| -> f32 {
        buf[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize]
    };
    if tx.abs() < 1e-3 && ty.abs() < 1e-3 {
        return at(xi, yi);
    }
    // Clamped Catmull-Rom keeps micro-detail crisp without ringing around high-contrast edges.
    let wx = catmull_weights(tx);
    let wy = catmull_weights(ty);
    let mut acc = 0.0f32;
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for j in 0..4isize {
        let mut row = 0.0f32;
        for i in 0..4isize {
            let v = at(xi + i - 1, yi + j - 1);
            if (i == 1 || i == 2) && (j == 1 || j == 2) {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            row += wx[i as usize] * v;
        }
        acc += wy[j as usize] * row;
    }
    acc.clamp(lo, hi)
}

/// Frequency-domain HDR+ Wiener merger for one single-channel plane (either one of the 4 Bayer
/// sub-planes at `(w/2, h/2)`, or a half-res chroma plane, or a full-res YUV luma plane).
///
/// Accumulates per-tile 16x16 DCT coefficients across all alternate frames in a burst, then runs
/// a single IDCT + Hann window synthesis in `finish()`, minimizing compute and eliminating
/// spatial-domain misalignment blur.
pub struct SubplaneMerger {
    w: usize,
    h: usize,
    gw: usize,
    gh: usize,
    /// Per-tile DCT of the reference frame (`gw * gh * 256`).
    ref_dct: Vec<f32>,
    /// Accumulated weighted alternate DCT delta sum (`sum((1 - A_z) * A_f)`).
    acc_dct: Vec<f32>,
    /// Accumulated temporal weight sum (`sum(1 - A_z)`).
    acc_wt: Vec<f32>,
    /// Per-tile noise variance from the reference tile mean.
    tile_var: Vec<f32>,
}

impl SubplaneMerger {
    pub fn new(refp: &[f32], w: usize, h: usize, noise: &NoiseModel) -> Self {
        assert_eq!(refp.len(), w * h);
        let (gw, gh) = grid_dims(w, h);
        let ntiles = gw * gh;
        let mut ref_dct = vec![0.0f32; ntiles * 256];
        let mut tile_var = vec![0.0f32; ntiles];
        ref_dct
            .par_chunks_mut(256)
            .zip(tile_var.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (dst, tv))| {
                let (i, j) = (idx % gw, idx / gw);
                let x0 = (i * 8) as isize - 8;
                let y0 = (j * 8) as isize - 8;
                let mut tile = [0.0f32; 256];
                let mut sum = 0.0f32;
                for v in 0..16isize {
                    let yy = (y0 + v).clamp(0, h as isize - 1) as usize;
                    for u in 0..16isize {
                        let xx = (x0 + u).clamp(0, w as isize - 1) as usize;
                        let val = refp[yy * w + xx];
                        tile[v as usize * 16 + u as usize] = val;
                        sum += val;
                    }
                }
                let mean = sum / 256.0;
                *tv = noise.var(mean);
                dst.copy_from_slice(&dct2_16(&tile));
            });
        SubplaneMerger {
            w,
            h,
            gw,
            gh,
            ref_dct,
            acc_dct: vec![0.0f32; ntiles * 256],
            acc_wt: vec![0.0f32; ntiles * 256],
            tile_var,
        }
    }

    /// Accumulate one aligned alternate sub-plane into the DCT Wiener accumulators.
    /// `offs` must match `(gw, gh)` in sub-plane pixel units.
    pub fn add(&mut self, alt: &[f32], offs: &Offsets) {
        let (w, h, gw, gh) = (self.w, self.h, self.gw, self.gh);
        assert_eq!(alt.len(), w * h);
        assert_eq!((offs.gw, offs.gh), (gw, gh));
        self.acc_dct
            .par_chunks_mut(256)
            .zip(self.acc_wt.par_chunks_mut(256))
            .enumerate()
            .for_each(|(idx, (ad, aw))| {
                let (i, j) = (idx % gw, idx / gw);
                let x0 = (i * 8) as isize - 8;
                let y0 = (j * 8) as isize - 8;
                let (ox, oy) = offs.exact(idx);
                let mut tile = [0.0f32; 256];
                for v in 0..16isize {
                    let fy = (y0 + v) as f32 + oy;
                    for u in 0..16isize {
                        let fx = (x0 + u) as f32 + ox;
                        tile[v as usize * 16 + u as usize] = sample_subplane(alt, w, h, fx, fy);
                    }
                }
                let af = dct2_16(&tile);
                let rf = &self.ref_dct[idx * 256..(idx + 1) * 256];
                let var = self.tile_var[idx];
                // Compute tile-level normalized energy difference to gate large ghost/occlusion tiles cleanly.
                let mut diff_energy = 0.0f32;
                for k in 0..256 {
                    let d = af[k] - rf[k];
                    diff_energy += d * d;
                }
                let tile_q = diff_energy / (256.0 * var);
                let tile_gate = 1.0 / (1.0 + (tile_q / 24.0).powi(2));
                // Per-frequency Wiener temporal weight: low frequencies average strongly, moving high
                // frequencies reject cleanly.
                for k in 0..256 {
                    let d = af[k] - rf[k];
                    let d2 = d * d;
                    let c_var = WIENER_C * var;
                    let a_z = d2 / (d2 + c_var);
                    let wk = (1.0 - a_z) * tile_gate;
                    ad[k] += wk * af[k];
                    aw[k] += wk;
                }
            });
    }

    /// Synthesize the merged sub-plane via IDCT + 50%-overlap raised-cosine Hann windowing.
    /// Also applies mild residual frequency-domain spatial cleaning on high frequencies.
    pub fn finish(self, spatial_clean: bool) -> Vec<f32> {
        let (w, h, gw, gh) = (self.w, self.h, self.gw, self.gh);
        let ntiles = gw * gh;
        // 1. Inverse DCT every merged tile in parallel.
        let merged_tiles: Vec<[f32; 256]> = (0..ntiles)
            .into_par_iter()
            .map(|idx| {
                let rf = &self.ref_dct[idx * 256..(idx + 1) * 256];
                let ad = &self.acc_dct[idx * 256..(idx + 1) * 256];
                let aw = &self.acc_wt[idx * 256..(idx + 1) * 256];
                let var = self.tile_var[idx];
                let mut mf = [0.0f32; 256];
                for k in 0..256 {
                    let n_eff = 1.0 + aw[k];
                    let mut val = (rf[k] + ad[k]) / n_eff;
                    if spatial_clean && k > 0 {
                        let u = k & 15;
                        let v = k >> 4;
                        let f2 = (u * u + v * v) as f32;
                        // Gentle high-frequency spatial Wiener clean (Hasinoff Eq. 7) scaled by remaining noise.
                        if f2 >= 9.0 {
                            let rem_var = var / n_eff;
                            let k_sp = 0.65 * (f2 / (f2 + 32.0));
                            let v2 = val * val;
                            val *= v2 / (v2 + k_sp * rem_var + 1e-12);
                        }
                    }
                    mf[k] = val;
                }
                idct2_16(&mf)
            })
            .collect();

        // 2. Overlap-add by horizontal bands of 8 rows (band b is covered by tile rows b and b+1).
        let win = hann16();
        let mut out = vec![0.0f32; w * h];
        let band_px = 8 * w;
        out.par_chunks_mut(band_px).enumerate().for_each(|(b, band)| {
            let ybase = b * 8;
            for j in [b, b + 1] {
                if j >= gh {
                    continue;
                }
                let y0 = (j * 8) as isize - 8;
                for i in 0..gw {
                    let x0 = (i * 8) as isize - 8;
                    let tile = &merged_tiles[j * gw + i];
                    let xs = x0.max(0) as usize;
                    let xe = ((x0 + 16).min(w as isize)).max(0) as usize;
                    for yy in 0..8 {
                        let y = ybase + yy;
                        if y >= h {
                            break;
                        }
                        let ty = (y as isize - y0) as usize;
                        let wy = win[ty];
                        let trow = &tile[ty * 16..(ty + 1) * 16];
                        let dst = &mut band[yy * w..(yy + 1) * w];
                        for x in xs..xe {
                            let tx = (x as isize - x0) as usize;
                            dst[x] += win[tx] * wy * trow[tx];
                        }
                    }
                }
            }
        });
        out
    }
}

/// Splits a Bayer plane `(w, h)` into its 4 half-resolution CFA sub-planes `(w/2, h/2)`.
fn split_bayer(bayer: &[f32], w: usize, h: usize) -> [Vec<f32>; 4] {
    let (hw, hh) = (w / 2, h / 2);
    let mut subs = [vec![0.0f32; hw * hh], vec![0.0f32; hw * hh], vec![0.0f32; hw * hh], vec![0.0f32; hw * hh]];
    let [s0, s1, s2, s3] = &mut subs;
    s0.par_chunks_mut(hw)
        .zip(s1.par_chunks_mut(hw).zip(s2.par_chunks_mut(hw).zip(s3.par_chunks_mut(hw))))
        .enumerate()
        .for_each(|(y, (r0, (r1, (r2, r3))))| {
            let top = &bayer[(2 * y) * w..(2 * y + 1) * w];
            let bot = &bayer[(2 * y + 1) * w..(2 * y + 2) * w];
            for x in 0..hw {
                r0[x] = top[2 * x];
                r1[x] = top[2 * x + 1];
                r2[x] = bot[2 * x];
                r3[x] = bot[2 * x + 1];
            }
        });
    subs
}

/// Re-interleaves 4 half-resolution CFA sub-planes back into a Bayer plane `(w, h)`.
fn join_bayer(subs: [Vec<f32>; 4], w: usize, h: usize) -> Vec<f32> {
    let hw = w / 2;
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(2 * w).enumerate().for_each(|(y, rows)| {
        let (top, bot) = rows.split_at_mut(w);
        let o = y * hw;
        for x in 0..hw {
            top[2 * x] = subs[0][o + x];
            top[2 * x + 1] = subs[1][o + x];
            bot[2 * x] = subs[2][o + x];
            bot[2 * x + 1] = subs[3][o + x];
        }
    });
    out
}

/// Full-burst HDR+ Bayer merger (runs 4 independent phase-locked DCT Wiener mergers, one per CFA channel).
pub struct BayerMerger {
    w: usize,
    h: usize,
    subs: [SubplaneMerger; 4],
}

impl BayerMerger {
    pub fn new(refp: &[f32], w: usize, h: usize, noise: &NoiseModel) -> Self {
        let (hw, hh) = (w / 2, h / 2);
        let s = split_bayer(refp, w, h);
        let subs = [
            SubplaneMerger::new(&s[0], hw, hh, noise),
            SubplaneMerger::new(&s[1], hw, hh, noise),
            SubplaneMerger::new(&s[2], hw, hh, noise),
            SubplaneMerger::new(&s[3], hw, hh, noise),
        ];
        BayerMerger { w, h, subs }
    }

    pub fn add(&mut self, alt: &[f32], offs: &Offsets) {
        let s = split_bayer(alt, self.w, self.h);
        for k in 0..4 {
            self.subs[k].add(&s[k], offs);
        }
    }

    pub fn finish(self) -> Vec<f32> {
        let [m0, m1, m2, m3] = self.subs;
        let out = [m0.finish(true), m1.finish(true), m2.finish(true), m3.finish(true)];
        join_bayer(out, self.w, self.h)
    }
}

/// Spatial-domain / frequency-domain hybrid Merger wrapper kept API-compatible for YUV & tests.
/// When `scale == 2` and `tile == 32` (Bayer / full-res luma), uses 4-subplane HDR+ 2D-DCT merge
/// (preserving CFA phase and sub-pixel precision!). When `scale == 1` and `tile == 16` (half-res chroma),
/// uses single-plane HDR+ 2D-DCT merge directly.
pub struct Merger {
    inner: MergerInner,
}

enum MergerInner {
    Subplane(SubplaneMerger),
    Bayer { w: usize, h: usize, inner: Option<BayerMerger> },
}

impl Merger {
    /// `tile` is the tile edge in plane pixels (16 for half-res chroma, 32 for Bayer/full-res luma).
    pub fn new(w: usize, h: usize, tile: usize) -> Merger {
        assert!(tile == 16 || tile == 32);
        if tile == 32 {
            Merger { inner: MergerInner::Bayer { w, h, inner: None } }
        } else {
            // Lazy init on first add or finish so API matches previous Merger::new(w, h, tile).
            Merger {
                inner: MergerInner::Subplane(SubplaneMerger {
                    w,
                    h,
                    gw: 0,
                    gh: 0,
                    ref_dct: Vec::new(),
                    acc_dct: Vec::new(),
                    acc_wt: Vec::new(),
                    tile_var: Vec::new(),
                }),
            }
        }
    }

    pub fn add(&mut self, refp: &[f32], alt: &[f32], offs: &Offsets, scale: i32, noise: &NoiseModel) {
        match &mut self.inner {
            MergerInner::Bayer { w, h, inner } => {
                debug_assert_eq!(scale, 2);
                let bm = inner.get_or_insert_with(|| BayerMerger::new(refp, *w, *h, noise));
                bm.add(alt, offs);
            }
            MergerInner::Subplane(sm) => {
                debug_assert_eq!(scale, 1);
                if sm.gw == 0 {
                    *sm = SubplaneMerger::new(refp, sm.w, sm.h, noise);
                }
                sm.add(alt, offs);
            }
        }
    }

    pub fn finish(self, refp: &[f32]) -> Vec<f32> {
        match self.inner {
            MergerInner::Bayer { inner, .. } => match inner {
                Some(bm) => bm.finish(),
                None => refp.to_vec(),
            },
            MergerInner::Subplane(sm) => {
                if sm.gw == 0 {
                    refp.to_vec()
                } else {
                    sm.finish(false)
                }
            }
        }
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
    fn dct16_roundtrips_exactly() {
        let mut tile = [0.0f32; 256];
        for (i, v) in tile.iter_mut().enumerate() {
            *v = (i as f32 * 0.17).sin() * 0.5 + 0.5;
        }
        let back = idct2_16(&dct2_16(&tile));
        for (a, b) in tile.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
        }
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
        assert!(v1 < v0 / 4.0, "variance {} -> {}", v0, v1);
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
