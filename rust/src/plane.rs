//! Single-channel float image and shared filters: separable Gaussian, O(1) box blur,
//! Fast Guided Filter (He & Sun), Gaussian pyramid downsample, bilinear and Catmull-Rom sampling.

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Plane { w, h, d: vec![0.0; w * h] }
    }

    pub fn from_vec(w: usize, h: usize, d: Vec<f32>) -> Self {
        assert_eq!(d.len(), w * h, "plane size mismatch");
        Plane { w, h, d }
    }

    /// Edge-replicating read.
    #[inline]
    pub fn at(&self, x: isize, y: isize) -> f32 {
        let xc = x.clamp(0, self.w as isize - 1) as usize;
        let yc = y.clamp(0, self.h as isize - 1) as usize;
        self.d[yc * self.w + xc]
    }

    /// Bilinear sample at continuous pixel-centre coordinates.
    #[inline]
    pub fn sample(&self, fx: f32, fy: f32) -> f32 {
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (xi, yi) = (x0 as isize, y0 as isize);
        let a = self.at(xi, yi);
        let b = self.at(xi + 1, yi);
        let c = self.at(xi, yi + 1);
        let d = self.at(xi + 1, yi + 1);
        let top = a + (b - a) * tx;
        let bot = c + (d - c) * tx;
        top + (bot - top) * ty
    }

    /// Clamped 4-tap Catmull-Rom bicubic sample for sharp, ring-free sub-pixel resampling.
    #[inline]
    pub fn sample_catmull_rom(&self, fx: f32, fy: f32) -> f32 {
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        if tx.abs() < 1e-4 && ty.abs() < 1e-4 {
            return self.at(x0 as isize, y0 as isize);
        }
        let (xi, yi) = (x0 as isize, y0 as isize);
        let wx = catmull_weights(tx);
        let wy = catmull_weights(ty);
        let mut acc = 0.0f32;
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for j in 0..4isize {
            let mut row = 0.0f32;
            for i in 0..4isize {
                let v = self.at(xi + i - 1, yi + j - 1);
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
}

#[inline]
pub fn catmull_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as isize;
    let mut k: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable convolution with edge replication. The kernel must have odd length.
pub fn blur_separable(p: &Plane, k: &[f32]) -> Plane {
    let (w, h) = (p.w, p.h);
    let r = (k.len() / 2) as isize;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &p.d[y * w..(y + 1) * w];
        for (x, out) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xs = (x as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                acc += kv * src[xs];
            }
            *out = acc;
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (i, kv) in k.iter().enumerate() {
            let ys = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
            let src = &tmp[ys * w..(ys + 1) * w];
            for (o, s) in row.iter_mut().zip(src) {
                *o += kv * s;
            }
        }
    });
    Plane { w, h, d: out }
}

pub fn gaussian_blur(p: &Plane, sigma: f32) -> Plane {
    blur_separable(p, &gaussian_kernel(sigma))
}

/// O(N) separable mean filter with window `(2*r + 1) x (2*r + 1)` and edge replication.
pub fn box_blur(p: &Plane, r: usize) -> Plane {
    if r == 0 || p.w == 0 || p.h == 0 {
        return p.clone();
    }
    let (w, h) = (p.w, p.h);
    let ri = r as isize;
    let inv = 1.0 / (2 * r + 1) as f64;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &p.d[y * w..(y + 1) * w];
        let mut acc = 0.0f64;
        for dx in -ri..=ri {
            acc += src[dx.clamp(0, w as isize - 1) as usize] as f64;
        }
        row[0] = (acc * inv) as f32;
        for x in 1..w {
            let add_x = (x as isize + ri).min(w as isize - 1) as usize;
            let sub_x = (x as isize - ri - 1).max(0) as usize;
            acc += src[add_x] as f64 - src[sub_x] as f64;
            row[x] = (acc * inv) as f32;
        }
    });
    let mut out = vec![0.0f32; w * h];
    // Vertical pass: transpose-free column-batched prefix sum per row band is cache friendly via running row sum.
    let mut col_sum = vec![0.0f64; w];
    for dy in -ri..=ri {
        let sy = dy.clamp(0, h as isize - 1) as usize;
        let row = &tmp[sy * w..(sy + 1) * w];
        for (cs, &v) in col_sum.iter_mut().zip(row) {
            *cs += v as f64;
        }
    }
    for (o, &cs) in out[0..w].iter_mut().zip(&col_sum) {
        *o = (cs * inv) as f32;
    }
    for y in 1..h {
        let add_y = (y as isize + ri).min(h as isize - 1) as usize;
        let sub_y = (y as isize - ri - 1).max(0) as usize;
        let add_row = &tmp[add_y * w..(add_y + 1) * w];
        let sub_row = &tmp[sub_y * w..(sub_y + 1) * w];
        let dst = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            col_sum[x] += add_row[x] as f64 - sub_row[x] as f64;
            dst[x] = (col_sum[x] * inv) as f32;
        }
    }
    Plane { w, h, d: out }
}

/// Computes the linear coefficients `(mean_a, mean_b)` of He et al.'s Guided Image Filter
/// at the resolution of `guide` and `src`. Evaluating `mean_a * I + mean_b` yields an
/// edge-preserving filtered output with zero halos even across sharp high-contrast edges.
pub fn guided_coeffs(guide: &Plane, src: &Plane, radius: usize, eps: f32) -> (Plane, Plane) {
    assert_eq!((guide.w, guide.h), (src.w, src.h));
    let (w, h) = (guide.w, guide.h);
    let n = w * h;
    let mean_i = box_blur(guide, radius);
    let mean_p = box_blur(src, radius);
    let mut ii = vec![0.0f32; n];
    let mut ip = vec![0.0f32; n];
    ii.par_iter_mut()
        .zip(ip.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (vii, vip))| {
            let g = guide.d[idx];
            *vii = g * g;
            *vip = g * src.d[idx];
        });
    let corr_i = box_blur(&Plane { w, h, d: ii }, radius);
    let corr_ip = box_blur(&Plane { w, h, d: ip }, radius);
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (va, vb))| {
            let mi = mean_i.d[idx];
            let mp = mean_p.d[idx];
            let var_i = (corr_i.d[idx] - mi * mi).max(0.0);
            let cov_ip = corr_ip.d[idx] - mi * mp;
            let ak = cov_ip / (var_i + eps);
            *va = ak;
            *vb = mp - ak * mi;
        });
    let mean_a = box_blur(&Plane { w, h, d: a }, radius);
    let mean_b = box_blur(&Plane { w, h, d: b }, radius);
    (mean_a, mean_b)
}

/// Edge-preserving Guided Image Filter (He et al.).
pub fn guided_filter(guide: &Plane, src: &Plane, radius: usize, eps: f32) -> Plane {
    let (ma, mb) = guided_coeffs(guide, src, radius, eps);
    let d: Vec<f32> = (0..guide.d.len())
        .into_par_iter()
        .map(|i| ma.d[i] * guide.d[i] + mb.d[i])
        .collect();
    Plane { w: guide.w, h: guide.h, d }
}

/// Fast Guided Filter (He & Sun): computes `(mean_a, mean_b)` at `1/scale` resolution and
/// evaluates `a * guide + b` at full resolution so edges stay pixel-sharp at a fraction of the cost.
pub fn fast_guided_filter(guide: &Plane, src: &Plane, radius: usize, eps: f32, scale: usize) -> Plane {
    if scale <= 1 {
        return guided_filter(guide, src, radius, eps);
    }
    let gs = box_downsample(guide, scale);
    let ps = box_downsample(src, scale);
    let rs = (radius / scale).max(1);
    let (ma, mb) = guided_coeffs(&gs, &ps, rs, eps);
    let (w, h) = (guide.w, guide.h);
    let (sx, sy) = (ma.w as f32 / w as f32, ma.h as f32 / h as f32);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let g_row = &guide.d[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            *o = ma.sample(fx, fy) * g_row[x] + mb.sample(fx, fy);
        }
    });
    Plane { w, h, d: out }
}

/// Gaussian pyramid step: [1 4 6 4 1]/16 blur, then keep every second sample.
pub fn downsample2(p: &Plane) -> Plane {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let b = blur_separable(p, &K);
    let (nw, nh) = (p.w.div_ceil(2), p.h.div_ceil(2));
    let mut out = vec![0.0f32; nw * nh];
    out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = b.d[(y * 2).min(p.h - 1) * p.w + (x * 2).min(p.w - 1)];
        }
    });
    Plane { w: nw, h: nh, d: out }
}

pub fn resize_bilinear(p: &Plane, nw: usize, nh: usize) -> Plane {
    let sx = p.w as f32 / nw as f32;
    let sy = p.h as f32 / nh as f32;
    let mut out = vec![0.0f32; nw * nh];
    out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        for (x, o) in row.iter_mut().enumerate() {
            *o = p.sample((x as f32 + 0.5) * sx - 0.5, fy);
        }
    });
    Plane { w: nw, h: nh, d: out }
}

/// Average of k x k blocks (edge blocks are averaged over the pixels that exist).
pub fn box_downsample(p: &Plane, k: usize) -> Plane {
    if k <= 1 {
        return p.clone();
    }
    let (nw, nh) = (p.w.div_ceil(k), p.h.div_ceil(k));
    let mut out = vec![0.0f32; nw * nh];
    out.par_chunks_mut(nw).enumerate().for_each(|(by, row)| {
        for (bx, o) in row.iter_mut().enumerate() {
            let (x0, y0) = (bx * k, by * k);
            let (x1, y1) = ((x0 + k).min(p.w), (y0 + k).min(p.h));
            let mut s = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    s += p.d[y * p.w + x];
                }
            }
            *o = s / ((x1 - x0) * (y1 - y0)) as f32;
        }
    });
    Plane { w: nw, h: nh, d: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_keeps_flat_and_mean() {
        let p = Plane::from_vec(17, 13, vec![0.4; 17 * 13]);
        let b = gaussian_blur(&p, 2.0);
        assert!(b.d.iter().all(|v| (v - 0.4).abs() < 1e-5));
        let bx = box_blur(&p, 3);
        assert!(bx.d.iter().all(|v| (v - 0.4).abs() < 1e-5));
    }

    #[test]
    fn guided_filter_preserves_sharp_step_edge() {
        let (w, h) = (64, 64);
        let step = Plane::from_vec(w, h, (0..w * h).map(|i| if i % w < 32 { 0.1 } else { 0.9 }).collect());
        let out = guided_filter(&step, &step, 6, 0.005);
        assert!(out.d[32 * w + 31] < 0.2, "left {}", out.d[32 * w + 31]);
        assert!(out.d[32 * w + 32] > 0.8, "right {}", out.d[32 * w + 32]);
    }

    #[test]
    fn downsample_and_resize_dims() {
        let p = Plane::new(33, 21);
        let d = downsample2(&p);
        assert_eq!((d.w, d.h), (17, 11));
        let r = resize_bilinear(&d, 33, 21);
        assert_eq!((r.w, r.h), (33, 21));
        let b = box_downsample(&p, 4);
        assert_eq!((b.w, b.h), (9, 6));
    }

    #[test]
    fn bilinear_and_catmull_ramp_are_exact() {
        let mut p = Plane::new(12, 12);
        for y in 0..12 {
            for x in 0..12 {
                p.d[y * 12 + x] = x as f32;
            }
        }
        assert!((p.sample(2.5, 3.0) - 2.5).abs() < 1e-6);
        assert!((p.sample_catmull_rom(5.25, 5.5) - 5.25).abs() < 1e-5);
    }
}
