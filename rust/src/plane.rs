//! Single-channel float image and the basic filters every stage shares.

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
    fn bilinear_ramp_is_exact() {
        let mut p = Plane::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                p.d[y * 8 + x] = x as f32;
            }
        }
        assert!((p.sample(2.5, 3.0) - 2.5).abs() < 1e-6);
    }
}
