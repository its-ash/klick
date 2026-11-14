//! Deterministic synthetic data for unit tests.

use crate::plane::Plane;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f32 {
        let (u1, u2) = (self.next_f32(), self.next_f32());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Smooth-but-detailed image in 0.1..0.9 (sum of sines plus blocks).
pub fn textured(w: usize, h: usize) -> Plane {
    let mut p = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32, y as f32);
            let v = 0.5
                + 0.18 * (fx * 0.21).sin() * (fy * 0.17).cos()
                + 0.12 * (fx * 0.07 + fy * 0.11).sin()
                + 0.08 * ((fx * 0.5).sin() * (fy * 0.43).sin())
                + if ((x / 9) + (y / 7)) % 3 == 0 { 0.08 } else { 0.0 };
            p.d[y * w + x] = v;
        }
    }
    p
}

/// Crop `w x h` of `src` starting at (x0, y0) and add Gaussian noise of std `sigma`.
pub fn shifted(src: &Plane, w: usize, h: usize, x0: usize, y0: usize, sigma: f32, seed: u64) -> Plane {
    let mut rng = Rng::new(seed);
    let mut out = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.d[y * w + x] = src.d[(y0 + y) * src.w + x0 + x] + sigma * rng.gauss();
        }
    }
    out
}

pub fn variance(v: &[f32]) -> f32 {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
}
