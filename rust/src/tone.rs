//! Stage 4: automatic exposure and HDR. A luminance histogram decides how many synthetic
//! exposures are worth fusing; Mertens exposure fusion (contrast x well-exposedness weights,
//! Laplacian pyramid blend) runs on a small luma image, and the result is carried to the full-size
//! image as a smooth per-pixel gain map. That keeps memory flat for 12 to 50 MP sensors.

use crate::plane::{box_downsample, downsample2, resize_bilinear, Plane};

#[inline]
pub fn srgb_encode(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
pub fn srgb_decode(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[derive(Clone, Copy, Debug)]
pub struct LumaStats {
    pub log_mean: f32,
    pub p02: f32,
    pub p20: f32,
    pub p50: f32,
    pub p99: f32,
    pub p999: f32,
    /// Fraction of pixels above 0.95 (linear).
    pub bright_frac: f32,
}

pub fn luma_stats(l: &Plane) -> LumaStats {
    let mut v: Vec<f32> = l.d.iter().map(|x| x.max(0.0)).collect();
    // Subsample so sorting stays cheap.
    if v.len() > 200_000 {
        let step = v.len() / 200_000 + 1;
        v = v.into_iter().step_by(step).collect();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len().max(1);
    let pct = |p: f32| v[((n as f32 - 1.0) * p) as usize];
    let log_mean = (v.iter().map(|x| (x + 1e-4).ln()).sum::<f32>() / n as f32).exp();
    let bright = v.iter().filter(|x| **x > 0.95).count() as f32 / n as f32;
    LumaStats { log_mean, p02: pct(0.02), p20: pct(0.2), p50: pct(0.5), p99: pct(0.99), p999: pct(0.999), bright_frac: bright }
}

/// Gentle global gain towards a mid-grey key. Never darkens by more than 1/1.4 or lifts by more than 2x.
pub fn auto_exposure_gain(stats: &LumaStats, scene: u8) -> f32 {
    let target = match scene {
        2 => 0.10, // night shots should stay moody
        _ => 0.16,
    };
    let g = (target / stats.log_mean.max(1e-4)).clamp(0.7, 2.0);
    // Half the correction in log space: a hint, not a rewrite of the scene.
    g.sqrt()
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusionPlan {
    /// EV offsets of the synthetic exposures, always containing 0.0 when strength > 0.
    pub evs: Vec<f32>,
    pub strength: f32,
}

/// Choose synthetic exposures from the histogram. `scene`: 0 normal, 1 hdr, 2 night.
pub fn plan_fusion(stats: &LumaStats, scene: u8) -> FusionPlan {
    let range_stops = (stats.p999.max(1e-3) / stats.p02.max(2e-3)).log2();
    let lift = match scene {
        1 => 1.2,
        2 => 0.8,
        _ => 1.0,
    };
    let bright_ev = ((0.22 / stats.p20.max(1e-3)).log2() * lift).clamp(0.0, 3.0);
    let dark_ev = if stats.p99 > 0.85 || stats.bright_frac > 0.01 {
        -((stats.p99.max(0.85) / 0.8).log2() + 0.7).clamp(0.5, 3.0)
    } else {
        0.0
    };
    let mut strength = ((range_stops - 5.0) / 4.0).clamp(0.0, 1.0);
    if scene == 1 {
        strength = strength.max(0.6);
    }
    let mut evs = vec![];
    if dark_ev <= -0.5 {
        evs.push(dark_ev);
    }
    evs.push(0.0);
    if bright_ev >= 0.5 {
        evs.push(bright_ev);
    }
    if evs.len() < 2 || strength < 0.05 {
        return FusionPlan { evs: vec![0.0], strength: 0.0 };
    }
    FusionPlan { evs, strength }
}

fn laplacian_weight(img: &Plane) -> Plane {
    let mut out = Plane::new(img.w, img.h);
    for y in 0..img.h as isize {
        for x in 0..img.w as isize {
            let l = img.at(x - 1, y) + img.at(x + 1, y) + img.at(x, y - 1) + img.at(x, y + 1) - 4.0 * img.at(x, y);
            out.d[y as usize * img.w + x as usize] = l.abs();
        }
    }
    out
}

fn gaussian_pyramid(p: &Plane, levels: usize) -> Vec<Plane> {
    let mut v = vec![p.clone()];
    while v.len() < levels {
        let n = downsample2(v.last().unwrap());
        v.push(n);
    }
    v
}

fn laplacian_pyramid(p: &Plane, levels: usize) -> Vec<Plane> {
    let g = gaussian_pyramid(p, levels);
    let mut out = Vec::with_capacity(levels);
    for l in 0..levels - 1 {
        let up = resize_bilinear(&g[l + 1], g[l].w, g[l].h);
        let d: Vec<f32> = g[l].d.iter().zip(&up.d).map(|(a, b)| a - b).collect();
        out.push(Plane { w: g[l].w, h: g[l].h, d });
    }
    out.push(g[levels - 1].clone());
    out
}

/// Fuse exposures of linear luma. Returns display-referred (gamma encoded) luma.
pub fn mertens_luma(lin: &Plane, evs: &[f32]) -> Plane {
    let levels = ((lin.w.min(lin.h) as f32 / 8.0).log2().floor() as usize).clamp(2, 7);
    let imgs: Vec<Plane> = evs
        .iter()
        .map(|ev| {
            let k = 2f32.powf(*ev);
            Plane { w: lin.w, h: lin.h, d: lin.d.iter().map(|v| srgb_encode(v * k)).collect() }
        })
        .collect();
    let mut weights: Vec<Plane> = imgs
        .iter()
        .map(|im| {
            let c = laplacian_weight(im);
            let d: Vec<f32> = im
                .d
                .iter()
                .zip(&c.d)
                .map(|(v, c)| (c + 0.01) * (-(v - 0.5) * (v - 0.5) / (2.0 * 0.2 * 0.2)).exp() + 1e-6)
                .collect();
            Plane { w: im.w, h: im.h, d }
        })
        .collect();
    let n = lin.w * lin.h;
    for i in 0..n {
        let s: f32 = weights.iter().map(|w| w.d[i]).sum();
        weights.iter_mut().for_each(|w| w.d[i] /= s);
    }
    let mut fused: Option<Vec<Plane>> = None;
    for (im, w) in imgs.iter().zip(&weights) {
        let lp = laplacian_pyramid(im, levels);
        let wp = gaussian_pyramid(w, levels);
        match &mut fused {
            None => {
                fused = Some(
                    lp.iter()
                        .zip(&wp)
                        .map(|(l, w)| Plane { w: l.w, h: l.h, d: l.d.iter().zip(&w.d).map(|(a, b)| a * b).collect() })
                        .collect(),
                )
            }
            Some(f) => {
                for ((fl, l), w) in f.iter_mut().zip(&lp).zip(&wp) {
                    for ((o, a), b) in fl.d.iter_mut().zip(&l.d).zip(&w.d) {
                        *o += a * b;
                    }
                }
            }
        }
    }
    let f = fused.expect("at least one exposure");
    let mut out = f[levels - 1].clone();
    for l in (0..levels - 1).rev() {
        let up = resize_bilinear(&out, f[l].w, f[l].h);
        out = Plane { w: f[l].w, h: f[l].h, d: f[l].d.iter().zip(&up.d).map(|(a, b)| a + b).collect() };
    }
    out
}

/// Low-resolution gain map (linear multiplier per pixel) from a linear luma thumbnail.
pub fn gain_map(thumb: &Plane, plan: &FusionPlan) -> Plane {
    let mut gm = Plane { w: thumb.w, h: thumb.h, d: vec![1.0; thumb.w * thumb.h] };
    if plan.strength <= 0.0 || plan.evs.len() < 2 {
        return gm;
    }
    let fused = mertens_luma(thumb, &plan.evs);
    for i in 0..gm.d.len() {
        let l = thumb.d[i].max(1e-4);
        let target = srgb_decode(fused.d[i]);
        let full = (target / l).clamp(0.25, 8.0);
        // Interpolate in log space with the plan strength.
        gm.d[i] = (full.ln() * plan.strength).exp();
    }
    // Smooth so the gain varies slowly and cannot create halos at full resolution.
    crate::plane::gaussian_blur(&gm, 1.5)
}

/// Thumbnail of the luma of a planar image, long side about `target`.
pub fn luma_thumbnail(r: &[f32], g: &[f32], b: &[f32], w: usize, h: usize, target: usize) -> Plane {
    let full = Plane { w, h, d: (0..w * h).map(|i| luma(r[i], g[i], b[i])).collect() };
    let k = (w.max(h) / target).max(1);
    box_downsample(&full, k)
}

/// Soft highlight roll-off that keeps hue: scales the pixel when its brightest channel passes `knee`.
#[inline]
pub fn rolloff_scale(m: f32) -> f32 {
    const KNEE: f32 = 0.8;
    if m <= KNEE {
        1.0
    } else {
        (KNEE + (1.0 - KNEE) * (1.0 - (-(m - KNEE) / (1.0 - KNEE)).exp())) / m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::textured;

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            assert!((srgb_decode(srgb_encode(x)) - x).abs() < 1e-4);
        }
    }

    fn ramp_stats(lo: f32, hi: f32) -> LumaStats {
        let n = 10_000;
        let p = Plane::from_vec(100, 100, (0..n).map(|i| lo + (hi - lo) * i as f32 / n as f32).collect());
        luma_stats(&p)
    }

    #[test]
    fn flat_scenes_skip_fusion() {
        let s = ramp_stats(0.1, 0.4);
        let plan = plan_fusion(&s, 0);
        assert_eq!(plan.strength, 0.0);
        assert_eq!(plan.evs, vec![0.0]);
    }

    #[test]
    fn high_contrast_scenes_get_dark_and_bright_exposures() {
        // Deep shadows and blown highlights in the same frame.
        let mut d = vec![0.004f32; 6000];
        d.extend(vec![1.2f32; 4000]);
        let p = Plane::from_vec(100, 100, d);
        let plan = plan_fusion(&luma_stats(&p), 1);
        assert!(plan.strength > 0.5);
        assert!(plan.evs.iter().any(|e| *e < -0.4) && plan.evs.iter().any(|e| *e > 0.4));
        assert!(plan.evs.contains(&0.0));
    }

    #[test]
    fn auto_exposure_is_bounded() {
        let dark = ramp_stats(0.0005, 0.002);
        let g = auto_exposure_gain(&dark, 0);
        assert!(g > 1.2 && g <= 2f32.sqrt() + 1e-4);
        let bright = ramp_stats(0.8, 1.0);
        assert!(auto_exposure_gain(&bright, 0) >= 0.7f32.sqrt() - 1e-4);
    }

    #[test]
    fn fusion_pulls_shadows_up_and_highlights_down() {
        // Left half dark, right half bright, with texture so contrast weights are meaningful.
        let (w, h) = (128, 64);
        let t = textured(w, h);
        let lin = Plane::from_vec(w, h, (0..w * h).map(|i| if i % w < w / 2 { 0.01 * t.d[i] } else { 1.4 * t.d[i] }).collect());
        let plan = FusionPlan { evs: vec![-2.0, 0.0, 3.0], strength: 1.0 };
        let gm = gain_map(&lin, &plan);
        let left = gm.d[32 * w + 20];
        let right = gm.d[32 * w + 110];
        assert!(left > 1.5, "shadow gain {}", left);
        assert!(right < 1.0, "highlight gain {}", right);
        assert!(gm.d.iter().all(|v| v.is_finite() && *v > 0.2 && *v < 9.0));
    }

    #[test]
    fn zero_strength_gain_map_is_identity() {
        let t = textured(32, 32);
        let gm = gain_map(&t, &FusionPlan { evs: vec![0.0], strength: 0.0 });
        assert!(gm.d.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn rolloff_is_hue_preserving_and_bounded() {
        assert_eq!(rolloff_scale(0.5), 1.0);
        for m in [0.9f32, 1.0, 2.0, 8.0] {
            let out = m * rolloff_scale(m);
            assert!(out <= 1.0 && out >= 0.8, "{} -> {}", m, out);
        }
    }
}
