//! Stage 4: automatic exposure, adaptive dehaze/black-floor anchor, halo-free Guided-Filter
//! Mertens HDR local tone mapping, and AgX-inspired filmic highlight roll-off.

use crate::plane::{box_downsample, downsample2, guided_coeffs, resize_bilinear, Plane};

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
    pub p005: f32,
    pub p02: f32,
    pub p20: f32,
    pub p50: f32,
    pub p98: f32,
    pub p99: f32,
    pub p999: f32,
    /// Fraction of pixels above 0.95 (linear).
    pub bright_frac: f32,
}

pub fn luma_stats(l: &Plane) -> LumaStats {
    let mut v: Vec<f32> = l.d.iter().map(|x| x.max(0.0)).collect();
    if v.len() > 200_000 {
        let step = v.len() / 200_000 + 1;
        v = v.into_iter().step_by(step).collect();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len().max(1);
    let pct = |p: f32| v[((n as f32 - 1.0) * p) as usize];
    let log_mean = (v.iter().map(|x| (x + 1e-4).ln()).sum::<f32>() / n as f32).exp();
    let bright = v.iter().filter(|x| **x > 0.95).count() as f32 / n as f32;
    LumaStats {
        log_mean,
        p005: pct(0.005),
        p02: pct(0.02),
        p20: pct(0.2),
        p50: pct(0.5),
        p98: pct(0.98),
        p99: pct(0.99),
        p999: pct(0.999),
        bright_frac: bright,
    }
}

/// Key-aware automatic exposure gain balancing geometric mean, median, and highlight headroom.
/// Never darkens by more than `sqrt(0.7)` or lifts by more than `sqrt(2.0)`.
pub fn auto_exposure_gain(stats: &LumaStats, scene: u8) -> f32 {
    let target = match scene {
        2 => 0.105, // night stays atmospheric and rich, never grey
        1 => 0.175,
        _ => 0.175,
    };
    let key = (stats.log_mean.max(1e-4) * stats.p50.max(1e-4)).sqrt();
    let mut g = (target / key).clamp(0.7, 2.0);
    // Protect highlight headroom on non-HDR scenes; HDR fusion + filmic rolloff handles >2x brights.
    if g > 1.0 && scene != 1 && stats.p98 > 0.7 {
        let max_g = (2.1 / stats.p98).clamp(1.0, 2.0);
        g = g.min(max_g);
    }
    g.sqrt()
}

/// Linear level the 20th luma percentile is lifted towards by the bright synthetic exposure.
const SHADOW_TARGET: f32 = 0.066;

/// Linear black point anchoring true blacks cleanly while clearing lens-flare veiling glare
/// and daylight atmospheric haze WITHOUT crushing legitimate dark subjects (buildings, suits, hair).
pub fn black_point(stats_p005: f32) -> f32 {
    ((stats_p005 - 0.008).max(0.0) * 0.65).clamp(0.0, 0.014)
}

/// Scene-aware black floor with subtle haze clearing when a daylight photo has elevated shadows
/// and compressed brights (lens flare / misty haze).
pub fn scene_black_point(stats: &LumaStats, scene: u8) -> f32 {
    let base = black_point(stats.p005);
    if scene != 2 && stats.p02 > 0.045 && stats.p99 < 0.78 {
        ((stats.p005 - 0.006).max(0.0) * 0.80).clamp(base, 0.022)
    } else {
        base
    }
}

/// Global S-curve on display-referred values (0..1, endpoints fixed). `a` in 0..1 is the strength:
/// slope `1 - a` at black and white, `1 + a` at mid grey, always monotonic.
#[inline]
pub fn s_curve(y: f32, a: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    y - a * (std::f32::consts::TAU * y).sin() / std::f32::consts::TAU
}

/// Smooth asymmetric photographer tone curve on OkLab perceptual lightness `L` in `[0, 1]`.
/// Anchors deep shadows cleanly, gives midtones crisp sculptural separation, and keeps
/// highlight gradations airy (`matte_toe` gently lifts the darkest blacks for the Film look).
#[inline]
pub fn filmic_lightness_curve(l: f32, contrast: f32, shadow_depth: f32, matte_toe: f32) -> f32 {
    let y = l.clamp(0.0, 1.0);
    // Asymmetric S-curve: slightly richer shadow anchoring + smooth upper shoulder.
    let mut o = s_curve(y, contrast);
    // Subtle shadow depth pull concentrated in L in [0.05..0.38] (leaves 0 and >0.5 untouched).
    let sh = (1.0 - y).powi(3) * y * 6.75;
    o = (o - shadow_depth * sh).clamp(0.0, 1.0);
    // Optional gentle matte toe (Film look): lifts pure 0 softly to `matte_toe`.
    if matte_toe > 0.0 {
        o = matte_toe + o * (1.0 - matte_toe);
    }
    o.clamp(0.0, 1.0)
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
        1 => 1.25,
        2 => 0.80,
        _ => 1.0,
    };
    let bright_ev = ((SHADOW_TARGET / stats.p20.max(1e-3)).log2() * lift).clamp(0.0, 2.6);
    let dark_ev = if stats.p99 > 0.85 || stats.bright_frac > 0.01 {
        -((stats.p99.max(0.85) / 0.78).log2() + 0.75).clamp(0.5, 3.0)
    } else {
        0.0
    };
    let mut strength = ((range_stops - 5.0) / 4.0).clamp(0.0, 1.0);
    if scene == 1 {
        strength = strength.max(0.65);
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

/// Fuse exposures of linear luma using Mertens Laplacian pyramid blending.
/// Returns display-referred (gamma encoded) luma.
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
                .map(|(v, c)| (c + 0.015) * (-(v - 0.5) * (v - 0.5) / (2.0 * 0.22 * 0.22)).exp() + 1e-6)
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

/// Edge-aware Fast Guided-Filter HDR gain map (`gain = exp(a * ln(luma) + b)`).
/// Because `(a, b)` are smooth low-res linear coefficients applied directly to full-res log-luma,
/// local tone mapping tracks exact pixel-sharp object silhouettes (buildings, trees, hair against
/// bright skies) with ZERO HDR halos.
#[derive(Clone, Debug)]
pub struct GuidedGainMap {
    pub a: Plane,
    pub b: Plane,
    pub identity: bool,
}

impl GuidedGainMap {
    /// Evaluate the exact full-res linear gain multiplier at thumbnail-normalized coordinates `(fx, fy)`
    /// given the full-res linear pixel luminance `lum`.
    #[inline]
    pub fn eval(&self, fx: f32, fy: f32, lum: f32) -> f32 {
        if self.identity {
            return 1.0;
        }
        let log_l = (lum.max(0.0) + 1e-4).ln();
        let lg = self.a.sample(fx, fy) * log_l + self.b.sample(fx, fy);
        lg.exp().clamp(0.25, 6.5)
    }
}

/// Build an edge-aware `GuidedGainMap` from a linear luma thumbnail and fusion plan.
pub fn guided_gain_map(thumb: &Plane, plan: &FusionPlan) -> GuidedGainMap {
    if plan.strength <= 0.0 || plan.evs.len() < 2 {
        return GuidedGainMap { a: Plane::new(1, 1), b: Plane::new(1, 1), identity: true };
    }
    let fused = mertens_luma(thumb, &plan.evs);
    let n = thumb.w * thumb.h;
    let mut log_l = vec![0.0f32; n];
    let mut log_g = vec![0.0f32; n];
    for i in 0..n {
        let l = thumb.d[i].max(1e-4);
        log_l[i] = (l + 1e-4).ln();
        let target = srgb_decode(fused.d[i]);
        let full = (target / l).clamp(0.25, 6.5);
        log_g[i] = full.ln() * plan.strength;
    }
    let guide = Plane { w: thumb.w, h: thumb.h, d: log_l };
    let src = Plane { w: thumb.w, h: thumb.h, d: log_g };
    // Radius ~2% of thumbnail short side, eps in log-luma variance units -> razor-sharp halo-free edge lock.
    let radius = (thumb.w.min(thumb.h) / 48).clamp(4, 16);
    let (a, b) = guided_coeffs(&guide, &src, radius, 0.08);
    GuidedGainMap { a, b, identity: false }
}

/// Low-resolution gain map evaluated on the thumbnail grid (kept for tests and noise-map preview).
pub fn gain_map(thumb: &Plane, plan: &FusionPlan) -> Plane {
    let gg = guided_gain_map(thumb, plan);
    if gg.identity {
        return Plane { w: thumb.w, h: thumb.h, d: vec![1.0; thumb.w * thumb.h] };
    }
    let d: Vec<f32> = (0..thumb.w * thumb.h)
        .map(|i| {
            let (x, y) = (i % thumb.w, i / thumb.w);
            gg.eval(x as f32, y as f32, thumb.d[i])
        })
        .collect();
    Plane { w: thumb.w, h: thumb.h, d }
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
    const KNEE: f32 = 0.78;
    if m <= KNEE {
        1.0
    } else {
        (KNEE + (1.0 - KNEE) * ((m - KNEE) / (1.0 - KNEE)).tanh()) / m
    }
}

/// AgX-inspired filmic highlight compression and chromatic roll-off on linear sRGB.
/// Below `0.74` passes colors faithfully without twisting midtones or skin; above `0.74`,
/// smoothly rolls luminance onto a C² film shoulder and gracefully attenuates extreme specular
/// saturation toward luminous white (no neon clipping on sunsets, tungsten lights, or bright skies).
#[inline]
pub fn filmic_highlight_rolloff(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r = r.max(0.0);
    let g = g.max(0.0);
    let b = b.max(0.0);
    let m = r.max(g).max(b);
    if m <= 0.74 {
        return (r, g, b);
    }
    // Smooth tanh film shoulder preserving local highlight texture far past 1.0 (up to 4-6x).
    const KNEE: f32 = 0.74;
    const SPAN: f32 = 0.26;
    let target_m = KNEE + SPAN * ((m - KNEE) / SPAN).tanh();
    let s = target_m / m;
    let (rs, gs, bs) = (r * s, g * s, b * s);
    // Smooth AgX-like highlight desaturation toward bright luminance white as exposure exceeds 0.85.
    let desat = if m > 0.85 {
        let t = ((m - 0.85) / (m + 1.15)).clamp(0.0, 0.72);
        t * t * 0.95
    } else {
        0.0
    };
    if desat <= 1e-4 {
        return (rs, gs, bs);
    }
    let lum = luma(rs, gs, bs).max(target_m * 0.82);
    (
        (rs + desat * (lum - rs)).clamp(0.0, 1.0),
        (gs + desat * (lum - gs)).clamp(0.0, 1.0),
        (bs + desat * (lum - bs)).clamp(0.0, 1.0),
    )
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
    fn guided_fusion_pulls_shadows_up_and_highlights_down_without_halos() {
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
    fn rolloff_and_filmic_are_bounded_and_smooth() {
        assert_eq!(rolloff_scale(0.5), 1.0);
        for m in [0.9f32, 1.0, 2.0, 8.0] {
            let out = m * rolloff_scale(m);
            assert!(out <= 1.0 && out >= 0.78, "{} -> {}", m, out);
        }
        // Saturated bright sunset rolls smoothly to warm bright highlight without clipping harshly.
        let (r, g, b) = filmic_highlight_rolloff(3.0, 1.2, 0.4);
        assert!(r <= 1.0 && g <= 1.0 && b <= 1.0 && r > g && g > b);
    }
}
