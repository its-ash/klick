//! Stage 3c: camera RGB to linear sRGB (DNG ForwardMatrix + ColorMatrix Bradford fallback)
//! and perceptual OkLab / OkLCh color-space transforms with exact gamut mapping.

use crate::demosaic::Rgb;
use crate::meta::Meta;
use rayon::prelude::*;

pub type Mat3 = [f32; 9];

pub const IDENTITY: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// XYZ (D50) to linear sRGB (Bradford adapted).
pub const XYZ_D50_TO_SRGB: Mat3 = [
    3.1338561, -1.6168667, -0.4906146, -0.9787684, 1.9161415, 0.0334540, 0.0719453, -0.2289914, 1.4052427,
];

/// Bradford cone response matrix (`XYZ -> LMS`) and its inverse.
const BRADFORD: Mat3 = [
    0.8951000, 0.2664000, -0.1614000, -0.7502000, 1.7135000, 0.0367000, 0.0389000, -0.0685000, 1.0296000,
];
const BRADFORD_INV: Mat3 = [
    0.9869929, -0.1470543, 0.1599627, 0.4323053, 0.5183603, 0.0492912, -0.0085287, 0.0400428, 0.9684867,
];
const D50_WHITE: [f32; 3] = [0.96422, 1.0, 0.82521];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    o
}

#[inline]
pub fn mul_vec(a: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

pub fn invert(m: &Mat3) -> Option<Mat3> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-9 {
        return None;
    }
    let s = 1.0 / det;
    Some([
        (e * i - f * h) * s,
        (c * h - b * i) * s,
        (b * f - c * e) * s,
        (f * g - d * i) * s,
        (a * i - c * g) * s,
        (c * d - a * f) * s,
        (d * h - e * g) * s,
        (b * g - a * h) * s,
        (a * e - b * d) * s,
    ])
}

fn is_set(m: &Mat3) -> bool {
    m.iter().any(|v| *v != 0.0) && m.iter().all(|v| v.is_finite())
}

fn lerp(a: &Mat3, b: &Mat3, t: f32) -> Mat3 {
    let mut o = [0.0; 9];
    for i in 0..9 {
        o[i] = a[i] * t + b[i] * (1.0 - t);
    }
    o
}

/// Bradford chromatic adaptation matrix from `src_xyz` white to `dst_xyz` white.
fn bradford_adapt(src_xyz: [f32; 3], dst_xyz: [f32; 3]) -> Mat3 {
    let s = mul_vec(&BRADFORD, src_xyz);
    let d = mul_vec(&BRADFORD, dst_xyz);
    let scale = [
        if s[0].abs() > 1e-6 { d[0] / s[0] } else { 1.0 },
        0.0,
        0.0,
        0.0,
        if s[1].abs() > 1e-6 { d[1] / s[1] } else { 1.0 },
        0.0,
        0.0,
        0.0,
        if s[2].abs() > 1e-6 { d[2] / s[2] } else { 1.0 },
    ];
    mul(&BRADFORD_INV, &mul(&scale, &BRADFORD))
}

/// McCamy's approximation of correlated colour temperature from CIE xy.
pub fn cct_from_xy(x: f32, y: f32) -> f32 {
    let n = (x - 0.3320) / (0.1858 - y).min(-1e-4);
    (449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33).clamp(1500.0, 15000.0)
}

/// Weight of matrix 1 for a colour temperature (interpolation in inverse temperature).
fn weight(cct: f32, cct1: f32, cct2: f32) -> f32 {
    if (cct1 - cct2).abs() < 1.0 {
        return 0.5;
    }
    let (inv, i1, i2) = (1.0 / cct, 1.0 / cct1, 1.0 / cct2);
    ((inv - i2) / (i1 - i2)).clamp(0.0, 1.0)
}

/// Illuminant colour temperature estimated from the as-shot neutral.
pub fn estimate_cct(meta: &Meta) -> f32 {
    let (c1, c2) = (meta.cct1.max(1000.0), meta.cct2.max(1000.0));
    if !is_set(&meta.cm1) && !is_set(&meta.cm2) {
        return 5000.0;
    }
    let cm1 = if is_set(&meta.cm1) { meta.cm1 } else { meta.cm2 };
    let cm2 = if is_set(&meta.cm2) { meta.cm2 } else { meta.cm1 };
    let mut cct = 0.5 * (c1 + c2);
    for _ in 0..12 {
        let cm = lerp(&cm1, &cm2, weight(cct, c1, c2));
        let Some(inv) = invert(&cm) else { break };
        let xyz = mul_vec(&inv, meta.neutral);
        let sum = xyz[0] + xyz[1] + xyz[2];
        if sum <= 0.0 {
            break;
        }
        let next = cct_from_xy(xyz[0] / sum, xyz[1] / sum).clamp(c1.min(c2) * 0.7, c1.max(c2) * 1.3);
        if (next - cct).abs() < 5.0 {
            cct = next;
            break;
        }
        cct = next;
    }
    cct
}

fn normalise_rows(mut m: Mat3) -> Mat3 {
    for r in 0..3 {
        let s = m[r * 3] + m[r * 3 + 1] + m[r * 3 + 2];
        if s.abs() > 1e-6 {
            for c in 0..3 {
                m[r * 3 + c] /= s;
            }
        }
    }
    m
}

/// Matrix taking white-balanced camera RGB to linear sRGB.
/// Uses interpolated DNG ForwardMatrix when present, and falls back to inverted ColorMatrix
/// with Bradford chromatic adaptation to D50 when a sensor omits ForwardMatrix (common on
/// ultrawide/telephoto lenses), preventing muddy identity-matrix RAW colors.
pub fn camera_to_srgb(meta: &Meta) -> Mat3 {
    let cct = estimate_cct(meta);
    let (c1, c2) = (meta.cct1.max(1000.0), meta.cct2.max(1000.0));
    let w = weight(cct, c1, c2);
    let fm = match (is_set(&meta.fm1), is_set(&meta.fm2)) {
        (true, true) => Some(lerp(&meta.fm1, &meta.fm2, w)),
        (true, false) => Some(meta.fm1),
        (false, true) => Some(meta.fm2),
        _ => None,
    };
    if let Some(fm) = fm {
        return normalise_rows(mul(&XYZ_D50_TO_SRGB, &fm));
    }
    // Fallback: derive XYZ_D50 from ColorMatrix1/2 (`XYZ_D50 -> CameraRGB`) + Bradford adaptation.
    let cm = match (is_set(&meta.cm1), is_set(&meta.cm2)) {
        (true, true) => Some(lerp(&meta.cm1, &meta.cm2, w)),
        (true, false) => Some(meta.cm1),
        (false, true) => Some(meta.cm2),
        _ => None,
    };
    if let Some(cm) = cm {
        if let Some(cam_to_xyz) = invert(&cm) {
            // Undo white balance back to camera neutral space via diagonal(neutral), map to XYZ, then Bradford adapt to D50.
            let diag = [meta.neutral[0], 0.0, 0.0, 0.0, meta.neutral[1], 0.0, 0.0, 0.0, meta.neutral[2]];
            let wb_to_xyz = mul(&cam_to_xyz, &diag);
            let src_white = mul_vec(&cam_to_xyz, meta.neutral);
            if src_white[1] > 1e-5 {
                let sw = [src_white[0] / src_white[1], 1.0, src_white[2] / src_white[1]];
                let adapt = bradford_adapt(sw, D50_WHITE);
                let xyz_d50 = mul(&adapt, &wb_to_xyz);
                return normalise_rows(mul(&XYZ_D50_TO_SRGB, &xyz_d50));
            }
            return normalise_rows(mul(&XYZ_D50_TO_SRGB, &wb_to_xyz));
        }
    }
    IDENTITY
}

pub fn apply_matrix(img: &mut Rgb, m: &Mat3) {
    let [r, g, b] = &mut img.c;
    r.par_iter_mut().zip(g.par_iter_mut().zip(b.par_iter_mut())).for_each(|(r, (g, b))| {
        let o = mul_vec(m, [*r, *g, *b]);
        *r = o[0];
        *g = o[1];
        *b = o[2];
    });
}

/// Linear sRGB (`0..1+`) to Björn Ottosson's perceptual OkLab `(L, a, b)`.
#[inline]
pub fn linear_srgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = (0.41222147 * r + 0.53633254 * g + 0.05144599 * b).max(0.0);
    let m = (0.21190350 * r + 0.68069955 * g + 0.10739696 * b).max(0.0);
    let s = (0.08830246 * r + 0.28171884 * g + 0.62997870 * b).max(0.0);
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    (
        0.21045426 * l_ + 0.79361779 * m_ - 0.00407205 * s_,
        1.97799850 * l_ - 2.42859220 * m_ + 0.45059371 * s_,
        0.02590404 * l_ + 0.78277177 * m_ - 0.80867577 * s_,
    )
}

/// OkLab `(L, a, b)` to linear sRGB (may be slightly outside `0..1` if out of gamut).
#[inline]
pub fn oklab_to_linear_srgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let l_ = l + 0.39633778 * a + 0.21580376 * b;
    let m_ = l - 0.10556135 * a - 0.06385417 * b;
    let s_ = l - 0.08948418 * a - 1.29148555 * b;
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;
    (
        4.07674166 * l3 - 3.30771159 * m3 + 0.23096993 * s3,
        -1.26843800 * l3 + 2.60975740 * m3 - 0.34131940 * s3,
        -0.00419609 * l3 - 0.70341861 * m3 + 1.70761470 * s3,
    )
}

/// Converts OkLab `(L, a, b)` back into in-gamut linear sRGB `[0, 1]^3` by preserving perceptual
/// lightness `L` and hue `atan2(b, a)` while smoothly compressing only excess chroma `C` onto
/// the sRGB gamut boundary (eliminates harsh channel-clip hue twists on bright flowers, skies, LEDs).
#[inline]
pub fn oklab_to_gamut_srgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let lc = l.clamp(0.0, 1.0);
    if lc <= 1e-5 {
        return (0.0, 0.0, 0.0);
    }
    if lc >= 1.0 - 1e-5 {
        return (1.0, 1.0, 1.0);
    }
    let (r, g, bl) = oklab_to_linear_srgb(lc, a, b);
    if r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && bl >= 0.0 && bl <= 1.0 {
        return (r, g, bl);
    }
    // 7-step binary search on chroma scale s in [0, 1] -> accuracy < 0.008 in chroma, zero hue twist.
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _ in 0..7 {
        let mid = 0.5 * (lo + hi);
        let (tr, tg, tb) = oklab_to_linear_srgb(lc, a * mid, b * mid);
        if tr >= 0.0 && tr <= 1.0 && tg >= 0.0 && tg <= 1.0 && tb >= 0.0 && tb <= 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (tr, tg, tb) = oklab_to_linear_srgb(lc, a * lo, b * lo);
    (tr.clamp(0.0, 1.0), tg.clamp(0.0, 1.0), tb.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_roundtrip() {
        let m = [0.9, 0.1, 0.0, 0.2, 0.7, 0.1, 0.0, 0.3, 0.8];
        let p = mul(&m, &invert(&m).unwrap());
        for i in 0..9 {
            assert!((p[i] - IDENTITY[i]).abs() < 1e-5);
        }
        assert!(invert(&[0.0; 9]).is_none());
    }

    #[test]
    fn cct_of_d65_and_a() {
        assert!((cct_from_xy(0.3127, 0.3290) - 6504.0).abs() < 150.0);
        assert!((cct_from_xy(0.4476, 0.4074) - 2856.0).abs() < 150.0);
    }

    #[test]
    fn fallback_cm_and_normalised_rows() {
        let mut meta = Meta::default();
        assert_eq!(camera_to_srgb(&meta), IDENTITY);
        // With only ColorMatrix set (no ForwardMatrix), camera_to_srgb derives a non-identity matrix keeping white neutral.
        meta.cm1 = [0.7, -0.1, 0.0, -0.2, 1.1, 0.05, 0.0, -0.1, 0.9];
        meta.neutral = [0.6, 1.0, 0.7];
        let m_cm = camera_to_srgb(&meta);
        assert_ne!(m_cm, IDENTITY);
        let w_cm = mul_vec(&m_cm, [1.0, 1.0, 1.0]);
        for c in w_cm {
            assert!((c - 1.0).abs() < 1e-4, "{:?}", w_cm);
        }
        let d50 = [0.9642, 1.0, 0.8249];
        let fm = [d50[0] / 3.0, d50[0] / 3.0, d50[0] / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, d50[2] / 3.0, d50[2] / 3.0, d50[2] / 3.0];
        meta.fm1 = fm;
        meta.fm2 = fm;
        let m = camera_to_srgb(&meta);
        let w = mul_vec(&m, [1.0, 1.0, 1.0]);
        for c in w {
            assert!((c - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn oklab_roundtrip_and_gamut_map() {
        for &(r, g, b) in &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.8, 0.35, 0.15), (0.1, 0.45, 0.9)] {
            let (l, a, bv) = linear_srgb_to_oklab(r, g, b);
            let (rr, gg, bb) = oklab_to_linear_srgb(l, a, bv);
            assert!((rr - r).abs() < 1e-4 && (gg - g).abs() < 1e-4 && (bb - b).abs() < 1e-4);
        }
        // Out-of-gamut high chroma compresses cleanly into [0,1] keeping hue sign.
        let (r, g, b) = oklab_to_gamut_srgb(0.65, 0.35, 0.20);
        assert!(r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && b >= 0.0 && b <= 1.0);
        assert!(r > g && r > b);
    }

    #[test]
    fn cct_estimate_follows_neutral() {
        let mut meta = Meta::default();
        meta.cm1 = [0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4];
        meta.cm2 = [0.6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9];
        meta.neutral = [0.7, 1.0, 0.6];
        let cct = estimate_cct(&meta);
        assert!(cct >= 2000.0 && cct <= 8500.0, "{}", cct);
    }
}
