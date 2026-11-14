//! Stage 3c: camera RGB to linear sRGB. Follows the DNG recipe: pick the illuminant from the
//! as-shot neutral by iterating over the two colour matrices, interpolate the forward matrix in
//! inverse colour temperature, then go XYZ(D50) to sRGB.

use crate::demosaic::Rgb;
use crate::meta::Meta;
use rayon::prelude::*;

pub type Mat3 = [f32; 9];

pub const IDENTITY: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// XYZ (D50) to linear sRGB (Bradford adapted).
pub const XYZ_D50_TO_SRGB: Mat3 = [
    3.1338561, -1.6168667, -0.4906146, -0.9787684, 1.9161415, 0.0334540, 0.0719453, -0.2289914, 1.4052427,
];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    o
}

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

/// Matrix taking white-balanced camera RGB to linear sRGB. Rows are normalised so that a
/// white-balanced neutral stays neutral. Falls back to identity when the sensor reports no matrices.
pub fn camera_to_srgb(meta: &Meta) -> Mat3 {
    let fm = match (is_set(&meta.fm1), is_set(&meta.fm2)) {
        (true, true) => {
            let cct = estimate_cct(meta);
            let (c1, c2) = (meta.cct1.max(1000.0), meta.cct2.max(1000.0));
            Some(lerp(&meta.fm1, &meta.fm2, weight(cct, c1, c2)))
        }
        (true, false) => Some(meta.fm1),
        (false, true) => Some(meta.fm2),
        _ => None,
    };
    let Some(fm) = fm else { return IDENTITY };
    let mut m = mul(&XYZ_D50_TO_SRGB, &fm);
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

pub fn apply_matrix(img: &mut Rgb, m: &Mat3) {
    let [r, g, b] = &mut img.c;
    r.par_iter_mut().zip(g.par_iter_mut().zip(b.par_iter_mut())).for_each(|(r, (g, b))| {
        let o = mul_vec(m, [*r, *g, *b]);
        *r = o[0];
        *g = o[1];
        *b = o[2];
    });
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
    fn fallback_and_normalised_rows() {
        let mut meta = Meta::default();
        assert_eq!(camera_to_srgb(&meta), IDENTITY);
        // A forward matrix that maps (1,1,1) to D50 white must keep white neutral.
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
    fn cct_estimate_follows_neutral() {
        // A colour matrix pair where XYZ->camera is a plain scale; the estimate must stay in range.
        let mut meta = Meta::default();
        meta.cm1 = [0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4];
        meta.cm2 = [0.6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9];
        meta.neutral = [0.7, 1.0, 0.6];
        let cct = estimate_cct(&meta);
        assert!(cct >= 2000.0 && cct <= 8500.0, "{}", cct);
    }
}
