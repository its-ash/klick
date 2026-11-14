//! Capture metadata shared with Kotlin as one flat float array.
//!
//! Index layout (keep in sync with `EngineMeta.kt`):
//! 0 cfa (0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR), 1 white level, 2..6 black level (row-major 2x2),
//! 6..9 as-shot neutral (R,G,B), 9..18 forward matrix 1, 18..27 forward matrix 2,
//! 27..36 color matrix 1, 36..45 color matrix 2, 45 cct 1, 46 cct 2,
//! 47 noise S, 48 noise O, 49 iso, 50 exposure ns, 51 ev gain, 52 scene (0 normal, 1 hdr, 2 night),
//! 53 lsc width, 54 lsc height.

use crate::merge::NoiseModel;

pub const META_LEN: usize = 64;

#[derive(Clone, Debug)]
pub struct Meta {
    pub cfa: u8,
    pub white: f32,
    pub black: [f32; 4],
    pub neutral: [f32; 3],
    pub fm1: [f32; 9],
    pub fm2: [f32; 9],
    pub cm1: [f32; 9],
    pub cm2: [f32; 9],
    pub cct1: f32,
    pub cct2: f32,
    pub noise: NoiseModel,
    pub iso: f32,
    pub exposure_ns: f32,
    pub ev_gain: f32,
    pub scene: u8,
    pub lsc_w: usize,
    pub lsc_h: usize,
}

fn arr9(s: &[f32]) -> [f32; 9] {
    let mut a = [0.0; 9];
    a.copy_from_slice(&s[..9]);
    a
}

impl Default for Meta {
    fn default() -> Self {
        Meta {
            cfa: 0,
            white: 1023.0,
            black: [64.0; 4],
            neutral: [1.0; 3],
            fm1: [0.0; 9],
            fm2: [0.0; 9],
            cm1: [0.0; 9],
            cm2: [0.0; 9],
            cct1: 2856.0,
            cct2: 6504.0,
            noise: NoiseModel { s: 1e-3, o: 1e-5 },
            iso: 100.0,
            exposure_ns: 10_000_000.0,
            ev_gain: 1.0,
            scene: 0,
            lsc_w: 0,
            lsc_h: 0,
        }
    }
}

impl Meta {
    pub fn from_slice(s: &[f32]) -> Result<Meta, String> {
        if s.len() < META_LEN {
            return Err(format!("meta array too short: {}", s.len()));
        }
        let m = Meta {
            cfa: (s[0] as i32).clamp(0, 3) as u8,
            white: s[1],
            black: [s[2], s[3], s[4], s[5]],
            neutral: [s[6], s[7], s[8]],
            fm1: arr9(&s[9..]),
            fm2: arr9(&s[18..]),
            cm1: arr9(&s[27..]),
            cm2: arr9(&s[36..]),
            cct1: s[45],
            cct2: s[46],
            noise: NoiseModel { s: s[47].max(0.0), o: s[48].max(0.0) },
            iso: s[49],
            exposure_ns: s[50],
            ev_gain: s[51].max(1.0),
            scene: (s[52] as i32).clamp(0, 2) as u8,
            lsc_w: s[53].max(0.0) as usize,
            lsc_h: s[54].max(0.0) as usize,
        };
        if !(m.white > m.black.iter().cloned().fold(0.0, f32::max) + 1.0) {
            return Err("white level must exceed black level".into());
        }
        if m.neutral.iter().any(|v| !(*v > 0.0)) {
            return Err("as-shot neutral must be positive".into());
        }
        Ok(m)
    }
}

pub const R: usize = 0;
pub const G: usize = 1;
pub const B: usize = 2;

const PATTERNS: [[usize; 4]; 4] = [
    [R, G, G, B], // RGGB
    [G, R, B, G], // GRBG
    [G, B, R, G], // GBRG
    [B, G, G, R], // BGGR
];

#[inline]
pub fn cfa_color(cfa: u8, x: usize, y: usize) -> usize {
    PATTERNS[cfa as usize][(y & 1) * 2 + (x & 1)]
}

/// Lens shading map channel index (RGGB order: R, Gr, Gb, B) of a Bayer site.
#[inline]
pub fn lsc_channel(cfa: u8, x: usize, y: usize) -> usize {
    match cfa_color(cfa, x, y) {
        R => 0,
        B => 3,
        _ => {
            if cfa_color(cfa, x ^ 1, y) == R {
                1
            } else {
                2
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns() {
        assert_eq!(cfa_color(0, 0, 0), R);
        assert_eq!(cfa_color(0, 1, 1), B);
        assert_eq!(cfa_color(3, 0, 0), B);
        assert_eq!(cfa_color(1, 1, 0), R);
        assert_eq!(lsc_channel(0, 1, 0), 1);
        assert_eq!(lsc_channel(0, 0, 1), 2);
    }

    #[test]
    fn rejects_bad_meta() {
        assert!(Meta::from_slice(&[0.0; 10]).is_err());
        let mut s = vec![0.0f32; META_LEN];
        s[1] = 1023.0;
        s[2] = 64.0;
        s[3] = 64.0;
        s[4] = 64.0;
        s[5] = 64.0;
        s[6] = 0.5;
        s[7] = 1.0;
        s[8] = 0.6;
        assert!(Meta::from_slice(&s).is_ok());
        s[1] = 10.0;
        assert!(Meta::from_slice(&s).is_err());
    }
}
