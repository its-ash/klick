//! Dev tool: develop one Android DNG (as saved by Klick) with the engine, for tuning on real data.
//!
//!     cargo run --release --example develop -- in.dng out.jpg [ev_gain] [scene 0|1|2] [frames]
//!
//! `frames` only tells the finishing stages how many frames were merged (denoise strength); the
//! input is still the single DNG frame. Reads uncompressed DNGs like the ones `DngCreator` writes.

use klick_engine::merge::NoiseModel;
use klick_engine::meta::Meta;
use std::collections::HashMap;
use std::time::Instant;

struct Tiff<'a> {
    b: &'a [u8],
    le: bool,
}

impl<'a> Tiff<'a> {
    fn u16(&self, o: usize) -> u16 {
        let v = [self.b[o], self.b[o + 1]];
        if self.le { u16::from_le_bytes(v) } else { u16::from_be_bytes(v) }
    }
    fn u32(&self, o: usize) -> u32 {
        let v = [self.b[o], self.b[o + 1], self.b[o + 2], self.b[o + 3]];
        if self.le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) }
    }

    /// Tag -> values (as f64; bytes for BYTE/UNDEFINED) of the IFD at `off`.
    fn ifd(&self, off: usize) -> HashMap<u16, Vec<f64>> {
        let mut out = HashMap::new();
        let n = self.u16(off) as usize;
        for k in 0..n {
            let e = off + 2 + k * 12;
            let (tag, typ, cnt) = (self.u16(e), self.u16(e + 2), self.u32(e + 4) as usize);
            let size = match typ {
                1 | 2 | 6 | 7 => 1,
                3 | 8 => 2,
                4 | 9 | 11 => 4,
                5 | 10 | 12 => 8,
                _ => continue,
            };
            let at = if size * cnt <= 4 { e + 8 } else { self.u32(e + 8) as usize };
            if at + size * cnt > self.b.len() {
                continue;
            }
            let vals: Vec<f64> = (0..cnt)
                .map(|i| {
                    let p = at + i * size;
                    match typ {
                        1 | 2 | 7 => self.b[p] as f64,
                        6 => self.b[p] as i8 as f64,
                        3 => self.u16(p) as f64,
                        8 => self.u16(p) as i16 as f64,
                        4 => self.u32(p) as f64,
                        9 => self.u32(p) as i32 as f64,
                        5 => self.u32(p) as f64 / self.u32(p + 4).max(1) as f64,
                        10 => self.u32(p) as i32 as f64 / (self.u32(p + 4) as i32).max(1) as f64,
                        11 => f32::from_bits(self.u32(p)) as f64,
                        _ => {
                            let hi = self.u32(p) as u64;
                            let lo = self.u32(p + 4) as u64;
                            f64::from_bits(if self.le { (lo << 32) | hi } else { (hi << 32) | lo })
                        }
                    }
                })
                .collect();
            out.insert(tag, vals);
        }
        out
    }
}

fn mat(t: &HashMap<u16, Vec<f64>>, tag: u16) -> [f32; 9] {
    let mut m = [0.0f32; 9];
    if let Some(v) = t.get(&tag).filter(|v| v.len() >= 9) {
        for i in 0..9 {
            m[i] = v[i] as f32;
        }
    }
    m
}

fn cct(illuminant: f64) -> f32 {
    match illuminant as u32 {
        17 | 3 => 2856.0,
        18 => 4874.0,
        19 => 6774.0,
        20 => 5503.0,
        21 => 6504.0,
        22 => 7504.0,
        23 => 5003.0,
        24 => 3200.0,
        2 => 4150.0,
        10 => 6500.0,
        11 => 7500.0,
        12 => 6430.0,
        13 => 5000.0,
        14 => 4230.0,
        15 => 3450.0,
        _ => 5500.0,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: develop in.dng out.jpg [ev_gain] [scene] [frames]");
        std::process::exit(2);
    }
    let ev: f32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let scene: u8 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let frames: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    let bytes = std::fs::read(&a[1]).expect("read dng");
    let t = Tiff { b: &bytes, le: &bytes[0..2] == b"II" };
    let mut ifd = t.ifd(t.u32(4) as usize);
    // The raw image is IFD0 unless IFD0 is a preview with the raw in a SubIFD.
    if ifd.get(&254).map(|v| v[0] as u32) == Some(1) {
        if let Some(sub) = ifd.get(&330).and_then(|v| v.first()).copied() {
            let tags0 = ifd.clone();
            ifd = t.ifd(sub as usize);
            for k in [50721u16, 50722, 50778, 50779, 50728, 50964, 50965, 51041] {
                if let (false, Some(v)) = (ifd.contains_key(&k), tags0.get(&k)) {
                    ifd.insert(k, v.clone());
                }
            }
        }
    }
    let w = ifd[&256][0] as usize;
    let h = ifd[&257][0] as usize;
    assert_eq!(ifd[&258][0] as u32, 16, "only 16-bit uncompressed DNGs");
    assert_eq!(ifd.get(&259).map(|v| v[0] as u32).unwrap_or(1), 1, "only uncompressed DNGs");
    let mut raw = Vec::with_capacity(w * h * 2);
    for (off, len) in ifd[&273].iter().zip(&ifd[&279]) {
        let s = &bytes[*off as usize..(*off + *len) as usize];
        for px in s.chunks_exact(2) {
            let v = if t.le { u16::from_le_bytes([px[0], px[1]]) } else { u16::from_be_bytes([px[0], px[1]]) };
            raw.extend_from_slice(&v.to_le_bytes());
        }
    }
    raw.truncate(w * h * 2);

    let cfa = match ifd.get(&33422).map(|v| v.iter().map(|x| *x as u8).collect::<Vec<_>>()).as_deref() {
        Some([1, 0, 2, 1]) => 1,
        Some([1, 2, 0, 1]) => 2,
        Some([2, 1, 1, 0]) => 3,
        _ => 0,
    };
    let bl = ifd.get(&50714).cloned().unwrap_or(vec![0.0]);
    let black = [0, 1, 2, 3].map(|i| bl[i.min(bl.len() - 1)] as f32);
    let white = ifd.get(&50717).map(|v| v[0] as f32).unwrap_or(1023.0);
    let n = ifd.get(&50728).cloned().unwrap_or(vec![1.0, 1.0, 1.0]);
    let noise = ifd.get(&51041).map(|v| NoiseModel { s: v[0] as f32, o: v[1] as f32 }).unwrap_or(NoiseModel { s: 1e-3, o: 1e-5 });
    let meta = Meta {
        cfa,
        white,
        black,
        neutral: [n[0] as f32, n[1] as f32, n[2] as f32],
        fm1: mat(&ifd, 50964),
        fm2: mat(&ifd, 50965),
        cm1: mat(&ifd, 50721),
        cm2: mat(&ifd, 50722),
        cct1: cct(ifd.get(&50778).map(|v| v[0]).unwrap_or(17.0)),
        cct2: cct(ifd.get(&50779).map(|v| v[0]).unwrap_or(21.0)),
        noise,
        ev_gain: ev,
        scene,
        ..Meta::default()
    };
    eprintln!(
        "{}x{} cfa {} black {:?} white {} neutral {:?} noise s={:.2e} o={:.2e} cct {}/{}",
        w, h, cfa, black, white, meta.neutral, noise.s, noise.o, meta.cct1, meta.cct2
    );

    let t0 = Instant::now();
    let norm = klick_engine::raw::normalize_raw(&raw, w, h, w * 2, &meta).expect("normalize");
    let rgb = klick_engine::pipeline::develop_bayer(norm, w, h, &meta, None, klick_engine::pipeline::DEFAULT_MAX_PIXELS);
    let (ow, oh) = (rgb.w, rgb.h);
    let params = klick_engine::finish::FinishParams { scene, ev_gain: ev, noise, frames, display_referred: false };
    let (px, info) = klick_engine::finish::finish(rgb, &params);
    let jpeg = klick_engine::encode::encode_jpeg(&px, ow, oh, 92).expect("encode");
    std::fs::write(&a[2], jpeg).expect("write");
    eprintln!("done in {:.2}s: gain {:.2} fusion {:?}@{:.2} luma sigma {:.4}", t0.elapsed().as_secs_f32(), info.total_gain, info.fusion_evs, info.fusion_strength, info.luma_sigma);
}
