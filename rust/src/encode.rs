//! Stage 6: High-fidelity JPEG encoding with quality-adaptive chroma subsampling
//! (4:4:4 at Q >= 95 for zero colored-edge fringing, 4:2:2 at Q >= 90) and optimized Huffman tables.

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

pub fn encode_jpeg(rgb8: &[u8], w: usize, h: usize, quality: u8) -> Result<Vec<u8>, String> {
    if w == 0 || h == 0 || w > 65535 || h > 65535 {
        return Err(format!("bad image size {}x{}", w, h));
    }
    if rgb8.len() != w * h * 3 {
        return Err("pixel buffer does not match the size".into());
    }
    let q = quality.clamp(1, 100);
    let mut out = Vec::with_capacity(w * h / 2);
    let mut enc = Encoder::new(&mut out, q);
    enc.set_optimized_huffman_tables(true);
    let sampling = if q >= 95 {
        SamplingFactor::R_4_4_4
    } else if q >= 90 {
        SamplingFactor::R_4_2_2
    } else {
        SamplingFactor::R_4_2_0
    };
    enc.set_sampling_factor(sampling);
    enc.encode(rgb8, w as u16, h as u16, ColorType::Rgb).map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_decodes_to_same_size_and_colour() {
        let (w, h) = (48, 32);
        let rgb: Vec<u8> = (0..w * h).flat_map(|_| [200u8, 120, 40]).collect();
        let jpg = encode_jpeg(&rgb, w, h, 95).unwrap();
        assert_eq!(&jpg[..2], &[0xFF, 0xD8]);
        let mut dec = jpeg_decoder::Decoder::new(&jpg[..]);
        let px = dec.decode().unwrap();
        let info = dec.info().unwrap();
        assert_eq!((info.width as usize, info.height as usize), (w, h));
        assert!((px[0] as i32 - 200).abs() < 4 && (px[1] as i32 - 120).abs() < 4 && (px[2] as i32 - 40).abs() < 4);
    }

    #[test]
    fn rejects_bad_buffers() {
        assert!(encode_jpeg(&[0; 10], 4, 4, 90).is_err());
        assert!(encode_jpeg(&[], 0, 0, 90).is_err());
    }
}
