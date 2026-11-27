//! Stages 4 to 5 glued together: Pro computational photography finishing pipeline.
//!
//! Linear sRGB -> Guided-Filter Halo-Free HDR Local Tone Mapping -> AgX-inspired Filmic Shoulder ->
//! Perceptual OkLab Lightness/Chroma Split -> Inter-Scale BiShrink Wavelet Luma & Guided-Upsampled
//! Chroma Denoising -> Halo-Free Guided Clarity -> Per-Pixel Noise-Masked Dual-Scale Sharpening ->
//! Skin-Protected OkLCh Photographer Color Grading & Split Toning -> Exact OkLCh Gamut Mapping ->
//! Band-Free Dithered 8-bit sRGB.

use crate::color::{linear_srgb_to_oklab, oklab_to_gamut_srgb};
use crate::demosaic::Rgb;
use crate::denoise::{jbu_upsample_chroma, strengths, wavelet_denoise};
use crate::merge::NoiseModel;
use crate::meta::PhotoLook;
use crate::plane::{box_downsample, Plane};
use crate::sharpen::{guided_clarity, pro_sharpen};
use crate::tone::{
    auto_exposure_gain, filmic_highlight_rolloff, filmic_lightness_curve, guided_gain_map, luma_stats,
    luma_thumbnail, plan_fusion, rolloff_scale, scene_black_point, srgb_decode, srgb_encode, GuidedGainMap,
};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct FinishParams {
    pub scene: u8,
    /// Digital gain that restores the deliberate under-exposure of the burst.
    pub ev_gain: f32,
    pub noise: NoiseModel,
    pub frames: usize,
    /// True when the pixels already went through a camera ISP tone curve (the YUV path).
    pub display_referred: bool,
    /// Photographer look profile (default: `PhotoLook::Pro`).
    pub look: PhotoLook,
}

#[derive(Clone, Debug, Default)]
pub struct FinishInfo {
    pub total_gain: f32,
    pub fusion_evs: Vec<f32>,
    pub fusion_strength: f32,
    pub luma_sigma: f32,
}

/// Luma wavelet BiShrink thresholds per scale (finest first): preserves crisp detail & fine organic grain.
const LUMA_K: [f32; 5] = [1.05, 0.95, 0.78, 0.58, 0.38];
/// Chroma wavelet thresholds at half resolution in OkLab a/b (4 scales keeps sharp color edges zero-bleed).
const CHROMA_K: [f32; 4] = [2.6, 2.4, 2.0, 1.5];
/// Stronger 6-scale chroma thresholds used only for noisy Night scenes to remove broad low-ISO blotches.
const CHROMA_K_NIGHT: [f32; 6] = [2.8, 2.8, 2.6, 2.4, 2.0, 1.6];

/// Tuned editorial color/tone recipe for each `PhotoLook`.
struct GradeRecipe {
    contrast: f32,
    shadow_depth: f32,
    matte_toe: f32,
    clarity: f32,
    fine_sharp: f32,
    tex_sharp: f32,
    vibrance: f32,
    saturation: f32,
    sky_pop: f32,
    foliage_shift: f32,
    warmth_a: f32,
    warmth_b: f32,
    split_cool: f32,
    split_warm: f32,
    vignette: f32,
    grain: f32,
    mono: bool,
}

fn recipe(look: PhotoLook, scene: u8, display_referred: bool) -> GradeRecipe {
    let base = match look {
        PhotoLook::Pro => GradeRecipe {
            contrast: match scene {
                2 => 0.21,
                1 => 0.24,
                _ => 0.27,
            },
            shadow_depth: match scene {
                1 => 0.004,
                2 => 0.008,
                _ => 0.010,
            },
            matte_toe: 0.0,
            clarity: match scene {
                2 => 0.12,
                1 => 0.22,
                _ => 0.20,
            },
            fine_sharp: match scene {
                2 => 0.55,
                _ => 0.80,
            },
            tex_sharp: match scene {
                2 => 0.12,
                _ => 0.24,
            },
            vibrance: 0.22,
            saturation: 1.04,
            sky_pop: 1.15,
            foliage_shift: 1.0,
            warmth_a: 0.0,
            warmth_b: 0.0,
            split_cool: 0.005,
            split_warm: 0.0055,
            vignette: 0.04,
            grain: 0.0,
            mono: false,
        },
        PhotoLook::Natural => GradeRecipe {
            contrast: match scene {
                2 => 0.13,
                _ => 0.16,
            },
            shadow_depth: 0.004,
            matte_toe: 0.0,
            clarity: 0.06,
            fine_sharp: 0.58,
            tex_sharp: 0.08,
            vibrance: 0.04,
            saturation: 1.0,
            sky_pop: 0.0,
            foliage_shift: 0.2,
            warmth_a: 0.0,
            warmth_b: 0.0,
            split_cool: 0.0,
            split_warm: 0.0,
            vignette: 0.0,
            grain: 0.0,
            mono: false,
        },
        PhotoLook::Vivid => GradeRecipe {
            contrast: match scene {
                2 => 0.26,
                _ => 0.34,
            },
            shadow_depth: 0.018,
            matte_toe: 0.0,
            clarity: 0.28,
            fine_sharp: 0.90,
            tex_sharp: 0.30,
            vibrance: 0.38,
            saturation: 1.18,
            sky_pop: 1.75,
            foliage_shift: 1.1,
            warmth_a: 0.001,
            warmth_b: 0.003,
            split_cool: 0.004,
            split_warm: 0.006,
            vignette: 0.035,
            grain: 0.0,
            mono: false,
        },
        PhotoLook::Film => GradeRecipe {
            contrast: 0.22,
            shadow_depth: 0.0,
            matte_toe: 0.048,
            clarity: 0.10,
            fine_sharp: 0.44,
            tex_sharp: 0.10,
            vibrance: 0.08,
            saturation: 0.93,
            sky_pop: 0.65,
            foliage_shift: 1.45,
            warmth_a: 0.0035,
            warmth_b: 0.011,
            split_cool: 0.016,
            split_warm: 0.022,
            vignette: 0.085,
            grain: 0.0085,
            mono: false,
        },
        PhotoLook::Mono => GradeRecipe {
            contrast: 0.38,
            shadow_depth: 0.028,
            matte_toe: 0.012,
            clarity: 0.28,
            fine_sharp: 0.85,
            tex_sharp: 0.30,
            vibrance: 0.0,
            saturation: 0.0,
            sky_pop: 0.0,
            foliage_shift: 0.0,
            warmth_a: 0.0,
            warmth_b: 0.0,
            split_cool: 0.0,
            split_warm: 0.0,
            vignette: 0.075,
            grain: 0.007,
            mono: true,
        },
    };
    if !display_referred {
        return base;
    }
    // YUV frames already passed through a camera ISP tone curve: keep contrast/sharpening gentle
    // so they still gain our editorial polish without ever looking over-processed.
    GradeRecipe {
        contrast: base.contrast * 0.32,
        shadow_depth: base.shadow_depth * 0.35,
        clarity: base.clarity * 0.55,
        fine_sharp: base.fine_sharp * 0.45,
        tex_sharp: base.tex_sharp * 0.45,
        vibrance: base.vibrance * 0.6,
        saturation: 1.0 + (base.saturation - 1.0) * 0.5,
        ..base
    }
}

/// OkLab-domain noise standard deviation on a 1/4-resolution grid. Follows the sensor noise model
/// through frame merge, exact guided local tone gain, black lift, filmic rolloff, and cube-root
/// OkLab lightness slope (`d(x^(1/3))/dx`), so denoising and sharpening adapt to every region.
#[allow(clippy::too_many_arguments)]
fn noise_map(
    rgb: &Rgb,
    gg: &GuidedGainMap,
    thumb_dims: (usize, usize),
    total_gain: f32,
    bp: f32,
    contrast: f32,
    noise: &NoiseModel,
    frames: usize,
) -> Plane {
    const B: usize = 4;
    let (w, h) = (rgb.w, rgb.h);
    let (bw, bh) = (w.div_ceil(B), h.div_ceil(B));
    let (sx, sy) = (thumb_dims.0 as f32 / w as f32, thumb_dims.1 as f32 / h as f32);
    let eff = 1.0 + 0.72 * (frames.max(1) as f32 - 1.0);
    let lift = 1.0 / (1.0 - bp);
    let [r, g, b] = &rgb.c;
    let mut d = vec![0.0f32; bw * bh];
    d.par_chunks_mut(bw).enumerate().for_each(|(by, row)| {
        for (bx, o) in row.iter_mut().enumerate() {
            let (x0, y0) = (bx * B, by * B);
            let (x1, y1) = ((x0 + B).min(w), (y0 + B).min(h));
            let mut sum = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * w + x;
                    sum += crate::tone::luma(r[i], g[i], b[i]);
                }
            }
            let lum = (sum / ((x1 - x0) * (y1 - y0)) as f32).max(0.0);
            let (cx, cy) = ((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5);
            let k = total_gain * gg.eval(cx * sx - 0.5, cy * sy - 0.5, lum * total_gain);
            let sd_lin = (noise.var(lum) / eff).sqrt() * k * lift;
            let out_lin = ((lum * k - bp).max(0.0) * lift).min(1.0);
            // Derivative of OkLab L ≈ Y^(1/3) with smooth shadow floor (matches sRGB-like visibility).
            let l_val = out_lin.cbrt();
            let slope = 1.0 / (3.0 * out_lin.max(0.004).powf(2.0 / 3.0));
            let curve = 1.0 - contrast * (std::f32::consts::TAU * l_val).cos();
            *o = (sd_lin * slope * curve).clamp(0.0015, 0.14);
        }
    });
    Plane { w: bw, h: bh, d }
}

/// Smooth angular distance wrapped to `[-pi, pi]`.
#[inline]
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// Fast deterministic triangular dither in `[-0.45, +0.45]` LSB to eliminate 8-bit sky banding.
#[inline]
fn tri_dither(x: usize, y: usize) -> f32 {
    let h1 = ((x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263))
        .wrapping_mul(1_274_126_177);
    let h2 = h1 ^ (h1 >> 13);
    let u1 = (h1 & 0xFFFF) as f32 / 65535.0;
    let u2 = (h2 & 0xFFFF) as f32 / 65535.0;
    (u1 + u2 - 1.0) * 0.45
}

pub fn finish(rgb: Rgb, p: &FinishParams) -> (Vec<u8>, FinishInfo) {
    let (w, h) = (rgb.w, rgb.h);
    let [r, g, b] = &rgb.c;
    let grade = recipe(p.look, p.scene, p.display_referred);

    // Stage 4: exposure, adaptive haze/black anchor, and halo-free Guided HDR gain map.
    let mut thumb = luma_thumbnail(r, g, b, w, h, 768);
    let pre = if p.display_referred { 1.0 } else { p.ev_gain.max(1.0) };
    thumb.d.iter_mut().for_each(|v| *v *= pre);
    let ae = if p.display_referred { 1.0 } else { auto_exposure_gain(&luma_stats(&thumb), p.scene) };
    thumb.d.iter_mut().for_each(|v| *v *= ae);
    let total_gain = pre * ae;
    let stats = luma_stats(&thumb);
    let bp = if p.display_referred { 0.0 } else { scene_black_point(&stats, p.scene) };

    let mut plan = plan_fusion(&stats, p.scene);
    if p.display_referred {
        plan.strength *= 0.45;
    }
    let gg = guided_gain_map(&thumb, &plan);
    let thumb_dims = (thumb.w, thumb.h);
    let (sx, sy) = (thumb_dims.0 as f32 / w as f32, thumb_dims.1 as f32 / h as f32);
    let mut sd = noise_map(&rgb, &gg, thumb_dims, total_gain, bp, grade.contrast, &p.noise, p.frames);
    if p.scene == 2 {
        sd.d.iter_mut().for_each(|v| *v *= 1.18);
    }

    // Apply exact edge-locked local HDR gain, AgX filmic shoulder, and move into perceptual OkLab.
    let mut lp = Plane::new(w, h);
    let mut ap = Plane::new(w, h);
    let mut bp_ch = Plane::new(w, h);
    let lift = 1.0 / (1.0 - bp);
    lp.d.par_chunks_mut(w)
        .zip(ap.d.par_chunks_mut(w).zip(bp_ch.d.par_chunks_mut(w)))
        .enumerate()
        .for_each(|(row, (lr, (ar, br)))| {
            let fy = (row as f32 + 0.5) * sy - 0.5;
            for x in 0..w {
                let i = row * w + x;
                let lum_pre = crate::tone::luma(r[i], g[i], b[i]) * total_gain;
                let k = total_gain * gg.eval((x as f32 + 0.5) * sx - 0.5, fy, lum_pre);
                let rv = ((r[i] * k) - bp).max(0.0) * lift;
                let gv = ((g[i] * k) - bp).max(0.0) * lift;
                let bv = ((b[i] * k) - bp).max(0.0) * lift;
                let (rf, gf, bf) = filmic_highlight_rolloff(rv, gv, bv);
                let (l0, a0, b0) = linear_srgb_to_oklab(rf, gf, bf);
                // Apply the filmic S-curve in display-referred sRGB lightness (centered at true 18% middle grey)
                // and convert back to OkLab L so shadows keep rich detail instead of being over-crushed by cube-root L.
                let l1 = if grade.contrast > 0.0 || grade.shadow_depth > 0.0 || grade.matte_toe > 0.0 {
                    let yd = srgb_encode((l0 * l0 * l0).clamp(0.0, 1.0));
                    let yd1 = filmic_lightness_curve(yd, grade.contrast, grade.shadow_depth, grade.matte_toe);
                    srgb_decode(yd1).cbrt()
                } else {
                    l0
                };
                // Scale chroma smoothly with lightness change so deepened shadows never turn muddy/radioactive.
                let c_scale = if l0 > 1e-4 {
                    let ratio = (l1 / l0).clamp(0.45, 1.35);
                    let shadow_clean = (l1 / 0.16).clamp(0.65, 1.0);
                    ratio.powf(0.58) * shadow_clean
                } else {
                    0.0
                };
                lr[x] = l1;
                ar[x] = a0 * c_scale;
                br[x] = b0 * c_scale;
            }
        });
    drop(rgb);

    // Stage 5a: Inter-scale BiShrink wavelet denoising in OkLab + Joint (L,a,b) Bilateral chroma upsampling.
    let (luma_sigma, _) = strengths(&p.noise, p.frames, total_gain, p.scene);
    let mut l_clean = wavelet_denoise(&lp, &sd, &LUMA_K);
    drop(lp);
    l_clean.d.par_iter_mut().for_each(|v| *v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });

    // Chroma in OkLab has ~0.55x the numeric amplitude of L, and 2x2 downsampling halves noise.
    let half_sd = Plane { w: sd.w, h: sd.h, d: sd.d.iter().map(|v| v * 0.32).collect() };
    let ck: &[f32] = if p.scene == 2 { &CHROMA_K_NIGHT } else { &CHROMA_K };
    let ah = wavelet_denoise(&box_downsample(&ap, 2), &half_sd, ck);
    let bh = wavelet_denoise(&box_downsample(&bp_ch, 2), &half_sd, ck);
    let (a_clean, b_clean) = jbu_upsample_chroma(&ah, &bh, &l_clean, &ap, &bp_ch, &sd);
    drop(ap);
    drop(bp_ch);

    // Stage 5b: Halo-free Guided Clarity + per-pixel noise-masked dual-scale sharpening.
    guided_clarity(&mut l_clean, &sd, grade.clarity);
    pro_sharpen(&mut l_clean, &sd, grade.fine_sharp, grade.tex_sharp, 0.018);

    // Stage 5c: Photographer OkLCh Color Grading, Skin Protection, Selective Harmony, Split-Toning,
    // Subtle Prime-Lens Vignette, Exact Perceptual Gamut Mapping, and Band-Free 8-bit sRGB output.
    let inv_w = 1.0 / w.max(1) as f32;
    let inv_h = 1.0 / h.max(1) as f32;
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(row, px)| {
        let ny = (row as f32 + 0.5) * inv_h - 0.5;
        let ny2 = ny * ny;
        for x in 0..w {
            let i = row * w + x;
            let mut lv = l_clean.d[i].clamp(0.0, 1.0);
            let mut av = a_clean.d[i];
            let mut bv = b_clean.d[i];

            if grade.mono {
                // Classic Leica/Tri-X panchromatic orange-red filter response: deepens blue skies,
                // brightens skin cleanly, and gives dramatic sculptural tonal separation.
                let pan = (lv + 0.34 * av + 0.14 * bv).clamp(0.0, 1.0);
                lv = pan;
                av = 0.0;
                bv = 0.0;
            } else {
                let c = (av * av + bv * bv).sqrt();
                if c > 1e-5 {
                    let h_rad = bv.atan2(av);
                    // 1. Skin-tone mask (OkLCh hue ~46° = 0.80 rad, moderate chroma & mid-lightness).
                    let d_skin = angle_diff(h_rad, 0.80);
                    let skin_h = (-d_skin * d_skin / (2.0 * 0.36 * 0.36)).exp();
                    let skin_c = (1.0 - ((c - 0.085).abs() / 0.11).clamp(0.0, 1.0)).max(0.0);
                    let skin_l = (1.0 - ((lv - 0.62).abs() / 0.35).clamp(0.0, 1.0)).max(0.0);
                    let skin_w = skin_h * skin_c * skin_l;

                    // 2. Skin-protected perceptual vibrance + highlight chroma rolloff.
                    let norm_c = (c / 0.24).min(1.0);
                    let vib_eff = grade.vibrance * (1.0 - 0.72 * skin_w);
                    let sat_eff = 1.0 + (grade.saturation - 1.0) * (1.0 - 0.6 * skin_w);
                    let hi_taper = 1.0 - 0.25 * ((lv - 0.82) / 0.18).clamp(0.0, 1.0).powi(2);
                    let mut c_new = c * sat_eff * (1.0 + vib_eff * (1.0 - norm_c)) * hi_taper;
                    let mut h_new = h_rad;

                    // 3. Polarizer-style sky depth (bright blues/cyans ~238° = -2.13 rad in atan2).
                    if grade.sky_pop > 0.0 && lv > 0.40 {
                        let d_sky = angle_diff(h_rad, -2.12);
                        let sky_w = (-d_sky * d_sky / (2.0 * 0.42 * 0.42)).exp()
                            * ((lv - 0.40) / 0.45).clamp(0.0, 1.0)
                            * (c / 0.07).clamp(0.0, 1.0);
                        lv = (lv - 0.022 * grade.sky_pop * sky_w).clamp(0.0, 1.0);
                        c_new *= 1.0 + 0.12 * grade.sky_pop * sky_w;
                    }

                    // 4. Natural lush foliage harmony (tames neon yellow-green phone grass ~132° = 2.30 rad).
                    if grade.foliage_shift > 0.0 {
                        let d_fol = angle_diff(h_rad, 2.28);
                        let fol_w = (-d_fol * d_fol / (2.0 * 0.35 * 0.35)).exp() * (c / 0.06).clamp(0.0, 1.0);
                        h_new += 0.07 * grade.foliage_shift * fol_w;
                        lv = (lv - 0.012 * grade.foliage_shift * fol_w).clamp(0.0, 1.0);
                        c_new *= 1.0 - 0.04 * grade.foliage_shift * fol_w;
                    }

                    // 5. Subtle skin polish: tiny warm luminous lift on skin midtones.
                    if skin_w > 0.05 {
                        lv = (lv + 0.008 * skin_w).clamp(0.0, 1.0);
                    }

                    av = c_new * h_new.cos();
                    bv = c_new * h_new.sin();
                }

                // 6. Editorial warmth & split toning (cool slate shadow clarity + warm golden highlight glow).
                if grade.warmth_a != 0.0 || grade.warmth_b != 0.0 {
                    let w_mid = (4.0 * lv * (1.0 - lv)).max(0.0);
                    av += grade.warmth_a * w_mid;
                    bv += grade.warmth_b * w_mid;
                }
                if grade.split_cool > 0.0 || grade.split_warm > 0.0 {
                    let sh_w = ((0.38 - lv) / 0.38).clamp(0.0, 1.0);
                    let sh_w = sh_w * sh_w * (lv / 0.06).clamp(0.0, 1.0);
                    let hi_w = ((lv - 0.56) / 0.38).clamp(0.0, 1.0);
                    let hi_w = hi_w * hi_w * ((1.0 - lv) / 0.08).clamp(0.0, 1.0);
                    // Cool teal-slate in shadows (-a, -b), warm golden ivory in highlights (+a, +b).
                    av += -0.45 * grade.split_cool * sh_w + 0.35 * grade.split_warm * hi_w;
                    bv += -1.0 * grade.split_cool * sh_w + 1.0 * grade.split_warm * hi_w;
                }
            }

            // Optional subtle organic 35mm film grain (Film & Mono looks, midtone-weighted).
            if grade.grain > 0.0 {
                let g_mid = (4.0 * lv * (1.0 - lv)).max(0.0);
                lv = (lv + grade.grain * g_mid * (tri_dither(x ^ 0x5A5A, row ^ 0xA5A5) * 2.2)).clamp(0.0, 1.0);
            }

            // Subtle natural prime-lens vignette polish (draws eye gently toward subject).
            if grade.vignette > 0.0 {
                let nx = (x as f32 + 0.5) * inv_w - 0.5;
                let r2 = (nx * nx + ny2) * 4.0; // 0 at center, ~2 at extreme corners
                let vig = (1.0 - grade.vignette * (r2 * r2).min(2.0) * 0.5).clamp(0.86, 1.0);
                lv *= vig;
            }

            // Exact perceptual OkLCh gamut mapping back into [0, 1] linear sRGB (zero hue clipping!).
            let (rl, gl, bl) = oklab_to_gamut_srgb(lv, av, bv);
            let (re, ge, be) = (srgb_encode(rl), srgb_encode(gl), srgb_encode(bl));
            let d = tri_dither(x, row);
            let quant = |v: f32| -> u8 {
                if v <= 0.002 {
                    0
                } else if v >= 0.998 {
                    255
                } else {
                    (v * 255.0 + 0.5 + d).clamp(0.0, 255.0) as u8
                }
            };
            px[3 * x] = quant(re);
            px[3 * x + 1] = quant(ge);
            px[3 * x + 2] = quant(be);
        }
    });
    (out, FinishInfo { total_gain, fusion_evs: plan.evs, fusion_strength: plan.strength, luma_sigma })
}

/// Cheap conversion of linear RGB to 8-bit sRGB for previews (no tone or detail processing).
pub fn quick_srgb8(rgb: &Rgb, gain: f32) -> Vec<u8> {
    let (w, h) = (rgb.w, rgb.h);
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(row, px)| {
        for x in 0..w {
            let i = row * w + x;
            let (mut r, mut g, mut b) = (rgb.c[0][i] * gain, rgb.c[1][i] * gain, rgb.c[2][i] * gain);
            let s = rolloff_scale(r.max(g).max(b));
            r *= s;
            g *= s;
            b *= s;
            px[3 * x] = (srgb_encode(r) * 255.0 + 0.5) as u8;
            px[3 * x + 1] = (srgb_encode(g) * 255.0 + 0.5) as u8;
            px[3 * x + 2] = (srgb_encode(b) * 255.0 + 0.5) as u8;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{textured, Rng};
    use crate::tone::s_curve;

    fn scene(w: usize, h: usize, scale: f32) -> Rgb {
        let t = textured(w, h);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            img.c[0][i] = t.d[i] * scale * 0.9;
            img.c[1][i] = t.d[i] * scale;
            img.c[2][i] = t.d[i] * scale * 0.7;
        }
        img
    }

    fn params(scene: u8, gain: f32) -> FinishParams {
        FinishParams {
            scene,
            ev_gain: gain,
            noise: NoiseModel { s: 1e-3, o: 1e-5 },
            frames: 6,
            display_referred: false,
            look: PhotoLook::Pro,
        }
    }

    #[test]
    fn output_has_right_size_and_uses_the_range() {
        let (out, info) = finish(scene(96, 64, 0.5), &params(0, 1.0));
        assert_eq!(out.len(), 96 * 64 * 3);
        assert!(out.iter().any(|v| *v > 150) && out.iter().any(|v| *v < 100));
        assert!(info.total_gain > 0.5 && info.total_gain < 3.0);
    }

    #[test]
    fn all_five_looks_render_cleanly_and_mono_is_achromatic() {
        for look in [PhotoLook::Pro, PhotoLook::Natural, PhotoLook::Vivid, PhotoLook::Film, PhotoLook::Mono] {
            let mut p = params(0, 1.0);
            p.look = look;
            let (out, _) = finish(scene(64, 48, 0.45), &p);
            assert_eq!(out.len(), 64 * 48 * 3);
            if look == PhotoLook::Mono {
                for px in out.chunks_exact(3) {
                    assert!((px[0] as i32 - px[1] as i32).abs() <= 1 && (px[1] as i32 - px[2] as i32).abs() <= 1);
                }
            }
        }
    }

    #[test]
    fn ev_gain_brightens_underexposed_input() {
        let dark = scene(96, 64, 0.04);
        let mean = |v: &[u8]| v.iter().map(|x| *x as f32).sum::<f32>() / v.len() as f32;
        let (a, _) = finish(dark.clone(), &params(2, 1.0));
        let (b, _) = finish(dark, &params(2, 4.0));
        assert!(mean(&b) > mean(&a) + 10.0, "{} vs {}", mean(&a), mean(&b));
    }

    #[test]
    fn hdr_scene_lifts_shadows() {
        let (w, h) = (128, 64);
        let t = textured(w, h);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            let k = if i % w < w / 2 { 0.01 } else { 1.6 };
            for c in 0..3 {
                img.c[c][i] = t.d[i] * k;
            }
        }
        let mut flat = params(1, 1.0);
        flat.frames = 8;
        let (out, info) = finish(img, &flat);
        assert!(info.fusion_strength > 0.0 && info.fusion_evs.len() >= 2);
        let dark_mean: f32 = (0..h)
            .flat_map(|y| (8..24).map(move |x| (y, x)))
            .map(|(y, x)| out[(y * w + x) * 3 + 1] as f32)
            .sum::<f32>()
            / (h * 16) as f32;
        assert!(dark_mean > 20.0, "shadows stayed black: {}", dark_mean);
    }

    #[test]
    fn deep_shadows_stay_dark_and_contrast_is_kept() {
        let (w, h) = (128, 64);
        let t = textured(w, h);
        let mut img = Rgb::new(w, h);
        for i in 0..w * h {
            let k = if i % w < 80 { 0.004 } else { 0.5 };
            for c in 0..3 {
                img.c[c][i] = t.d[i] * k;
            }
        }
        let (out, _) = finish(img, &params(1, 1.0));
        let mean = |x0: usize, x1: usize| -> f32 {
            (0..h)
                .flat_map(|y| (x0..x1).map(move |x| (y, x)))
                .map(|(y, x)| out[(y * w + x) * 3 + 1] as f32)
                .sum::<f32>()
                / (h * (x1 - x0)) as f32
        };
        let (dark, bright) = (mean(8, 64), mean(96, 120));
        assert!(dark < 75.0, "blacks lifted to grey: {}", dark);
        assert!(bright - dark > 80.0, "flat: dark {} bright {}", dark, bright);
    }

    #[test]
    fn s_curve_is_monotonic_with_fixed_ends() {
        assert!(s_curve(0.0, 0.3).abs() < 1e-6 && (s_curve(1.0, 0.3) - 1.0).abs() < 1e-6);
        let mut last = -1.0;
        for i in 0..=100 {
            let v = s_curve(i as f32 / 100.0, 0.3);
            assert!(v >= last);
            last = v;
        }
        assert!(s_curve(0.25, 0.3) < 0.25 && s_curve(0.75, 0.3) > 0.75);
    }

    #[test]
    fn noisy_input_comes_out_smoother() {
        let (w, h) = (96, 64);
        let mut r = Rng::new(9);
        let mut img = Rgb::new(w, h);
        let mut p = params(2, 1.0);
        p.frames = 1;
        let sd = p.noise.var(0.2).sqrt();
        for i in 0..w * h {
            for c in 0..3 {
                img.c[c][i] = 0.2 + sd * r.gauss();
            }
        }
        let var_in = crate::testutil::variance(&img.c[1]);
        let (out, _) = finish(img, &p);
        let g: Vec<f32> = (0..w * h).map(|i| out[3 * i + 1] as f32 / 255.0).collect();
        let var_out = crate::testutil::variance(&g);
        assert!(var_out < var_in * 1.44 * 0.7, "in {} out {}", var_in * 1.44, var_out);
    }
}
