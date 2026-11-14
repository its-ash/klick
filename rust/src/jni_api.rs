//! JNI binding for `com.itsash.klick.NativeEngine`. Frames arrive as direct ByteBuffers (no copy);
//! results are written to a file path and a one-line status string is returned
//! (`OK <width> <height> <frames> <total_gain> <fusion_strength> <ref_index>` or `ERR <message>`).
//! The reference calls report `... 1 1.000 0.000 0`.

use crate::meta::Meta;
use crate::pipeline::{process_raw, process_yuv, reference_raw, reference_yuv, YuvFrame, YuvLayout, DEFAULT_MAX_PIXELS};
use crate::raw::Lsc;
use jni::objects::{JByteBuffer, JClass, JFloatArray, JIntArray, JObject, JObjectArray, JString};
use jni::sys::{jint, jstring};
use jni::JNIEnv;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn buffers<'a>(env: &mut JNIEnv, arr: &JObjectArray) -> Result<Vec<&'a [u8]>, String> {
    let n = env.get_array_length(arr).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let obj = env.get_object_array_element(arr, i).map_err(|e| e.to_string())?;
        let bb = JByteBuffer::from(obj);
        let ptr = env.get_direct_buffer_address(&bb).map_err(|e| format!("buffer {} is not direct: {}", i, e))?;
        let cap = env.get_direct_buffer_capacity(&bb).map_err(|e| e.to_string())?;
        if ptr.is_null() {
            return Err(format!("buffer {} is null", i));
        }
        // The Kotlin side keeps the buffers alive for the duration of the call.
        out.push(unsafe { std::slice::from_raw_parts(ptr as *const u8, cap) });
    }
    Ok(out)
}

fn floats(env: &mut JNIEnv, arr: &JObject) -> Result<Vec<f32>, String> {
    if arr.is_null() {
        return Ok(Vec::new());
    }
    let a = JFloatArray::from(unsafe { JObject::from_raw(arr.as_raw()) });
    let len = env.get_array_length(&a).map_err(|e| e.to_string())? as usize;
    let mut v = vec![0.0f32; len];
    env.get_float_array_region(&a, 0, &mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

fn string(env: &mut JNIEnv, s: &JString) -> Result<String, String> {
    env.get_string(s).map(|s| s.into()).map_err(|e| e.to_string())
}

fn finish_call(env: &mut JNIEnv, r: std::thread::Result<Result<String, String>>) -> jstring {
    let msg = match r {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => format!("ERR {}", e),
        Err(_) => "ERR engine panicked".to_string(),
    };
    env.new_string(msg).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

fn lsc_from(meta: &Meta, data: &[f32]) -> Option<Lsc> {
    Lsc::from_slice(meta.lsc_w, meta.lsc_h, data)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_itsash_klick_NativeEngine_processRaw(
    mut env: JNIEnv,
    _class: JClass,
    frames: JObjectArray,
    width: jint,
    height: jint,
    row_stride: jint,
    meta: JFloatArray,
    lsc: JObject,
    out_path: JString,
    quality: jint,
) -> jstring {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let bufs = buffers(&mut env, &frames)?;
        let meta_v = floats(&mut env, &meta)?;
        let m = Meta::from_slice(&meta_v)?;
        let lsc_v = floats(&mut env, &lsc)?;
        let l = lsc_from(&m, &lsc_v);
        let path = string(&mut env, &out_path)?;
        let out = process_raw(&bufs, width as usize, height as usize, row_stride as usize, &m, l.as_ref(), quality.clamp(1, 100) as u8, DEFAULT_MAX_PIXELS)?;
        std::fs::write(&path, &out.jpeg).map_err(|e| format!("write {}: {}", path, e))?;
        Ok(format!("OK {} {} {} {:.3} {:.3} {}", out.width, out.height, out.frames, out.info.total_gain, out.info.fusion_strength, out.ref_index))
    }));
    finish_call(&mut env, r)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_itsash_klick_NativeEngine_referenceRaw(
    mut env: JNIEnv,
    _class: JClass,
    frame: JObjectArray,
    width: jint,
    height: jint,
    row_stride: jint,
    meta: JFloatArray,
    lsc: JObject,
    out_path: JString,
    quality: jint,
) -> jstring {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let bufs = buffers(&mut env, &frame)?;
        let first = bufs.first().ok_or("no frame")?;
        let meta_v = floats(&mut env, &meta)?;
        let m = Meta::from_slice(&meta_v)?;
        let lsc_v = floats(&mut env, &lsc)?;
        let l = lsc_from(&m, &lsc_v);
        let path = string(&mut env, &out_path)?;
        let (jpeg, w, h) = reference_raw(first, width as usize, height as usize, row_stride as usize, &m, l.as_ref(), quality.clamp(1, 100) as u8)?;
        std::fs::write(&path, &jpeg).map_err(|e| format!("write {}: {}", path, e))?;
        Ok(format!("OK {} {} 1 1.000 0.000 0", w, h))
    }));
    finish_call(&mut env, r)
}

fn yuv_frames<'a>(bufs: &'a [&'a [u8]]) -> Result<Vec<YuvFrame<'a>>, String> {
    if bufs.is_empty() || bufs.len() % 3 != 0 {
        return Err("YUV planes must come in groups of three (Y, U, V)".into());
    }
    Ok(bufs.chunks(3).map(|c| YuvFrame { y: c[0], u: c[1], v: c[2] }).collect())
}

fn layout_from(env: &mut JNIEnv, w: jint, h: jint, strides: &JIntArray) -> Result<YuvLayout, String> {
    let mut s = [0i32; 3];
    env.get_int_array_region(strides, 0, &mut s).map_err(|e| e.to_string())?;
    if s.iter().any(|v| *v <= 0) {
        return Err("bad YUV strides".into());
    }
    Ok(YuvLayout { w: w as usize, h: h as usize, y_row: s[0] as usize, uv_row: s[1] as usize, uv_pixel: s[2] as usize })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_itsash_klick_NativeEngine_processYuv(
    mut env: JNIEnv,
    _class: JClass,
    planes: JObjectArray,
    width: jint,
    height: jint,
    strides: JIntArray,
    meta: JFloatArray,
    out_path: JString,
    quality: jint,
) -> jstring {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let bufs = buffers(&mut env, &planes)?;
        let frames = yuv_frames(&bufs)?;
        let layout = layout_from(&mut env, width, height, &strides)?;
        let meta_v = floats(&mut env, &meta)?;
        let m = Meta::from_slice(&meta_v)?;
        let path = string(&mut env, &out_path)?;
        let out = process_yuv(&frames, &layout, &m, quality.clamp(1, 100) as u8)?;
        std::fs::write(&path, &out.jpeg).map_err(|e| format!("write {}: {}", path, e))?;
        Ok(format!("OK {} {} {} {:.3} {:.3} {}", out.width, out.height, out.frames, out.info.total_gain, out.info.fusion_strength, out.ref_index))
    }));
    finish_call(&mut env, r)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_itsash_klick_NativeEngine_referenceYuv(
    mut env: JNIEnv,
    _class: JClass,
    planes: JObjectArray,
    width: jint,
    height: jint,
    strides: JIntArray,
    out_path: JString,
    quality: jint,
) -> jstring {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let bufs = buffers(&mut env, &planes)?;
        let frames = yuv_frames(&bufs)?;
        let layout = layout_from(&mut env, width, height, &strides)?;
        let path = string(&mut env, &out_path)?;
        let (jpeg, w, h) = reference_yuv(&frames[0], &layout, quality.clamp(1, 100) as u8)?;
        std::fs::write(&path, &jpeg).map_err(|e| format!("write {}: {}", path, e))?;
        Ok(format!("OK {} {} 1 1.000 0.000 0", w, h))
    }));
    finish_call(&mut env, r)
}

#[no_mangle]
pub extern "system" fn Java_com_itsash_klick_NativeEngine_version(env: JNIEnv, _class: JClass) -> jstring {
    env.new_string(env!("CARGO_PKG_VERSION")).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}
