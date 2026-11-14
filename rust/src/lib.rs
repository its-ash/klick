//! Klick engine: RAW/YUV burst alignment, merge, develop, tone map, denoise, sharpen, JPEG.
//!
//! Every stage is a pure function over plain buffers (see the module docs) so it can be tested
//! on the host. `jni_api` is the thin Android binding.

pub mod align;
pub mod color;
pub mod demosaic;
pub mod denoise;
pub mod encode;
pub mod finish;
pub mod merge;
pub mod meta;
pub mod pipeline;
pub mod plane;
pub mod raw;
pub mod sharpen;
pub mod tone;

#[cfg(test)]
pub(crate) mod testutil;

pub mod jni_api;
