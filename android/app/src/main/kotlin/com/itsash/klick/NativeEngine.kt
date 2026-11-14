package com.itsash.klick

import java.nio.ByteBuffer

/** JNI binding to the Rust engine (`rust/`, crate klick-engine). All calls are blocking: run them off the main thread. */
object NativeEngine {
    val available: Boolean = try {
        System.loadLibrary("klick_engine")
        true
    } catch (e: UnsatisfiedLinkError) {
        false
    }

    /** Status line: `OK w h frames gain fusion refIndex` or `ERR message`. Buffers must be direct. */
    @JvmStatic
    external fun processRaw(
        frames: Array<ByteBuffer>, width: Int, height: Int, rowStride: Int,
        meta: FloatArray, lsc: FloatArray?, outPath: String, quality: Int,
    ): String

    /** Quick half-resolution render of one RAW frame (frames holds exactly one buffer). */
    @JvmStatic
    external fun referenceRaw(
        frame: Array<ByteBuffer>, width: Int, height: Int, rowStride: Int,
        meta: FloatArray, lsc: FloatArray?, outPath: String, quality: Int,
    ): String

    /** planes = Y, U, V for each frame, in order. strides = [yRow, uvRow, uvPixel]. */
    @JvmStatic
    external fun processYuv(
        planes: Array<ByteBuffer>, width: Int, height: Int, strides: IntArray,
        meta: FloatArray, outPath: String, quality: Int,
    ): String

    @JvmStatic
    external fun referenceYuv(
        planes: Array<ByteBuffer>, width: Int, height: Int, strides: IntArray,
        outPath: String, quality: Int,
    ): String

    @JvmStatic
    external fun version(): String

    class Status(val ok: Boolean, val message: String, val width: Int = 0, val height: Int = 0, val frames: Int = 0, val refIndex: Int = 0)

    fun parse(line: String): Status {
        if (!line.startsWith("OK")) return Status(false, line.removePrefix("ERR").trim().ifEmpty { "engine failed" })
        val p = line.split(" ")
        return Status(
            true, line,
            width = p.getOrNull(1)?.toIntOrNull() ?: 0,
            height = p.getOrNull(2)?.toIntOrNull() ?: 0,
            frames = p.getOrNull(3)?.toIntOrNull() ?: 0,
            refIndex = p.getOrNull(6)?.toIntOrNull() ?: 0,
        )
    }
}
