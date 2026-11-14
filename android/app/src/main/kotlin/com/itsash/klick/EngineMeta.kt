package com.itsash.klick

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult

/** Builds the flat float array the Rust engine reads (layout documented in `rust/src/meta.rs`). */
object EngineMeta {
    private const val LEN = 64

    private fun cct(illuminant: Int?): Float = when (illuminant) {
        17 -> 2856f; 18 -> 4874f; 19 -> 6774f; 20 -> 5503f; 21 -> 6504f; 22 -> 7504f; 23 -> 5003f; 24 -> 3200f
        1, 4, 9 -> 5500f; 2, 14 -> 4150f; 3 -> 2850f; 10 -> 6500f; 11 -> 7500f; 12 -> 6430f; 13 -> 5000f; 15 -> 3500f
        else -> 5000f
    }

    private fun matrix(t: android.hardware.camera2.params.ColorSpaceTransform?): FloatArray? {
        t ?: return null
        val r = arrayOfNulls<android.util.Rational>(9)
        t.copyElements(r, 0)
        return FloatArray(9) { r[it]!!.toFloat() }
    }

    class Built(val meta: FloatArray, val lsc: FloatArray?)

    fun rawMeta(ch: CameraCharacteristics, res: TotalCaptureResult, plan: ScenePlanner.Plan): Built {
        val m = FloatArray(LEN)
        m[0] = (ch.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT) ?: 0).coerceIn(0, 3).toFloat()
        m[1] = (ch.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023).toFloat()
        val dyn = res.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
        if (dyn != null && dyn.size >= 4) {
            for (i in 0..3) m[2 + i] = dyn[i]
        } else {
            val pattern = IntArray(4)
            ch.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.copyTo(pattern, 0)
            for (i in 0..3) m[2 + i] = pattern[i].toFloat()
        }
        val neutral = res.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)
        if (neutral != null) {
            for (i in 0..2) m[6 + i] = neutral[i].toFloat()
        } else {
            val g = res.get(CaptureResult.COLOR_CORRECTION_GAINS)
            if (g != null) {
                m[6] = 1f / g.red; m[7] = 2f / (g.greenEven + g.greenOdd); m[8] = 1f / g.blue
            } else {
                m[6] = 1f; m[7] = 1f; m[8] = 1f
            }
        }
        matrix(ch.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX1))?.copyInto(m, 9)
        matrix(ch.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX2))?.copyInto(m, 18)
        matrix(ch.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1))?.copyInto(m, 27)
        matrix(ch.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2))?.copyInto(m, 36)
        m[45] = cct(ch.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1))
        m[46] = cct(ch.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2)?.toInt())
        val np = res.get(CaptureResult.SENSOR_NOISE_PROFILE)
        if (np != null && np.isNotEmpty()) {
            m[47] = np.map { it.first }.average().toFloat()
            m[48] = np.map { it.second }.average().toFloat()
        } else {
            val iso = (res.get(CaptureResult.SENSOR_SENSITIVITY) ?: 100).toFloat()
            m[47] = 2e-5f * iso / 100f
            m[48] = 1e-6f * iso / 100f
        }
        fill(m, res, plan)
        var lscData: FloatArray? = null
        val map = res.get(CaptureResult.STATISTICS_LENS_SHADING_CORRECTION_MAP)
        if (map != null && map.columnCount >= 2 && map.rowCount >= 2) {
            lscData = FloatArray(map.columnCount * map.rowCount * 4)
            map.copyGainFactors(lscData, 0)
            m[53] = map.columnCount.toFloat()
            m[54] = map.rowCount.toFloat()
        }
        return Built(m, lscData)
    }

    /** YUV frames are already ISP-processed: only noise level, scene and gain matter. */
    fun yuvMeta(res: TotalCaptureResult?, plan: ScenePlanner.Plan): FloatArray {
        val m = FloatArray(LEN)
        m[1] = 255f
        m[6] = 1f; m[7] = 1f; m[8] = 1f
        val iso = (res?.get(CaptureResult.SENSOR_SENSITIVITY) ?: plan.iso).toFloat()
        m[47] = 0f
        m[48] = 3.5e-5f * iso / 100f
        m[45] = 2856f; m[46] = 6504f
        fill(m, res, plan)
        m[51] = 1f
        return m
    }

    private fun fill(m: FloatArray, res: TotalCaptureResult?, plan: ScenePlanner.Plan) {
        m[49] = (res?.get(CaptureResult.SENSOR_SENSITIVITY) ?: plan.iso).toFloat()
        m[50] = (res?.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: plan.exposureNs).toFloat()
        m[51] = plan.evGain.toFloat()
        m[52] = plan.scene.code.toFloat()
    }
}
