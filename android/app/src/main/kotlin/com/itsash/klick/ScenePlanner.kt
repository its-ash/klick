package com.itsash.klick

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Chooses scene, frame count and per-frame exposure from the live metering.
 * Mirrored exactly by `lib/core/scene.dart` (which has the unit tests): change both together.
 */
object ScenePlanner {
    enum class Scene(val code: Int, val label: String) { NORMAL(0, "normal"), HDR(1, "hdr"), NIGHT(2, "night") }
    enum class Shake { STEADY, MODERATE, SHAKY }

    const val NIGHT_BELOW_EV = 5.0
    const val HDR_FROM_EV = 12.0
    const val STEADY_RMS = 0.03
    const val MODERATE_RMS = 0.12

    class Input(val iso: Int, val exposureNs: Long, val aperture: Double, val shakeRms: Double, val maxIso: Int)

    class Plan(
        val scene: Scene, val frames: Int, val exposureNs: Long, val iso: Int,
        /** Digital gain that restores the deliberate under-exposure. */
        val evGain: Double, val ev: Double, val shake: Shake,
    )

    private fun log2(x: Double) = ln(x) / ln(2.0)

    /** Exposure value at ISO 100 of the metered scene. */
    fun sceneEv(iso: Int, exposureNs: Long, aperture: Double): Double {
        val n = if (aperture > 0) aperture else 1.8
        val t = max(exposureNs, 1L) / 1e9
        return log2(n * n / t) - log2(max(iso, 1) / 100.0)
    }

    fun shakeOf(rms: Double) = when {
        rms < STEADY_RMS -> Shake.STEADY
        rms < MODERATE_RMS -> Shake.MODERATE
        else -> Shake.SHAKY
    }

    fun plan(i: Input): Plan {
        val ev = sceneEv(i.iso, i.exposureNs, i.aperture)
        val shake = shakeOf(i.shakeRms)
        val scene = when {
            ev < NIGHT_BELOW_EV -> Scene.NIGHT
            ev >= HDR_FROM_EV -> Scene.HDR
            else -> Scene.NORMAL
        }
        val frames = when (scene) {
            Scene.NIGHT -> (8 + ((NIGHT_BELOW_EV - ev) * 1.5).roundToInt()).coerceIn(8, 12)
            Scene.HDR -> 4
            Scene.NORMAL -> if (ev >= 9.0) 3 else 5
        }
        val under = when (scene) {
            Scene.HDR, Scene.NIGHT -> 0.5
            Scene.NORMAL -> if (ev >= 9.0) 0.8 else 0.7
        }
        val capNs = when (shake) {
            Shake.STEADY -> if (scene == Scene.NIGHT) 250_000_000L else 125_000_000L
            Shake.MODERATE -> 33_000_000L
            Shake.SHAKY -> 16_000_000L
        }
        val desired = max((i.exposureNs * under).toLong(), 1L)
        val t = min(desired, capNs)
        val iso = min(i.maxIso.toDouble(), (i.iso * desired.toDouble() / t).roundToInt().toDouble()).roundToInt()
        val evGain = max(1.0, (i.iso.toDouble() * i.exposureNs) / (iso.toDouble() * t))
        return Plan(scene, frames, t, iso, evGain, ev, shake)
    }
}
