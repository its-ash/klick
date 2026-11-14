package com.itsash.klick

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.OrientationEventListener
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Rolling RMS of the gyroscope magnitude (rad/s) over about half a second, plus the device rotation.
 * Also reports the roll for the on-screen level: [onLevel] gets degrees clockwise from natural
 * upright and whether the phone is lying flat (roll is meaningless then). Called on the main thread.
 */
class ShakeMonitor(context: Context, private val onLevel: (roll: Double, flat: Boolean) -> Unit = { _, _ -> }) : SensorEventListener {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravity: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GRAVITY) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val g = FloatArray(3)
    private var gReady = false
    private var lastLevelMs = 0L
    private val window = DoubleArray(64)
    private var count = 0
    private var next = 0

    @Volatile var rms: Double = 0.0
        private set

    /** Device rotation snapped to 0, 90, 180, 270 (clockwise from natural upright). */
    @Volatile var deviceRotation: Int = 0
        private set

    private val orientation = object : OrientationEventListener(context) {
        override fun onOrientationChanged(degrees: Int) {
            if (degrees == ORIENTATION_UNKNOWN) return
            deviceRotation = ((degrees + 45) / 90 * 90) % 360
        }
    }

    fun start() {
        gyro?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gravity?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
        if (orientation.canDetectOrientation()) orientation.enable()
    }

    fun stop() {
        sm.unregisterListener(this)
        orientation.disable()
        count = 0
        next = 0
        rms = 0.0
        gReady = false
    }

    override fun onSensorChanged(e: SensorEvent) {
        if (e.sensor.type != Sensor.TYPE_GYROSCOPE) {
            onGravity(e.values)
            return
        }
        val m2 = (e.values[0] * e.values[0] + e.values[1] * e.values[1] + e.values[2] * e.values[2]).toDouble()
        window[next] = m2
        next = (next + 1) % window.size
        if (count < window.size) count++
        var s = 0.0
        for (i in 0 until count) s += window[i]
        rms = sqrt(s / count)
    }

    private fun onGravity(v: FloatArray) {
        if (!gReady) {
            v.copyInto(g, endIndex = 3); gReady = true
        } else {
            for (i in 0..2) g[i] += 0.2f * (v[i] - g[i]) // low-pass; also smooths a raw accelerometer
        }
        val now = System.currentTimeMillis()
        if (now - lastLevelMs < 50) return
        lastLevelMs = now
        val norm = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        if (norm < 1f) return
        val roll = Math.toDegrees(atan2(-g[0].toDouble(), g[1].toDouble()))
        onLevel(roll, abs(g[2]) / norm > 0.85f)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
