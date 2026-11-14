package com.itsash.klick

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.OrientationEventListener
import kotlin.math.sqrt

/** Rolling RMS of the gyroscope magnitude (rad/s) over about half a second, plus the device rotation. */
class ShakeMonitor(context: Context) : SensorEventListener {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
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
        if (orientation.canDetectOrientation()) orientation.enable()
    }

    fun stop() {
        sm.unregisterListener(this)
        orientation.disable()
        count = 0
        next = 0
        rms = 0.0
    }

    override fun onSensorChanged(e: SensorEvent) {
        val m2 = (e.values[0] * e.values[0] + e.values[1] * e.values[1] + e.values[2] * e.values[2]).toDouble()
        window[next] = m2
        next = (next + 1) % window.size
        if (count < window.size) count++
        var s = 0.0
        for (i in 0 until count) s += window[i]
        rms = sqrt(s / count)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
