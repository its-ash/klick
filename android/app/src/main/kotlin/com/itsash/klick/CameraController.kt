package com.itsash.klick

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.MeteringRectangle
import android.location.Location
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Size
import android.view.Surface
import io.flutter.view.TextureRegistry
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Camera2 back camera: preview into a Flutter texture, continuous AF/AE/AWB, tap to focus,
 * and automatic RAW_SENSOR (or YUV_420_888 fallback) bursts handed to [PhotoProcessor].
 */
class CameraController(
    private val context: Context,
    private val textures: TextureRegistry,
    private val processor: PhotoProcessor,
    private val shake: ShakeMonitor,
    private val emit: (Map<String, Any?>) -> Unit,
) {
    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val thread = HandlerThread("klick-camera").apply { start() }
    private val handler = Handler(thread.looper)
    private val main = Handler(Looper.getMainLooper())

    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var entry: TextureRegistry.SurfaceTextureEntry? = null
    private var previewSurface: Surface? = null
    private var reader: ImageReader? = null
    private var chars: CameraCharacteristics? = null
    private var lensId: String? = null
    private var useRaw = false
    private var captureSize = Size(0, 0)
    private var previewBuilder: CaptureRequest.Builder? = null
    private var lastResult: TotalCaptureResult? = null
    private var sceneTick = 0L

    @Volatile private var burst: Burst? = null
    @Volatile private var capturing = false

    class Lens(val id: String, val label: String, val zoom: Double)

    // ---- lenses -----------------------------------------------------------------------------

    private fun equivFocal(id: String): Double? {
        val c = manager.getCameraCharacteristics(id)
        val f = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
        val s = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
        return f.toDouble() * 36.0 / s.width
    }

    fun lenses(): List<Lens> {
        val backs = manager.cameraIdList.filter {
            manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK &&
                (manager.getCameraCharacteristics(it).get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf())
                    .contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE)
        }
        val withFocal = backs.mapNotNull { id -> equivFocal(id)?.let { id to it } }
        if (withFocal.isEmpty()) return backs.map { Lens(it, "1x", 1.0) }
        val main = withFocal.minByOrNull { abs(it.second - 26.0) }!!
        val seen = mutableSetOf<Int>()
        return withFocal.map { (id, f) ->
            val z = f / main.second
            Lens(id, zoomLabel(z), z)
        }.sortedBy { it.zoom }.filter { seen.add((it.zoom * 2).toInt()) }
    }

    private fun zoomLabel(z: Double): String {
        val r = Math.round(z * 2) / 2.0
        return if (r == r.toLong().toDouble()) "${r.toLong()}x" else "${r}x"
    }

    // ---- open / close -----------------------------------------------------------------------

    @SuppressLint("MissingPermission")
    fun start(requestedLens: String?, done: (Result<Map<String, Any?>>) -> Unit) {
        handler.post {
            try {
                closeInternal()
                val lenses = lenses()
                val id = requestedLens?.takeIf { r -> lenses.any { it.id == r } }
                    ?: lenses.firstOrNull { abs(it.zoom - 1.0) < 0.2 }?.id
                    ?: lenses.firstOrNull()?.id
                    ?: error("No back camera found")
                val c = manager.getCameraCharacteristics(id)
                chars = c
                lensId = id
                val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
                val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
                val rawSizes = map.getOutputSizes(ImageFormat.RAW_SENSOR)
                val cfa = c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT) ?: 4
                useRaw = caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW) && !rawSizes.isNullOrEmpty() && cfa in 0..3
                captureSize = if (useRaw) {
                    rawSizes.maxByOrNull { it.width.toLong() * it.height }!!
                } else {
                    val sizes = map.getOutputSizes(ImageFormat.YUV_420_888).toList()
                    sizes.filter { it.width.toLong() * it.height <= 13_000_000L }.maxByOrNull { it.width.toLong() * it.height }
                        ?: sizes.minByOrNull { it.width.toLong() * it.height }!!
                }
                captureSize = Size(captureSize.width and 1.inv(), captureSize.height and 1.inv())
                val aspect = captureSize.width.toDouble() / captureSize.height
                val previewSizes = map.getOutputSizes(SurfaceTexture::class.java).toList()
                val preview = previewSizes.filter { abs(it.width.toDouble() / it.height - aspect) < 0.02 && it.width <= 1920 }
                    .maxByOrNull { it.width * it.height }
                    ?: previewSizes.filter { it.width <= 1920 }.maxByOrNull { it.width * it.height } ?: previewSizes.first()

                val e = textures.createSurfaceTexture()
                entry = e
                e.surfaceTexture().setDefaultBufferSize(preview.width, preview.height)
                previewSurface = Surface(e.surfaceTexture())

                val fmt = if (useRaw) ImageFormat.RAW_SENSOR else ImageFormat.YUV_420_888
                val r = ImageReader.newInstance(captureSize.width, captureSize.height, fmt, 5)
                r.setOnImageAvailableListener({ rd -> onImage(rd) }, handler)
                reader = r

                manager.openCamera(id, object : CameraDevice.StateCallback() {
                    override fun onOpened(camera: CameraDevice) {
                        device = camera
                        createSession(camera, preview, lenses, done)
                    }

                    override fun onDisconnected(camera: CameraDevice) {
                        camera.close(); device = null
                        main.post { emit(mapOf("type" to "error", "message" to "Camera disconnected")) }
                    }

                    override fun onError(camera: CameraDevice, error: Int) {
                        camera.close(); device = null
                        main.post { done(Result.failure(IllegalStateException("Camera error $error"))) }
                    }
                }, handler)
            } catch (t: Throwable) {
                main.post { done(Result.failure(t)) }
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun createSession(camera: CameraDevice, preview: Size, lenses: List<Lens>, done: (Result<Map<String, Any?>>) -> Unit) {
        try {
            camera.createCaptureSession(
                listOf(previewSurface!!, reader!!.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) {
                        session = s
                        startPreview()
                        val c = chars!!
                        val info = mapOf<String, Any?>(
                            "textureId" to entry!!.id(),
                            "previewWidth" to preview.width,
                            "previewHeight" to preview.height,
                            "sensorOrientation" to (c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90),
                            "raw" to useRaw,
                            "captureWidth" to captureSize.width,
                            "captureHeight" to captureSize.height,
                            "flash" to (c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false),
                            "lensId" to lensId,
                            "lenses" to lenses.map { mapOf("id" to it.id, "label" to it.label, "zoom" to it.zoom) },
                        )
                        main.post { done(Result.success(info)) }
                    }

                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        main.post { done(Result.failure(IllegalStateException("Could not configure the camera"))) }
                    }
                },
                handler,
            )
        } catch (t: Throwable) {
            main.post { done(Result.failure(t)) }
        }
    }

    private fun startPreview() {
        val cam = device ?: return
        val s = session ?: return
        val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        b.addTarget(previewSurface!!)
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        b.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        b.set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE_ON)
        previewBuilder = b
        s.setRepeatingRequest(b.build(), previewCallback, handler)
    }

    private val previewCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            lastResult = result
            val now = System.currentTimeMillis()
            if (now - sceneTick < 400) return
            sceneTick = now
            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: return
            val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: return
            val ap = result.get(CaptureResult.LENS_APERTURE) ?: 0f
            val plan = ScenePlanner.plan(ScenePlanner.Input(iso, exp, ap.toDouble(), shake.rms, maxIso()))
            main.post {
                emit(
                    mapOf(
                        "type" to "scene", "iso" to iso, "exposureNs" to exp, "aperture" to ap, "shakeRms" to shake.rms,
                        "maxIso" to maxIso(), "scene" to plan.scene.label, "frames" to plan.frames, "ev" to plan.ev,
                    ),
                )
            }
        }
    }

    fun stop(done: () -> Unit) {
        handler.post {
            closeInternal()
            main.post { done() }
        }
    }

    private fun closeInternal() {
        runCatching { session?.stopRepeating() }
        runCatching { session?.close() }
        runCatching { device?.close() }
        runCatching { reader?.close() }
        runCatching { previewSurface?.release() }
        runCatching { entry?.release() }
        session = null; device = null; reader = null; previewSurface = null; entry = null
        burst = null
        capturing = false
    }

    fun release() {
        stop {}
        thread.quitSafely()
    }

    private fun maxIso(): Int {
        val c = chars ?: return 800
        val analog = c.get(CameraCharacteristics.SENSOR_MAX_ANALOG_SENSITIVITY)
        val range = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        return analog ?: range?.upper ?: 800
    }

    // ---- tap to focus -----------------------------------------------------------------------

    /** nx, ny in 0..1 of the upright (portrait) preview. */
    fun tapFocus(nx: Double, ny: Double) {
        handler.post {
            val c = chars ?: return@post
            val s = session ?: return@post
            val b = previewBuilder ?: return@post
            val active: Rect = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return@post
            val orientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            val (sx, sy) = when (orientation) {
                90 -> ny to 1 - nx
                180 -> 1 - nx to 1 - ny
                270 -> 1 - ny to nx
                else -> nx to ny
            }
            val half = active.width() / 10
            val cx = active.left + (sx.coerceIn(0.0, 1.0) * active.width()).toInt()
            val cy = active.top + (sy.coerceIn(0.0, 1.0) * active.height()).toInt()
            val rect = Rect(
                max(active.left, cx - half), max(active.top, cy - half),
                min(active.right - 1, cx + half), min(active.bottom - 1, cy + half),
            )
            val region = arrayOf(MeteringRectangle(rect, MeteringRectangle.METERING_WEIGHT_MAX - 1))
            runCatching {
                b.set(CaptureRequest.CONTROL_AF_REGIONS, region)
                b.set(CaptureRequest.CONTROL_AE_REGIONS, region)
                b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                s.setRepeatingRequest(b.build(), previewCallback, handler)
                b.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                s.capture(b.build(), null, handler)
                b.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_IDLE)
            }
            handler.postDelayed({
                val bb = previewBuilder ?: return@postDelayed
                runCatching {
                    bb.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                    bb.set(CaptureRequest.CONTROL_AF_REGIONS, null)
                    bb.set(CaptureRequest.CONTROL_AE_REGIONS, null)
                    session?.setRepeatingRequest(bb.build(), previewCallback, handler)
                }
            }, 3500)
        }
    }

    // ---- capture ----------------------------------------------------------------------------

    class Burst(val expected: Int, val raw: Boolean) {
        val frames = ArrayList<Frame>()
        val results = HashMap<Long, TotalCaptureResult>()
        var failed = 0
        var finished = false
    }

    class Frame(val timestamp: Long, val buffers: List<ByteBuffer>, val rowStride: Int, val yuvStrides: IntArray)

    class Options(val flash: String, val quality: Int, val keepDng: Boolean, val location: Location?)

    fun capture(opts: Options, done: (Result<Map<String, Any?>>) -> Unit) {
        handler.post {
            val cam = device
            val s = session
            val c = chars
            val rd = reader
            if (cam == null || s == null || c == null || rd == null) return@post fail(done, "Camera is not ready")
            if (capturing) return@post fail(done, "Still capturing the previous burst")
            val res = lastResult ?: return@post fail(done, "Camera is still starting")
            val iso = res.get(CaptureResult.SENSOR_SENSITIVITY) ?: 100
            val exp = res.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 10_000_000L
            val ap = res.get(CaptureResult.LENS_APERTURE) ?: 1.8f
            var plan = ScenePlanner.plan(ScenePlanner.Input(iso, exp, ap.toDouble(), shake.rms, maxIso()))
            val useFlash = opts.flash == "auto" && (c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true) && plan.ev < 2.5
            val frameBytes = captureSize.width.toLong() * captureSize.height * (if (useRaw) 2 else 3) / (if (useRaw) 1 else 2)
            var n = if (useFlash) 1 else plan.frames
            n = min(n, max(1, (BURST_BYTES / frameBytes).toInt()))
            if (!processor.canAccept(frameBytes * n)) return@post fail(done, "Still processing earlier photos. Try again in a moment.")

            // Clamp the plan to what the sensor can do.
            val expRange = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            val t = expRange?.let { plan.exposureNs.coerceIn(it.lower, it.upper) } ?: plan.exposureNs
            val isoUsed = isoRange?.let { plan.iso.coerceIn(it.lower, it.upper) } ?: plan.iso
            val gain = max(1.0, (iso.toDouble() * exp) / (isoUsed.toDouble() * t))
            plan = ScenePlanner.Plan(plan.scene, n, t, isoUsed, gain, plan.ev, plan.shake)

            val b = Burst(n, useRaw)
            burst = b
            capturing = true
            pendingPlan = plan
            pendingOpts = opts
            pendingDone = done
            try {
                val builders = (0 until n).map { buildCaptureRequest(cam, rd, plan, useFlash) }
                val callback = object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                        synchronized(b) {
                            b.results[result.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L] = result
                            maybeFinish(b, plan, opts, done)
                        }
                    }

                    override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
                        synchronized(b) {
                            b.failed++
                            maybeFinish(b, plan, opts, done)
                        }
                    }
                }
                val fire = Runnable {
                    try {
                        s.stopRepeating()
                        s.captureBurst(builders.map { it.build() }, callback, handler)
                        handler.postDelayed({ synchronized(b) { if (!b.finished) { b.finished = true; abortBurst(b, plan, opts, done) } } }, 25_000)
                    } catch (t: Throwable) {
                        capturing = false
                        burst = null
                        startPreviewSafe()
                        fail(done, t.message ?: "Capture failed")
                    }
                }
                if (useFlash) {
                    runPrecapture(s)
                    handler.postDelayed(fire, 800)
                } else {
                    fire.run()
                }
            } catch (t: Throwable) {
                capturing = false
                burst = null
                startPreviewSafe()
                fail(done, t.message ?: "Capture failed")
            }
        }
    }

    private fun runPrecapture(s: CameraCaptureSession) {
        val cam = device ?: return
        val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        b.addTarget(previewSurface!!)
        b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH)
        b.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
        // Fire and forget: the burst is started after a short settle time (see capture()).
        s.capture(b.build(), null, handler)
    }

    private fun buildCaptureRequest(cam: CameraDevice, rd: ImageReader, plan: ScenePlanner.Plan, flash: Boolean): CaptureRequest.Builder {
        val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
        b.addTarget(rd.surface)
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        b.set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE_ON)
        if (flash) {
            b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_ALWAYS_FLASH)
            b.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_SINGLE)
        } else {
            b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, plan.exposureNs)
            b.set(CaptureRequest.SENSOR_SENSITIVITY, plan.iso)
            b.set(CaptureRequest.SENSOR_FRAME_DURATION, max(plan.exposureNs, 33_333_333L))
            b.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
        }
        if (useRaw) {
            // The engine does its own denoising and sharpening on the merged frames.
            b.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            b.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
        } else {
            b.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_FAST)
            b.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_FAST)
        }
        return b
    }

    private fun onImage(rd: ImageReader) {
        val img: Image = try { rd.acquireNextImage() } catch (e: Exception) { null } ?: return
        try {
            val b = burst
            if (b == null || b.finished) return
            val frame = copy(img, b.raw)
            synchronized(b) {
                b.frames.add(frame)
            }
        } catch (t: Throwable) {
            burst?.let { synchronized(it) { it.failed++ } }
        } finally {
            img.close()
        }
        burst?.let { b -> synchronized(b) { maybeFinishDeferred(b) } }
    }

    // Pending completion info, stored when the burst request was made.
    private var pendingPlan: ScenePlanner.Plan? = null
    private var pendingOpts: Options? = null
    private var pendingDone: ((Result<Map<String, Any?>>) -> Unit)? = null

    private fun maybeFinishDeferred(b: Burst) {
        val p = pendingPlan ?: return
        val o = pendingOpts ?: return
        val d = pendingDone ?: return
        maybeFinish(b, p, o, d)
    }

    private fun copy(img: Image, raw: Boolean): Frame {
        val planes = img.planes
        if (raw) {
            val p = planes[0]
            val src = p.buffer
            val dst = ByteBuffer.allocateDirect(src.remaining())
            dst.put(src)
            dst.rewind()
            return Frame(img.timestamp, listOf(dst), p.rowStride, IntArray(3))
        }
        val bufs = planes.map { p ->
            val src = p.buffer
            val dst = ByteBuffer.allocateDirect(src.remaining())
            dst.put(src)
            dst.rewind()
            dst
        }
        return Frame(img.timestamp, bufs, 0, intArrayOf(planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride))
    }

    private fun maybeFinish(b: Burst, plan: ScenePlanner.Plan, opts: Options, done: (Result<Map<String, Any?>>) -> Unit) {
        if (b.finished) return
        val want = b.expected - b.failed
        if (want <= 0) {
            b.finished = true
            abortBurst(b, plan, opts, done)
            return
        }
        if (b.frames.size < want || b.results.size < want) return
        b.finished = true
        val ordered = b.frames.sortedBy { it.timestamp }.filter { b.results.containsKey(it.timestamp) }
        if (ordered.isEmpty()) {
            abortBurst(b, plan, opts, done)
            return
        }
        val c = chars!!
        val sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val jpegOrientation = (sensorOrientation + shake.deviceRotation) % 360
        val job = BurstJob(
            id = processor.nextId(),
            raw = b.raw,
            width = captureSize.width,
            height = captureSize.height,
            buffers = ordered.flatMap { it.buffers },
            frameCount = ordered.size,
            rowStride = ordered[0].rowStride,
            yuvStrides = ordered[0].yuvStrides,
            results = ordered.map { b.results[it.timestamp]!! },
            characteristics = c,
            plan = plan,
            jpegOrientation = jpegOrientation,
            quality = opts.quality,
            keepDng = opts.keepDng,
            location = opts.location,
            aperture = lastResult?.get(CaptureResult.LENS_APERTURE) ?: 0f,
            focalLength = lastResult?.get(CaptureResult.LENS_FOCAL_LENGTH) ?: 0f,
        )
        burst = null
        capturing = false
        pendingPlan = null; pendingOpts = null; pendingDone = null
        startPreviewSafe()
        processor.submit(job)
        main.post { done(Result.success(mapOf("id" to job.id, "frames" to ordered.size, "scene" to plan.scene.label, "raw" to b.raw))) }
    }

    private fun abortBurst(b: Burst, plan: ScenePlanner.Plan, opts: Options, done: (Result<Map<String, Any?>>) -> Unit) {
        burst = null
        capturing = false
        startPreviewSafe()
        main.post { done(Result.failure(IllegalStateException("The burst did not complete"))) }
    }

    private fun startPreviewSafe() {
        runCatching { startPreview() }
    }

    private fun fail(done: (Result<Map<String, Any?>>) -> Unit, msg: String) {
        main.post { done(Result.failure(IllegalStateException(msg))) }
    }

    companion object {
        /** Direct memory for one burst, before the processor budget. */
        const val BURST_BYTES = 420L * 1024 * 1024
    }
}
