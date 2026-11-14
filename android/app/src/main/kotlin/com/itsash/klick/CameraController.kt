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
 * Camera2 back cameras: preview into Flutter textures, continuous AF/AE/AWB, tap to focus, and
 * automatic RAW_SENSOR (or YUV_420_888 fallback) bursts handed to [PhotoProcessor].
 *
 * Where the phone has a close-focusing lens (usually the ultrawide), the main and macro cameras
 * run at the same time: the one on screen at full rate, the other as a low-rate standby whose
 * autofocus keeps reporting. Switching between them is then instant (swap the texture on screen
 * and the camera that takes the photo), and the auto macro decision can use both lenses' focus.
 * Phones that refuse two open cameras fall back to one camera and a restart per switch.
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

    /** One open camera: device, session, preview texture and capture reader (camera thread only). */
    private inner class Rig(val id: String, val entry: TextureRegistry.SurfaceTextureEntry) {
        val chars: CameraCharacteristics = manager.getCameraCharacteristics(id)
        var device: CameraDevice? = null
        var session: CameraCaptureSession? = null
        var previewSurface: Surface? = null
        var reader: ImageReader? = null
        var previewBuilder: CaptureRequest.Builder? = null
        @Volatile var lastResult: TotalCaptureResult? = null
        var useRaw = false
        var captureSize = Size(0, 0)
        var previewSize = Size(0, 0)
        var ready = false

        val callback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                onPreviewResult(this@Rig, result)
            }
        }

        fun close() {
            runCatching { session?.stopRepeating() }
            runCatching { session?.close() }
            runCatching { device?.close() }
            runCatching { reader?.close() }
            runCatching { previewSurface?.release() }
            session = null; device = null; reader = null; previewSurface = null; previewBuilder = null
            ready = false
            lastResult = null
        }
    }

    private val rigs = LinkedHashMap<String, Rig>()
    @Volatile private var active: Rig? = null

    /** Flutter textures per lens id (main thread only), kept for the controller's lifetime. */
    private val entries = HashMap<String, TextureRegistry.SurfaceTextureEntry>()

    // The active camera, as used by tap-to-focus and capture.
    private val device: CameraDevice? get() = active?.device
    private val session: CameraCaptureSession? get() = active?.session
    private val chars: CameraCharacteristics? get() = active?.chars
    private val reader: ImageReader? get() = active?.reader
    private val previewSurface: Surface? get() = active?.previewSurface
    private val previewBuilder: CaptureRequest.Builder? get() = active?.previewBuilder
    private val lastResult: TotalCaptureResult? get() = active?.lastResult
    private val useRaw: Boolean get() = active?.useRaw ?: false
    private val captureSize: Size get() = active?.captureSize ?: Size(0, 0)
    private var sceneTick = 0L
    private var lensList: List<Lens> = emptyList()

    /** False once the phone refused a second open camera; then lens switches restart the camera. */
    @Volatile private var dualAllowed = true

    // Auto macro: the main lens cannot focus closer than its minimum distance; a lens that focuses
    // much closer (usually the ultrawide) takes over for close-ups.
    @Volatile var autoMacro = true
    private var mainLensId: String? = null
    private var macroLensId: String? = null
    private var mainMinFocus = 0f
    /** The macro lens is on screen because auto macro chose it (so auto macro may also leave it). */
    private var autoEntered = false
    private var closeSince = 0L
    private var farSince = 0L
    private var macroSent: Boolean? = null

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

    /** Lenses that failed to open (hidden by the vendor for third-party apps); never offered again. */
    private val blocked = java.util.Collections.synchronizedSet(mutableSetOf<String>())

    /**
     * Listed camera ids plus ids some vendors hide from [CameraManager.getCameraIdList] (for example
     * the ultrawide on Motorola phones) but still describe. Whether they open is found out on use.
     */
    private val candidateIds: List<String> by lazy {
        val listed = manager.cameraIdList.toList()
        val hidden = (0..15).map { it.toString() }.filter { it !in listed && runCatching { manager.getCameraCharacteristics(it) }.isSuccess }
        listed + hidden
    }

    fun lenses(): List<Lens> {
        val backs = candidateIds.filter { id ->
            if (id in blocked) return@filter false
            val c = manager.getCameraCharacteristics(id)
            val px = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            c.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK &&
                (c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf())
                    .contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE) &&
                c.physicalCameraIds.isEmpty() && // logical multi-cameras duplicate their physical lenses
                (px == null || px.width.toLong() * px.height >= 5_000_000L) // skip depth / 2 MP helper sensors
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

    private fun minFocus(id: String): Float =
        manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f

    /** The lens to use for close-ups: one that focuses at least twice as close as the main lens. */
    private fun pickMacro(lenses: List<Lens>, main: String?): String? {
        main ?: return null
        val mainMin = minFocus(main)
        if (mainMin <= 0f) return null // fixed-focus main lens: nothing to compare
        val best = lenses.filter { it.id != main }.maxByOrNull { minFocus(it.id) } ?: return null
        return best.id.takeIf { minFocus(it) >= mainMin * 2f }
    }

    private fun zoomLabel(z: Double): String {
        // Below 1x one decimal tells 0.5x and 0.6x ultrawides apart; above, half steps are enough.
        val r = if (z < 1.0) Math.round(z * 10) / 10.0 else Math.round(z * 2) / 2.0
        return if (r == r.toLong().toDouble()) "${r.toLong()}x" else "${r}x"
    }

    // ---- open / close -----------------------------------------------------------------------

    /**
     * Opens the requested lens (or the main one) and, when possible, the partner lens for instant
     * switching: macro beside main, or main beside macro. Must be called on the main thread.
     */
    fun start(requestedLens: String?, done: (Result<Map<String, Any?>>) -> Unit) {
        val lenses: List<Lens>
        val ents: Map<String, TextureRegistry.SurfaceTextureEntry>
        val first: String
        val partner: String?
        val mainId: String?
        val macroId: String?
        try {
            lenses = lenses()
            mainId = lenses.minByOrNull { abs(it.zoom - 1.0) }?.id
            macroId = pickMacro(lenses, mainId)
            first = requestedLens?.takeIf { r -> lenses.any { it.id == r } } ?: mainId ?: error("No back camera found")
            partner = when (first) {
                mainId -> macroId
                macroId -> mainId
                else -> null
            }?.takeIf { dualAllowed }
            // Flutter's TextureRegistry is @UiThread: textures are made here, one per lens.
            ents = listOfNotNull(first, partner).associateWith { entries.getOrPut(it) { textures.createSurfaceTexture() } }
        } catch (t: Throwable) {
            done(Result.failure(t))
            return
        }
        handler.post {
            closeAll()
            lensList = lenses
            mainLensId = mainId
            macroLensId = macroId
            mainMinFocus = mainId?.let { minFocus(it) } ?: 0f
            autoEntered = false
            closeSince = 0L; farSince = 0L; macroSent = null
            openRig(first, ents.getValue(first), primary = true) { r ->
                main.post { done(r) }
                if (r.isSuccess && partner != null) openRig(partner, ents.getValue(partner), primary = false) {}
            }
        }
    }

    /**
     * Shows the given lens: instant when it is already running as the standby camera, otherwise a
     * restart on that lens. Main thread.
     */
    fun selectLens(id: String, done: (Result<Map<String, Any?>>) -> Unit) {
        handler.post {
            val rig = rigs[id]
            if (rig != null && rig.ready && !capturing) {
                switchTo(rig, auto = false, notify = false)
                val info = info(rig)
                main.post { done(Result.success(info)) }
            } else {
                main.post { start(id, done) }
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun openRig(id: String, entry: TextureRegistry.SurfaceTextureEntry, primary: Boolean, cb: (Result<Map<String, Any?>>) -> Unit) {
        val rig = try {
            Rig(id, entry)
        } catch (t: Throwable) {
            if (primary) blockIfAuxiliary(id)
            cb(Result.failure(t))
            return
        }
        try {
            val c = rig.chars
            val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
            val rawSizes = map.getOutputSizes(ImageFormat.RAW_SENSOR)
            val cfa = c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT) ?: 4
            rig.useRaw = caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW) && !rawSizes.isNullOrEmpty() && cfa in 0..3
            val cap = if (rig.useRaw) {
                rawSizes.maxByOrNull { it.width.toLong() * it.height }!!
            } else {
                val sizes = map.getOutputSizes(ImageFormat.YUV_420_888).toList()
                sizes.filter { it.width.toLong() * it.height <= 13_000_000L }.maxByOrNull { it.width.toLong() * it.height }
                    ?: sizes.minByOrNull { it.width.toLong() * it.height }!!
            }
            rig.captureSize = Size(cap.width and 1.inv(), cap.height and 1.inv())
            val aspect = rig.captureSize.width.toDouble() / rig.captureSize.height
            val previewSizes = map.getOutputSizes(SurfaceTexture::class.java).toList()
            // The viewfinder only needs screen-width detail: a light preview stream keeps it smooth.
            // Photos never come from it; they use full-resolution RAW (or YUV) bursts.
            rig.previewSize = previewSizes.filter { abs(it.width.toDouble() / it.height - aspect) < 0.02 && it.width <= PREVIEW_MAX }
                .maxByOrNull { it.width * it.height }
                ?: previewSizes.filter { it.width <= PREVIEW_MAX }.maxByOrNull { it.width * it.height } ?: previewSizes.first()
            entry.surfaceTexture().setDefaultBufferSize(rig.previewSize.width, rig.previewSize.height)
            rig.previewSurface = Surface(entry.surfaceTexture())
            // Buffers are only allocated when the reader is used, so the standby camera's costs little.
            val fmt = if (rig.useRaw) ImageFormat.RAW_SENSOR else ImageFormat.YUV_420_888
            rig.reader = ImageReader.newInstance(rig.captureSize.width, rig.captureSize.height, fmt, 5).apply {
                setOnImageAvailableListener({ rd -> onImage(rd) }, handler)
            }
            rigs[id] = rig
            if (primary) active = rig
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    rig.device = camera
                    createSession(rig, cb)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close(); rig.device = null; rig.ready = false
                    if (rig === active) main.post { emit(mapOf("type" to "error", "message" to "Camera disconnected")) }
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    rig.close()
                    rigs.remove(id)
                    if (primary) blockIfAuxiliary(id) else dualAllowed = false
                    cb(Result.failure(IllegalStateException("Camera error $error")))
                }
            }, handler)
        } catch (t: Throwable) {
            rig.close()
            rigs.remove(id)
            if (primary) blockIfAuxiliary(id) else dualAllowed = false
            cb(Result.failure(t))
        }
    }

    /** A hidden or extra lens that fails to open is dropped from the lens list; the main one never is. */
    private fun blockIfAuxiliary(id: String) {
        val main = lenses().minByOrNull { abs(it.zoom - 1.0) }?.id
        if (id != main) blocked.add(id)
    }

    @Suppress("DEPRECATION")
    private fun createSession(rig: Rig, cb: (Result<Map<String, Any?>>) -> Unit) {
        try {
            rig.device!!.createCaptureSession(
                listOf(rig.previewSurface!!, rig.reader!!.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) {
                        rig.session = s
                        rig.ready = true
                        startPreview(rig)
                        cb(Result.success(info(rig)))
                    }

                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        if (rig !== active) {
                            rig.close(); rigs.remove(rig.id); dualAllowed = false
                        }
                        cb(Result.failure(IllegalStateException("Could not configure the camera")))
                    }
                },
                handler,
            )
        } catch (t: Throwable) {
            cb(Result.failure(t))
        }
    }

    private fun info(rig: Rig): Map<String, Any?> = mapOf(
        "textureId" to rig.entry.id(),
        "previewWidth" to rig.previewSize.width,
        "previewHeight" to rig.previewSize.height,
        "sensorOrientation" to (rig.chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90),
        "raw" to rig.useRaw,
        "captureWidth" to rig.captureSize.width,
        "captureHeight" to rig.captureSize.height,
        "flash" to (rig.chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false),
        "lensId" to rig.id,
        "mainLensId" to mainLensId,
        "macroLensId" to macroLensId,
        "lenses" to lensList.map { mapOf("id" to it.id, "label" to it.label, "zoom" to it.zoom) },
    )

    /** (Re)starts the repeating preview of a camera, at full rate on screen and low rate on standby. */
    private fun startPreview(rig: Rig) {
        val cam = rig.device ?: return
        val s = rig.session ?: return
        val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        b.addTarget(rig.previewSurface!!)
        applySubjectFocus(b, rig.chars)
        (if (rig === active) previewFps(rig.chars) else standbyFps(rig.chars))?.let { b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
        b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        b.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        b.set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE_ON)
        rig.previewBuilder = b
        s.setRepeatingRequest(b.build(), rig.callback, handler)
    }

    /** Puts [rig] on screen; the previous camera keeps running as the standby. Camera thread. */
    private fun switchTo(rig: Rig, auto: Boolean, notify: Boolean) {
        val old = active
        if (rig === old) return
        handler.removeCallbacks(restoreFocus)
        active = rig
        autoEntered = auto && rig.id == macroLensId
        closeSince = 0L; farSince = 0L
        runCatching { startPreview(rig) }
        old?.let { runCatching { startPreview(it) } }
        if (notify) {
            val info = info(rig)
            main.post { emit(mapOf("type" to "lens", "info" to info, "macro" to autoEntered)) }
        }
    }

    /**
     * Preview frame-rate range: the fastest top rate (up to 30), with the floor nearest 24 fps. The
     * default lets AE drop to 10 fps in dim light, which makes the viewfinder feel sluggish; the burst
     * picks its own exposure times, so this only affects the preview.
     */
    private fun previewFps(c: CameraCharacteristics): android.util.Range<Int>? {
        val ranges = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: return null
        val top = ranges.filter { it.upper <= 30 }.maxOfOrNull { it.upper } ?: return null
        return ranges.filter { it.upper == top }.minByOrNull { abs(it.lower - 24) }
    }

    /** Standby rate: the slowest fixed range of at least 10 fps, enough for autofocus to follow. */
    private fun standbyFps(c: CameraCharacteristics): android.util.Range<Int>? {
        val ranges = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: return null
        return ranges.filter { it.lower == it.upper && it.upper >= 10 }.minByOrNull { it.upper } ?: previewFps(c)
    }

    /**
     * Automatic subject focus: continuous AF, and where the camera supports it, face detection with
     * the face-priority scene mode so AF, AE and AWB follow the faces in frame (falls back to the
     * whole scene when there are none).
     */
    private fun applySubjectFocus(b: CaptureRequest.Builder, c: CameraCharacteristics?) {
        val faceModes = c?.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES) ?: intArrayOf()
        val faceMode = when {
            faceModes.contains(CameraMetadata.STATISTICS_FACE_DETECT_MODE_SIMPLE) -> CameraMetadata.STATISTICS_FACE_DETECT_MODE_SIMPLE
            faceModes.contains(CameraMetadata.STATISTICS_FACE_DETECT_MODE_FULL) -> CameraMetadata.STATISTICS_FACE_DETECT_MODE_FULL
            else -> null
        }
        val scenes = c?.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES) ?: intArrayOf()
        val facePriority = faceMode != null && scenes.contains(CameraMetadata.CONTROL_SCENE_MODE_FACE_PRIORITY)
        if (facePriority) {
            b.set(CaptureRequest.STATISTICS_FACE_DETECT_MODE, faceMode)
            b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_USE_SCENE_MODE)
            b.set(CaptureRequest.CONTROL_SCENE_MODE, CameraMetadata.CONTROL_SCENE_MODE_FACE_PRIORITY)
        } else {
            b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        }
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        b.set(CaptureRequest.CONTROL_AF_REGIONS, null)
        b.set(CaptureRequest.CONTROL_AE_REGIONS, null)
    }

    private val restoreFocus = Runnable {
        val r = active ?: return@Runnable
        val bb = r.previewBuilder ?: return@Runnable
        runCatching {
            applySubjectFocus(bb, r.chars)
            r.session?.setRepeatingRequest(bb.build(), r.callback, handler)
        }
    }

    private fun onPreviewResult(rig: Rig, result: TotalCaptureResult) {
        rig.lastResult = result
        val now = android.os.SystemClock.elapsedRealtime()
        checkMacro(now)
        if (rig !== active || now - sceneTick < 400) return
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

    private fun focused(r: TotalCaptureResult?): Boolean {
        val af = r?.get(CaptureResult.CONTROL_AF_STATE) ?: return false
        return af == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED || af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED
    }

    /**
     * Auto macro, run on every preview frame of either camera. Main to macro when the main lens sits
     * at its closest focus without locking and (with both cameras running) the macro lens has locked
     * focus on something close. Macro to main once the macro lens focuses beyond ~17 cm and the main
     * lens has focus. With both cameras running the switch is instant, so the dwell is short; with one
     * camera a `macro` event asks the app to restart on the other lens.
     */
    private fun checkMacro(now: Long) {
        if (!autoMacro || capturing) return
        val macroId = macroLensId ?: return
        val mainId = mainLensId ?: return
        val act = active ?: return
        val mainRig = rigs[mainId]
        val macroRig = rigs[macroId]
        val dual = mainRig?.ready == true && macroRig?.ready == true
        val hunting = setOf(
            CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED,
            CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN,
            CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED,
        )
        when (act.id) {
            mainId -> {
                val r = mainRig?.lastResult ?: return
                val fd = r.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: return
                val af = r.get(CaptureResult.CONTROL_AF_STATE) ?: return
                val tooClose = mainMinFocus > 0f && fd >= mainMinFocus * 0.9f && af in hunting
                val macroSharp = !dual || (focused(macroRig?.lastResult) && (macroRig?.lastResult?.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f) >= 6f)
                closeSince = if (!(tooClose && macroSharp)) 0L else if (closeSince == 0L) now else closeSince
                if (closeSince != 0L && now - closeSince >= (if (dual) 250 else 600)) {
                    closeSince = 0L
                    if (dual) switchTo(macroRig!!, auto = true, notify = true) else sendMacro(true, macroId)
                }
            }
            macroId -> {
                if (dual && !autoEntered) return // the user chose this lens
                val r = macroRig?.lastResult ?: return
                val fd = r.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: return
                val far = fd < 6f // diopters: 0 is infinity, 6 is about 17 cm
                val mainOk = !dual || focused(mainRig?.lastResult)
                farSince = if (!(far && mainOk)) 0L else if (farSince == 0L) now else farSince
                if (farSince != 0L && now - farSince >= (if (dual) 400 else 800)) {
                    farSince = 0L
                    if (dual) switchTo(mainRig!!, auto = true, notify = true) else sendMacro(false, mainId)
                }
            }
        }
    }

    private fun sendMacro(on: Boolean, target: String) {
        if (macroSent == on) return
        macroSent = on
        main.post { emit(mapOf("type" to "macro", "on" to on, "lensId" to target)) }
    }

    fun stop(done: () -> Unit) {
        handler.post {
            closeAll()
            main.post { done() }
        }
    }

    private fun closeAll() {
        handler.removeCallbacks(restoreFocus)
        rigs.values.forEach { it.close() }
        rigs.clear()
        active = null
        burst = null
        capturing = false
    }

    fun release() {
        stop {
            // Runs on the main thread, as the TextureRegistry requires.
            entries.values.forEach { runCatching { it.release() } }
            entries.clear()
        }
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
            handler.removeCallbacks(restoreFocus)
            runCatching {
                // Manual point: leave face priority so the tapped region wins.
                b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                b.set(CaptureRequest.CONTROL_AF_REGIONS, region)
                b.set(CaptureRequest.CONTROL_AE_REGIONS, region)
                b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                // `active` here is the sensor's active array; the camera on screen is this@CameraController.active.
                s.setRepeatingRequest(b.build(), this@CameraController.active!!.callback, handler)
                b.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                s.capture(b.build(), null, handler)
                b.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_IDLE)
            }
            handler.postDelayed(restoreFocus, 3500)
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
                        // The standby camera pauses during the burst so all bandwidth goes to the shot.
                        rigs.values.filter { it !== active }.forEach { r -> runCatching { r.session?.stopRepeating() } }
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
        active?.let { a -> runCatching { startPreview(a) } }
        rigs.values.filter { it !== active && it.ready }.forEach { r -> runCatching { startPreview(r) } }
    }

    private fun fail(done: (Result<Map<String, Any?>>) -> Unit, msg: String) {
        main.post { done(Result.failure(IllegalStateException(msg))) }
    }

    companion object {
        /** Direct memory for one burst, before the processor budget. */
        const val BURST_BYTES = 420L * 1024 * 1024

        /** Longest preview edge in pixels (1440 x 1080 for 4:3). */
        const val PREVIEW_MAX = 1440
    }
}
