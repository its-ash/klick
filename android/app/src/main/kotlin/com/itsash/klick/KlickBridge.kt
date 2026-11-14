package com.itsash.klick

import android.Manifest
import android.app.Activity
import android.content.ContentUris
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.location.Location
import android.location.LocationManager
import android.media.MediaActionSound
import android.net.Uri
import android.provider.MediaStore
import android.util.Size
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/** MethodChannel `com.itsash.klick/camera` and EventChannel `com.itsash.klick/events`. */
class KlickBridge(private val activity: Activity, engine: FlutterEngine) : MethodChannel.MethodCallHandler, EventChannel.StreamHandler {
    private val main = android.os.Handler(android.os.Looper.getMainLooper())
    private var sink: EventChannel.EventSink? = null
    private val shake = ShakeMonitor(activity)
    private val processor = PhotoProcessor(activity) { e -> main.post { sink?.success(e) } }
    private val camera = CameraController(activity, engine.renderer, processor, shake) { e -> sink?.success(e) }
    private val io = Executors.newFixedThreadPool(2)
    private val sound by lazy { MediaActionSound().apply { load(MediaActionSound.SHUTTER_CLICK) } }
    private var permissionResult: MethodChannel.Result? = null
    private var permissionWantsLocation = false

    init {
        MethodChannel(engine.dartExecutor.binaryMessenger, "com.itsash.klick/camera").setMethodCallHandler(this)
        EventChannel(engine.dartExecutor.binaryMessenger, "com.itsash.klick/events").setStreamHandler(this)
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { sink = events }
    override fun onCancel(arguments: Any?) { sink = null }

    fun dispose() {
        shake.stop()
        camera.release()
    }

    fun onResume() = shake.start()
    fun onPause() = shake.stop()

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "engine" -> result.success(
                mapOf("available" to NativeEngine.available, "version" to if (NativeEngine.available) NativeEngine.version() else ""),
            )
            "permission" -> requestPermissions(call.argument<Boolean>("location") == true, result)
            "start" -> camera.start(call.argument<String>("lensId")) { r -> reply(result, r) }
            "stop" -> camera.stop { result.success(null) }
            "tapFocus" -> {
                camera.tapFocus(call.argument<Double>("x") ?: 0.5, call.argument<Double>("y") ?: 0.5)
                result.success(null)
            }
            "capture" -> {
                if (call.argument<Boolean>("sound") == true) runCatching { sound.play(MediaActionSound.SHUTTER_CLICK) }
                val loc = if (call.argument<Boolean>("location") == true) lastLocation() else null
                val opts = CameraController.Options(
                    call.argument<String>("flash") ?: "off",
                    call.argument<Int>("quality") ?: 95,
                    call.argument<Boolean>("keepDng") ?: true,
                    loc,
                )
                camera.capture(opts) { r -> reply(result, r) }
            }
            "listPhotos" -> io.execute { runCatching { listPhotos() }.fold({ main.post { result.success(it) } }, { e -> main.post { result.error("list", e.message, null) } }) }
            "loadImage" -> {
                val uri = Uri.parse(call.argument<String>("uri"))
                val side = call.argument<Int>("maxSide") ?: 512
                io.execute {
                    runCatching { loadImage(uri, side) }.fold({ b -> main.post { result.success(b) } }, { e -> main.post { result.error("load", e.message, null) } })
                }
            }
            "share" -> {
                val uri = Uri.parse(call.argument<String>("uri"))
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "image/jpeg"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                activity.startActivity(Intent.createChooser(send, "Share photo"))
                result.success(null)
            }
            "delete" -> io.execute {
                val uri = Uri.parse(call.argument<String>("uri"))
                val ok = runCatching { activity.contentResolver.delete(uri, null, null) > 0 }.getOrDefault(false)
                main.post { result.success(ok) }
            }
            else -> result.notImplemented()
        }
    }

    private fun reply(result: MethodChannel.Result, r: Result<Map<String, Any?>>) {
        r.fold({ result.success(it) }, { result.error("camera", it.message ?: "Camera error", null) })
    }

    // ---- permissions ------------------------------------------------------------------------

    private fun requestPermissions(withLocation: Boolean, result: MethodChannel.Result) {
        val needed = mutableListOf(Manifest.permission.CAMERA)
        if (withLocation) needed += Manifest.permission.ACCESS_FINE_LOCATION
        val missing = needed.filter { ContextCompat.checkSelfPermission(activity, it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) {
            result.success(true)
            return
        }
        if (permissionResult != null) {
            result.error("busy", "Permission request in progress", null)
            return
        }
        permissionResult = result
        permissionWantsLocation = withLocation
        ActivityCompat.requestPermissions(activity, missing.toTypedArray(), REQUEST_CODE)
    }

    fun onPermissionResult(requestCode: Int) {
        if (requestCode != REQUEST_CODE) return
        val cam = ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        permissionResult?.success(cam)
        permissionResult = null
    }

    @Suppress("MissingPermission")
    private fun lastLocation(): Location? {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) return null
        val lm = activity.getSystemService(Activity.LOCATION_SERVICE) as LocationManager
        return lm.getProviders(true).mapNotNull { runCatching { lm.getLastKnownLocation(it) }.getOrNull() }.maxByOrNull { it.time }
    }

    // ---- gallery ----------------------------------------------------------------------------

    private fun listPhotos(): List<Map<String, Any?>> {
        val out = ArrayList<Map<String, Any?>>()
        val refs = java.io.File(activity.filesDir, "refs")
        val proj = arrayOf(
            MediaStore.Images.Media._ID, MediaStore.Images.Media.DISPLAY_NAME, MediaStore.Images.Media.DATE_ADDED,
            MediaStore.Images.Media.WIDTH, MediaStore.Images.Media.HEIGHT,
        )
        val sel = "${MediaStore.Images.Media.RELATIVE_PATH} LIKE ? AND ${MediaStore.Images.Media.MIME_TYPE} = ?"
        val args = arrayOf("Pictures/Klick%", "image/jpeg")
        activity.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, proj, sel, args, "${MediaStore.Images.Media.DATE_ADDED} DESC",
        )?.use { c ->
            while (c.moveToNext()) {
                val id = c.getLong(0)
                val name = c.getString(1)
                val ref = java.io.File(refs, name)
                out += mapOf(
                    "uri" to ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id).toString(),
                    "name" to name,
                    "added" to c.getLong(2) * 1000,
                    "width" to c.getInt(3),
                    "height" to c.getInt(4),
                    "ref" to if (ref.exists()) ref.path else null,
                )
            }
        }
        return out
    }

    private fun loadImage(uri: Uri, maxSide: Int): ByteArray {
        val bmp: Bitmap = if (maxSide <= 640) {
            activity.contentResolver.loadThumbnail(uri, Size(maxSide, maxSide), null)
        } else {
            val src = ImageDecoder.createSource(activity.contentResolver, uri)
            ImageDecoder.decodeBitmap(src) { dec, info, _ ->
                val scale = maxSide.toFloat() / maxOf(info.size.width, info.size.height)
                if (scale < 1f) dec.setTargetSize((info.size.width * scale).toInt(), (info.size.height * scale).toInt())
                dec.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        }
        val out = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
        bmp.recycle()
        return out.toByteArray()
    }

    companion object {
        const val REQUEST_CODE = 4107
    }
}
