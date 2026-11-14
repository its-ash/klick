package com.itsash.klick

import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.DngCreator
import android.hardware.camera2.TotalCaptureResult
import android.location.Location
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Size
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/** One captured burst waiting for the engine. Buffers are direct and owned by the job. */
class BurstJob(
    val id: Long,
    val raw: Boolean,
    val width: Int,
    val height: Int,
    /** RAW: one buffer per frame. YUV: Y, U, V buffers per frame. */
    val buffers: List<ByteBuffer>,
    val frameCount: Int,
    val rowStride: Int,
    /** YUV only: [yRow, uvRow, uvPixel]. */
    val yuvStrides: IntArray,
    val results: List<TotalCaptureResult>,
    val characteristics: CameraCharacteristics,
    val plan: ScenePlanner.Plan,
    val jpegOrientation: Int,
    val quality: Int,
    val keepDng: Boolean,
    val location: Location?,
    val aperture: Float,
    val focalLength: Float,
) {
    val bytes: Long get() = buffers.sumOf { it.capacity().toLong() }
}

/** Background queue: one burst is processed at a time; more can be queued within a memory budget. */
class PhotoProcessor(private val context: Context, private val emit: (Map<String, Any?>) -> Unit) {
    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "klick-engine").apply { priority = Thread.NORM_PRIORITY - 1 } }
    private val inFlight = AtomicLong(0)
    private val ids = AtomicLong(1)

    fun nextId(): Long = ids.getAndIncrement()

    /** Keeps total pending RAM bounded: always accept the first burst, then up to the budget. */
    fun canAccept(bytes: Long): Boolean = inFlight.get() == 0L || inFlight.get() + bytes <= BUDGET_BYTES

    fun pending(): Int = pendingCount.get()
    private val pendingCount = java.util.concurrent.atomic.AtomicInteger(0)

    fun submit(job: BurstJob) {
        inFlight.addAndGet(job.bytes)
        pendingCount.incrementAndGet()
        emit(event(job, "queued"))
        executor.execute {
            emit(event(job, "processing"))
            try {
                val out = run(job)
                emit(event(job, "done") + out)
            } catch (t: Throwable) {
                emit(event(job, "failed") + mapOf("message" to (t.message ?: t.javaClass.simpleName)))
            } finally {
                inFlight.addAndGet(-job.bytes)
                pendingCount.decrementAndGet()
            }
        }
    }

    private fun event(job: BurstJob, state: String) = mapOf<String, Any?>(
        "type" to "job", "id" to job.id, "state" to state, "scene" to job.plan.scene.label, "frames" to job.frameCount,
        "pending" to pendingCount.get(),
    )

    private fun run(job: BurstJob): Map<String, Any?> {
        if (!NativeEngine.available) error("The processing engine (libklick_engine.so) is not installed in this build")
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val base = "KLICK_${stamp}_${job.id}"
        val tmp = File(context.cacheDir, "$base.jpg")
        val refDir = File(context.filesDir, "refs").apply { mkdirs() }
        val refFile = File(refDir, "$base.jpg")

        val status: NativeEngine.Status
        var refIndex = 0
        if (job.raw) {
            val built = EngineMeta.rawMeta(job.characteristics, job.results[0], job.plan)
            status = NativeEngine.parse(
                NativeEngine.processRaw(job.buffers.toTypedArray(), job.width, job.height, job.rowStride, built.meta, built.lsc, tmp.path, job.quality),
            )
            if (!status.ok) error(status.message)
            refIndex = status.refIndex.coerceIn(0, job.frameCount - 1)
            val refMeta = EngineMeta.rawMeta(job.characteristics, job.results[refIndex], job.plan)
            NativeEngine.referenceRaw(arrayOf(job.buffers[refIndex]), job.width, job.height, job.rowStride, refMeta.meta, refMeta.lsc, refFile.path, 80)
        } else {
            val meta = EngineMeta.yuvMeta(job.results.firstOrNull(), job.plan)
            status = NativeEngine.parse(
                NativeEngine.processYuv(job.buffers.toTypedArray(), job.width, job.height, job.yuvStrides, meta, tmp.path, job.quality),
            )
            if (!status.ok) error(status.message)
            refIndex = status.refIndex.coerceIn(0, job.frameCount - 1)
            NativeEngine.referenceYuv(job.buffers.subList(refIndex * 3, refIndex * 3 + 3).toTypedArray(), job.width, job.height, job.yuvStrides, refFile.path, 80)
        }
        // The "before" render must turn the same way as the photo in the compare slider.
        tagOrientation(refFile, job.jpegOrientation)

        writeExif(tmp, job, refIndex, status)
        val uri = insert("$base.jpg", "image/jpeg") { out -> tmp.inputStream().use { it.copyTo(out) } }
        tmp.delete()

        var dngUri: Uri? = null
        if (job.raw && job.keepDng) {
            dngUri = runCatching {
                insert("$base.dng", "image/x-adobe-dng") { out -> writeDng(out, job, refIndex) }
            }.getOrNull()
        }
        return mapOf(
            "uri" to uri.toString(), "name" to "$base.jpg", "dng" to (dngUri != null),
            "ref" to refFile.path, "width" to status.width, "height" to status.height,
        )
    }

    private fun writeExif(file: File, job: BurstJob, refIndex: Int, status: NativeEngine.Status) {
        val ex = ExifInterface(file)
        val orientation = exifOrientation(job.jpegOrientation)
        val res = job.results.getOrNull(refIndex)
        val iso = res?.get(android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY) ?: job.plan.iso
        val expNs = res?.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME) ?: job.plan.exposureNs
        ex.setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString())
        ex.setAttribute(ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY, iso.toString())
        ex.setAttribute(ExifInterface.TAG_EXPOSURE_TIME, (expNs / 1e9).toString())
        ex.setAttribute(ExifInterface.TAG_F_NUMBER, job.aperture.toString())
        ex.setAttribute(ExifInterface.TAG_FOCAL_LENGTH, "${(job.focalLength * 1000).toInt()}/1000")
        val now = SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US).format(Date())
        ex.setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, now)
        ex.setAttribute(ExifInterface.TAG_DATETIME, now)
        ex.setAttribute(ExifInterface.TAG_MAKE, android.os.Build.MANUFACTURER)
        ex.setAttribute(ExifInterface.TAG_MODEL, android.os.Build.MODEL)
        ex.setAttribute(ExifInterface.TAG_SOFTWARE, "Klick ${job.plan.scene.label}, ${status.frames} frames")
        job.location?.let { ex.setGpsInfo(it) }
        ex.saveAttributes()
    }

    private fun writeDng(out: java.io.OutputStream, job: BurstJob, refIndex: Int) {
        val dng = DngCreator(job.characteristics, job.results[refIndex])
        dng.setOrientation(exifOrientation(job.jpegOrientation))
        job.location?.let { dng.setLocation(it) }
        val src = job.buffers[refIndex].duplicate().apply { rewind() }
        val tight = job.width * 2
        val pixels: ByteBuffer = if (job.rowStride == tight) src else {
            // DngCreator expects tightly packed rows.
            val packed = ByteBuffer.allocateDirect(tight * job.height)
            val row = ByteArray(tight)
            for (y in 0 until job.height) {
                src.position(y * job.rowStride)
                src.get(row)
                packed.put(row)
            }
            packed.rewind()
            packed
        }
        dng.writeByteBuffer(out, Size(job.width, job.height), pixels, 0)
        dng.close()
    }

    private fun insert(name: String, mime: String, write: (java.io.OutputStream) -> Unit): Uri {
        val resolver: ContentResolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, mime)
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Klick")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("MediaStore refused $name")
        try {
            resolver.openOutputStream(uri)!!.use { write(it) }
            resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
        } catch (t: Throwable) {
            resolver.delete(uri, null, null)
            throw t
        }
        return uri
    }

    companion object {
        /** Direct memory budget for bursts waiting or being processed. */
        const val BUDGET_BYTES = 640L * 1024 * 1024

        /** EXIF orientation tag for a clockwise rotation in degrees. */
        fun exifOrientation(degrees: Int): Int = when (((degrees % 360) + 360) % 360) {
            90 -> ExifInterface.ORIENTATION_ROTATE_90
            180 -> ExifInterface.ORIENTATION_ROTATE_180
            270 -> ExifInterface.ORIENTATION_ROTATE_270
            else -> ExifInterface.ORIENTATION_NORMAL
        }

        /** Tags a JPEG so viewers turn it the way the phone was held when the photo was taken. */
        fun tagOrientation(file: File, degrees: Int) {
            runCatching {
                val ex = ExifInterface(file)
                ex.setAttribute(ExifInterface.TAG_ORIENTATION, exifOrientation(degrees).toString())
                ex.saveAttributes()
            }
        }
    }
}
