package com.itsash.klick

import android.os.Bundle
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private var bridge: KlickBridge? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A camera must not dim or sleep while it is open; the flag only applies while visible.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        bridge = KlickBridge(this, flutterEngine)
    }

    override fun onResume() {
        super.onResume()
        bridge?.onResume()
    }

    override fun onPause() {
        bridge?.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        bridge?.dispose()
        bridge = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        bridge?.onPermissionResult(requestCode)
    }
}
