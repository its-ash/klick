# Klick

A fully automatic camera that shoots RAW bursts and merges them into one clean photo.

**[Website](https://klick.itsash.in)** · **[Download APK](https://github.com/its-ash/klick/releases/latest/download/klick.apk)** · Get it in **[Ash Store](https://store.itsash.in)**

## Features

- **Scene aware bursts**: bright light gets 3 or 4 frames, low light gets 8 to 12. A badge shows Night, HDR or Normal and the frame count.
- **Highlight protection**: frames are exposed a little short on purpose; the engine restores brightness from the merged result.
- **Shake limited**: the gyroscope caps each exposure time, so a shaky hand gets shorter frames instead of blur.
- **Native Rust engine**: alignment, noise-weighted merge, RAW development, exposure fusion, denoise, sharpen, JPEG.
- **Keeps the RAW**: the reference frame is saved as a DNG (switch off in Settings).
- **Keep shooting**: photos process in the background with a visible queue.
- **Gallery**: before/after toggle (reference frame vs processed), share, delete.
- **Private**: no accounts, no ads. Location tagging is off by default. The only network call is the update check in Settings.

## Install

- **Ash Store**: install [Ash Store](https://store.itsash.in) once, then install and update Klick from it.
- **APK**: download the [latest APK](https://github.com/its-ash/klick/releases/latest/download/klick.apk) on your phone and open it. Allow your browser to install unknown apps if Android asks.

Requires Android 10 (API 29). RAW capture needs a back camera that exposes RAW to apps; phones without it use a full-resolution YUV burst through the same engine.

## Develop

Needs Flutter ≥ 3.44, the shared theme package at `../theme`, Rust with the Android targets and `cargo-ndk` (NDK 28).

```bash
make run      # build the Rust engine for Android, then run on a device
make build    # cargo-ndk (arm64-v8a, armeabi-v7a, x86_64) + release APK
make test     # cargo test + flutter analyze + flutter test
make deploy   # bump version, build, tag, push, GitHub release (see store.mk)
```

The engine `.so` files are built into `android/app/src/main/jniLibs/` by `make build` and are not committed.

## Release

`make deploy` follows the Ash Store kit (`store.mk`, `store.json`): it bumps the
version in `pubspec.yaml`, builds, commits, tags `vX.Y.Z`, pushes, and publishes
a GitHub release with `klick.apk`. Ash Store and the website pick up the new
version from that release.

## Project layout

```
lib/
├── main.dart                    # UiApp + camera screen
├── core/                        # pure, tested logic
│   ├── scene.dart               # scene / frame count / exposure rules (mirror of ScenePlanner.kt)
│   ├── queue.dart               # processing queue state
│   └── photo.dart
├── platform/camera_channel.dart # CameraApi + MethodChannel/EventChannel
├── settings/settings_store.dart
└── screens/                     # camera, gallery, viewer, settings
android/app/src/main/kotlin/com/itsash/klick/
├── CameraController.kt          # Camera2: preview texture, AF/AE, RAW/YUV bursts
├── ScenePlanner.kt              # scene decision rules
├── ShakeMonitor.kt              # gyroscope RMS, device rotation
├── EngineMeta.kt                # CameraCharacteristics/CaptureResult -> engine metadata
├── PhotoProcessor.kt            # background queue, DNG, EXIF, MediaStore
├── NativeEngine.kt              # JNI declarations
└── KlickBridge.kt, MainActivity.kt
rust/                            # crate klick-engine (cdylib)
└── src/                         # align, merge, raw, demosaic, color, tone, denoise, sharpen, encode, pipeline, jni_api
```

## License

No license file is included yet.

Built by [itsash.in](https://itsash.in).
