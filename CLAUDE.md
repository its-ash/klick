# Klick (package: klick / com.itsash.klick)

Fully automatic camera. On shutter it captures a burst of RAW_SENSOR frames (YUV_420_888 when the
back camera has no RAW), chosen automatically from the scene, and a native Rust engine merges them
into one JPEG. The Flutter UI is a viewfinder, a gallery with before/after, and Settings.

## Pipeline

1. **Scene plan** (`ScenePlanner.kt`, mirrored in `lib/core/scene.dart`; change both together).
   Scene EV = log2(N^2/t) - log2(ISO/100) from the preview metering. EV < 5 Night (8 to 12 frames,
   half exposure, steady phone may expose up to 250 ms); EV >= 12 HDR (4 frames, half exposure);
   otherwise Normal (3 frames from EV 9, else 5, 0.8 or 0.7 exposure). The gyroscope RMS caps the
   per-frame exposure (steady 125 ms, moderate 33 ms, shaky 16 ms) and ISO is raised (up to the
   analog maximum) to compensate. `evGain` is the digital gain that restores the under-exposure.
   Auto flash (EV < 2.5, flash available) shoots a single flash frame.
2. **Capture** (`CameraController.kt`, Camera2): preview into a Flutter `Texture`, continuous
   AF/AE/AWB, tap to focus. Bursts use manual exposure/ISO, NR and edge off for RAW, lens shading
   map on. Frames are copied into direct ByteBuffers (memory budget in `PhotoProcessor`).
3. **Engine** (`rust/`, crate `klick-engine`, JNI in `jni_api.rs`, called by `NativeEngine.kt`):
   - `align.rs`: Gaussian pyramid of the 2x2-binned grey, coarse-to-fine 16 px tiles (L2 coarse,
     L1 fine), whole-pixel offsets in the binned grid (so always CFA-aligned). Reference = sharpest of
     the first three frames.
   - `merge.rs`: tile merge with a Wiener-like weight from the noise profile (difference-to-noise),
     tile and pixel ghost rejection, Hann-window tiles at 50% overlap.
   - `raw.rs`, `demosaic.rs`, `color.rs`: black/white level, lens shading map, as-shot white balance
     with a small grey-world refinement, Malvar-He-Cutler demosaic (sensors over 17 MP are 2x2
     quad-binned instead), DNG-style forward matrix interpolation to linear sRGB.
   - `tone.rs`: histogram-driven auto exposure and HDR plan; Mertens exposure fusion on a luma
     thumbnail, applied to the full image as a smooth gain map; highlight roll-off.
   - `denoise.rs`, `sharpen.rs`: bilateral luma, cross-bilateral chroma, both scaled by the noise
     model, ISO and gain; unsharp with coring and halo limit, local contrast, vibrance.
   - `encode.rs`: JPEG (`jpeg-encoder`). `finish.rs` and `pipeline.rs` compose the stages; every stage
     is a pure function over plain buffers with host unit tests.
4. **Save** (`PhotoProcessor.kt`): EXIF via `androidx.exifinterface`, MediaStore `Pictures/Klick`,
   optional DNG of the reference frame (`DngCreator`), a half-size render of the reference frame in
   `files/refs/` for before/after.

Metadata passes as one float array: layout in `rust/src/meta.rs`, built by `EngineMeta.kt`.

## Code layout

- `lib/core/`: pure Dart with tests: `scene.dart`, `queue.dart` (processing queue state machine
  fed by native `job` events), `photo.dart`.
- `lib/platform/camera_channel.dart`: `CameraApi` (faked in tests) over MethodChannel
  `com.itsash.klick/camera` and EventChannel `/events` (`scene`, `job`, `error`).
- `lib/screens/`: `camera_screen` (dark viewfinder; the only screen with hardcoded colors),
  `gallery_screen`, `viewer_screen`, `settings_screen` (ends with `UiAppearanceSection`,
  `UiUpdateSection(storeId: 'klick')`, `UiDeveloperSection()`).
- `lib/settings/settings_store.dart`: keep DNG (on), JPEG quality (95), shutter sound, grid,
  location tagging (off).
- `android/app/src/main/kotlin/com/itsash/klick/`: see README layout.
- `rust/`: `cargo test` runs on the host.

## Commands

- `make build`: `cargo ndk` for arm64-v8a, armeabi-v7a, x86_64 into `android/app/src/main/jniLibs/`
  (git-ignored, rebuilt every time), then `flutter build apk --release`. `make run`, `make test`
  (cargo test + flutter analyze + flutter test), `make clean`.
- Icon: SVG sources in `assets/icon/` (a pixel camera with an aperture) -> `rsvg-convert -w 1024 -h 1024`
  to `*-1024.png`, then `dart run flutter_launcher_icons`.

## Conventions

- minSdk 29 (scoped storage via MediaStore `RELATIVE_PATH`, no `WRITE_EXTERNAL_STORAGE`). UI only from
  the shared `theme` package (`../theme`, `../theme/COMPONENTS.md`) except the viewfinder. Starts in light mode.
- Permissions: `CAMERA`, `INTERNET` (update check), `ACCESS_FINE/COARSE_LOCATION` (only requested when
  location tagging is switched on). Portrait only.
- Engine missing (`libklick_engine.so` not packaged) shows a warning pill and jobs fail with a message.

## Release (Ash Store)

- `store.json` (id `klick`, repo its-ash/klick), `store.mk`, `index.html` + `CNAME`
  (`klick.itsash.in`). See `../ash-store/store-kit/STORE.md`.
- `make deploy` = bump version -> build -> commit, tag `vX.Y.Z`, push -> GitHub release with
  `klick.apk` -> topics, homepage, Pages. Never create releases, tags or asset names by hand.
- Do not edit `store.mk` here. Edit `../ash-store/store-kit/store.mk`, then run
  `make store-sync` at the workspace root.
