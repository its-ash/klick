STORE_ID := klick

# The Rust engine is not committed as binaries: every build compiles it for the three ABIs
# Flutter packages (arm64-v8a, armeabi-v7a, x86_64) into android/app/src/main/jniLibs.
JNILIBS := ../android/app/src/main/jniLibs
RUST_ANDROID := cd rust && cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -o $(JNILIBS) build --release

.PHONY: run build test clean rust-android

rust-android:
	$(RUST_ANDROID)

run: rust-android
	flutter pub get && flutter run

build: rust-android
	flutter pub get && flutter build apk --release

test:
	cd rust && cargo test
	flutter analyze
	flutter test

clean:
	flutter clean
	cd rust && cargo clean
	rm -rf dist android/app/src/main/jniLibs

# deploy, release, bump, store-check, store-meta, version
include store.mk
