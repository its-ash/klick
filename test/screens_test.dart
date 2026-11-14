import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:klick/core/photo.dart';
import 'package:klick/main.dart';
import 'package:klick/platform/camera_channel.dart';
import 'package:klick/screens/gallery_screen.dart';
import 'package:klick/screens/settings_screen.dart';
import 'package:klick/screens/viewer_screen.dart';
import 'package:klick/settings/settings_store.dart';
import 'package:theme/theme.dart';

Map<String, dynamic> _infoMap(String lensId) => {
  'textureId': lensId == '2' ? 2 : 1,
  'previewWidth': 4,
  'previewHeight': 3,
  'sensorOrientation': 90,
  'raw': true,
  'flash': true,
  'lensId': lensId,
  'lenses': [
    {'id': '2', 'label': '0.5x', 'zoom': 0.5},
    {'id': '0', 'label': '1x', 'zoom': 1.0},
  ],
};

class FakeCamera implements CameraApi {
  FakeCamera({this.permissionGranted = true, this.photos = const []});

  final bool permissionGranted;
  final List<Photo> photos;
  final controller = StreamController<Map<dynamic, dynamic>>.broadcast();
  final captures = <Map<String, Object?>>[];
  int stops = 0;

  @override
  Future<EngineInfo> engine() async => const EngineInfo(available: true, version: '0.1.0');

  @override
  Future<bool> permission({bool location = false}) async => permissionGranted;

  @override
  Future<CameraInfo> start({String? lensId, bool autoMacro = true}) async => info(lensId ?? '0');

  final selected = <String>[];

  @override
  Future<CameraInfo> selectLens(String lensId) async {
    selected.add(lensId);
    return info(lensId);
  }

  bool? autoMacro;

  /// Report a main and a macro lens (the phone has a close-focusing ultrawide).
  bool macro = false;

  @override
  Future<void> setAutoMacro(bool on) async => autoMacro = on;

  CameraInfo info(String lensId) => CameraInfo(
    mainLensId: macro ? '0' : null,
    macroLensId: macro ? '2' : null,
    textureId: lensId == '2' ? 2 : 1,
    previewWidth: 4,
    previewHeight: 3,
    sensorOrientation: 90,
    raw: true,
    flash: true,
    lensId: lensId,
    lenses: const [
      Lens(id: '2', label: '0.5x', zoom: 0.5),
      Lens(id: '0', label: '1x', zoom: 1),
    ],
  );

  @override
  Future<void> stop() async => stops++;

  @override
  Future<void> tapFocus(double x, double y) async {}

  @override
  Future<CaptureAck> capture({
    required String flash,
    required int quality,
    required bool keepDng,
    required bool sound,
    required bool location,
  }) async {
    captures.add({'flash': flash, 'quality': quality, 'keepDng': keepDng});
    return const CaptureAck(id: 1, frames: 4, scene: 'normal');
  }

  @override
  Stream<Map<dynamic, dynamic>> events() => controller.stream;

  @override
  Future<List<Photo>> listPhotos() async => photos;

  @override
  Future<Uint8List> loadImage(String uri, int maxSide) async => throw UnimplementedError('no image in tests');

  @override
  Future<void> share(String uri) async {}

  @override
  Future<bool> delete(String uri) async => true;
}

Photo photo(String name, {String? ref}) => Photo(uri: 'content://x/$name', name: name, added: DateTime(2026, 1, 1), refPath: ref);

void main() {
  setUpAll(() => UiTypography.useGoogleFonts = false);

  testWidgets('viewfinder shows the shutter, lenses and the live scene badge', (tester) async {
    final cam = FakeCamera();
    await tester.pumpWidget(KlickApp(settings: SettingsStore.memory(), camera: cam));
    await tester.pump();
    await tester.pump();
    expect(find.bySemanticsLabel('Take photo'), findsOneWidget);
    expect(find.text('0.5x'), findsOneWidget);
    expect(find.text('1x'), findsOneWidget);
    cam.controller.add({'type': 'scene', 'iso': 1600, 'exposureNs': 100000000, 'aperture': 1.8, 'shakeRms': 0.0, 'maxIso': 3200});
    await tester.pump();
    expect(find.text('Night · 12 frames'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('shutter sends the settings to the camera', (tester) async {
    final cam = FakeCamera();
    final settings = SettingsStore.memory()
      ..setJpegQuality(90)
      ..setKeepDng(false);
    await tester.pumpWidget(KlickApp(settings: settings, camera: cam));
    await tester.pump();
    await tester.pump();
    await tester.tap(find.bySemanticsLabel('Take photo'));
    await tester.pump();
    await tester.pump();
    expect(cam.captures.single, {'flash': 'auto', 'quality': 90, 'keepDng': false});
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('queue events show the processing pill', (tester) async {
    final cam = FakeCamera();
    await tester.pumpWidget(KlickApp(settings: SettingsStore.memory(), camera: cam));
    await tester.pump();
    await tester.pump();
    cam.controller.add({'type': 'job', 'id': 1, 'state': 'processing', 'scene': 'night', 'frames': 9});
    await tester.pump();
    expect(find.text('Processing photo…'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('lens chips switch lens, and auto macro lens events swap the preview', (tester) async {
    final cam = FakeCamera();
    await tester.pumpWidget(KlickApp(settings: SettingsStore.memory(), camera: cam));
    await tester.pump();
    await tester.pump();
    await tester.tap(find.text('0.5x'));
    await tester.pump();
    expect(cam.selected, ['2']);
    // Native auto macro switched back to the main lens and then to macro: the texture follows.
    cam.controller.add({'type': 'lens', 'info': _infoMap('2'), 'macro': true});
    await tester.pump();
    expect(find.text('Macro'), findsOneWidget);
    expect(tester.widget<Texture>(find.byType(Texture)).textureId, 2);
    cam.controller.add({'type': 'lens', 'info': _infoMap('0'), 'macro': false});
    await tester.pump();
    expect(find.text('Macro'), findsNothing);
    expect(tester.widget<Texture>(find.byType(Texture)).textureId, 1);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('phones with a macro lens get a Normal | Macro switch', (tester) async {
    final cam = FakeCamera()..macro = true;
    await tester.pumpWidget(KlickApp(settings: SettingsStore.memory(), camera: cam));
    await tester.pump();
    await tester.pump();
    expect(find.text('Normal'), findsOneWidget);
    expect(find.text('Macro'), findsOneWidget);
    // The macro lens is not repeated as a zoom chip.
    expect(find.text('0.5x'), findsNothing);
    await tester.tap(find.text('Macro'));
    await tester.pump();
    expect(cam.selected, ['2']);
    await tester.tap(find.text('Normal'));
    await tester.pump();
    expect(cam.selected, ['2', '0']);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('denied permission shows a themed message with a retry', (tester) async {
    await tester.pumpWidget(KlickApp(settings: SettingsStore.memory(), camera: FakeCamera(permissionGranted: false)));
    await tester.pump();
    await tester.pump();
    expect(find.text('Camera access is off'), findsOneWidget);
    expect(find.text('Allow camera'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('settings screen has the options and ends with updates and developer', (tester) async {
    final settings = SettingsStore.memory();
    await tester.pumpWidget(
      UiApp(
        home: SettingsScreen(settings: settings, camera: FakeCamera()),
      ),
    );
    await tester.pump(const Duration(milliseconds: 500));
    expect(find.text('Keep RAW (DNG)'), findsOneWidget);
    expect(find.text('Location tagging'), findsOneWidget);
    expect(settings.keepDng, isTrue);
    expect(settings.locationTag, isFalse);
    await tester.tap(find.text('Keep RAW (DNG)'));
    await tester.pump();
    expect(settings.keepDng, isFalse);
    await tester.drag(find.byType(Scrollable).last, const Offset(0, -3000));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.byType(UiAppearanceSection), findsOneWidget);
    expect(find.byType(UiUpdateSection), findsOneWidget);
    expect(find.byType(UiDeveloperSection), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('empty gallery explains itself', (tester) async {
    await tester.pumpWidget(UiApp(home: GalleryScreen(camera: FakeCamera())));
    await tester.pump();
    await tester.pump();
    expect(find.text('No photos yet'), findsOneWidget);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('viewer offers before/after only when a reference frame exists', (tester) async {
    await tester.pumpWidget(
      UiApp(
        home: ViewerScreen(
          photos: [photo('a', ref: '/tmp/a.jpg')],
          index: 0,
          camera: FakeCamera(),
        ),
      ),
    );
    await tester.pump();
    // The processed photo shows alone until Compare is tapped.
    expect(find.text('Before'), findsNothing);
    await tester.tap(find.byTooltip('Compare before and after'));
    await tester.pump();
    expect(find.text('Before'), findsOneWidget);
    expect(find.text('After'), findsOneWidget);
    // Dragging the divider handle moves it.
    final handle = find.byKey(const Key('compare-handle'));
    final x0 = tester.getCenter(handle).dx;
    await tester.drag(handle, const Offset(-120, 0));
    await tester.pump();
    expect(tester.getCenter(handle).dx, lessThan(x0 - 60));
    await tester.pumpWidget(const SizedBox());
    await tester.pumpWidget(
      UiApp(
        home: ViewerScreen(photos: [photo('b')], index: 0, camera: FakeCamera()),
      ),
    );
    await tester.pump();
    expect(find.text('Before'), findsNothing);
    expect(find.byTooltip('Compare before and after'), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });

  test('settings defaults', () {
    final s = SettingsStore.memory();
    expect(s.keepDng, isTrue);
    expect(s.jpegQuality, 95);
    expect(s.shutterSound, isTrue);
    expect(s.grid, isFalse);
    expect(s.locationTag, isFalse);
    s.setJpegQuality(77);
    expect(s.jpegQuality, 95);
  });
}
