import 'dart:async';

import 'package:flutter/services.dart';

import '../core/photo.dart';

class Lens {
  const Lens({required this.id, required this.label, required this.zoom});
  final String id;
  final String label;
  final double zoom;
}

class CameraInfo {
  const CameraInfo({
    required this.textureId,
    required this.previewWidth,
    required this.previewHeight,
    required this.sensorOrientation,
    required this.raw,
    required this.flash,
    required this.lensId,
    required this.lenses,
    this.mainLensId,
    this.macroLensId,
  });

  final int textureId;
  final int previewWidth;
  final int previewHeight;
  final int sensorOrientation;

  /// True when the back camera delivers RAW; otherwise bursts use the YUV path.
  final bool raw;
  final bool flash;
  final String lensId;
  final List<Lens> lenses;

  /// The everyday lens (about 1x) and the close-focusing one, when the phone has both.
  final String? mainLensId;
  final String? macroLensId;

  static CameraInfo fromMap(Map<dynamic, dynamic> m) => CameraInfo(
    textureId: (m['textureId'] as num).toInt(),
    previewWidth: (m['previewWidth'] as num).toInt(),
    previewHeight: (m['previewHeight'] as num).toInt(),
    sensorOrientation: (m['sensorOrientation'] as num?)?.toInt() ?? 90,
    raw: m['raw'] == true,
    flash: m['flash'] == true,
    lensId: (m['lensId'] as String?) ?? '',
    mainLensId: m['mainLensId'] as String?,
    macroLensId: m['macroLensId'] as String?,
    lenses: [
      for (final l in (m['lenses'] as List? ?? const []))
        Lens(id: l['id'] as String, label: l['label'] as String, zoom: (l['zoom'] as num).toDouble()),
    ],
  );
}

class CaptureAck {
  const CaptureAck({required this.id, required this.frames, required this.scene});
  final int id;
  final int frames;
  final String scene;
}

class EngineInfo {
  const EngineInfo({required this.available, required this.version});
  final bool available;
  final String version;
}

/// Everything the UI needs from the native side. Faked in tests.
abstract class CameraApi {
  Future<EngineInfo> engine();
  Future<bool> permission({bool location = false});
  Future<CameraInfo> start({String? lensId, bool autoMacro = true});

  /// Shows another lens: instant when it already runs as the standby camera, else a restart.
  Future<CameraInfo> selectLens(String lensId);

  /// Lets the camera switch to the close-focusing lens by itself (and back).
  Future<void> setAutoMacro(bool on);
  Future<void> stop();
  Future<void> tapFocus(double x, double y);
  Future<CaptureAck> capture({
    required String flash,
    required int quality,
    required bool keepDng,
    required bool sound,
    required bool location,
  });

  /// Native events: `scene` metering ticks, `job` queue updates, `lens` (auto macro switched the
  /// lens on screen: `info`, `macro`), `macro` (single-camera phones: restart on `lensId`), `level`,
  /// `error`.
  Stream<Map<dynamic, dynamic>> events();

  Future<List<Photo>> listPhotos();
  Future<Uint8List> loadImage(String uri, int maxSide);
  Future<void> share(String uri);
  Future<bool> delete(String uri);
}

class CameraChannel implements CameraApi {
  CameraChannel();

  static const _method = MethodChannel('com.itsash.klick/camera');
  static const _events = EventChannel('com.itsash.klick/events');

  @override
  Future<EngineInfo> engine() async {
    final m = await _method.invokeMapMethod<String, dynamic>('engine');
    return EngineInfo(available: m?['available'] == true, version: (m?['version'] as String?) ?? '');
  }

  @override
  Future<bool> permission({bool location = false}) async =>
      (await _method.invokeMethod<bool>('permission', {'location': location})) ?? false;

  @override
  Future<CameraInfo> start({String? lensId, bool autoMacro = true}) async {
    final m = await _method.invokeMapMethod<String, dynamic>('start', {'lensId': lensId, 'autoMacro': autoMacro});
    return CameraInfo.fromMap(m!);
  }

  @override
  Future<CameraInfo> selectLens(String lensId) async {
    final m = await _method.invokeMapMethod<String, dynamic>('selectLens', {'lensId': lensId});
    return CameraInfo.fromMap(m!);
  }

  @override
  Future<void> setAutoMacro(bool on) => _method.invokeMethod('setAutoMacro', {'on': on});

  @override
  Future<void> stop() => _method.invokeMethod('stop');

  @override
  Future<void> tapFocus(double x, double y) => _method.invokeMethod('tapFocus', {'x': x, 'y': y});

  @override
  Future<CaptureAck> capture({
    required String flash,
    required int quality,
    required bool keepDng,
    required bool sound,
    required bool location,
  }) async {
    final m = await _method.invokeMapMethod<String, dynamic>('capture', {
      'flash': flash,
      'quality': quality,
      'keepDng': keepDng,
      'sound': sound,
      'location': location,
    });
    return CaptureAck(
      id: (m?['id'] as num?)?.toInt() ?? 0,
      frames: (m?['frames'] as num?)?.toInt() ?? 0,
      scene: (m?['scene'] as String?) ?? 'normal',
    );
  }

  @override
  Stream<Map<dynamic, dynamic>> events() => _events.receiveBroadcastStream().where((e) => e is Map).map((e) => e as Map<dynamic, dynamic>);

  @override
  Future<List<Photo>> listPhotos() async {
    final l = await _method.invokeListMethod<Map<dynamic, dynamic>>('listPhotos');
    return [for (final m in l ?? const <Map<dynamic, dynamic>>[]) Photo.fromMap(m)];
  }

  @override
  Future<Uint8List> loadImage(String uri, int maxSide) async =>
      (await _method.invokeMethod<Uint8List>('loadImage', {'uri': uri, 'maxSide': maxSide}))!;

  @override
  Future<void> share(String uri) => _method.invokeMethod('share', {'uri': uri});

  @override
  Future<bool> delete(String uri) async => (await _method.invokeMethod<bool>('delete', {'uri': uri})) ?? false;
}
