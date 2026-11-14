import 'dart:math' as math;

/// Scene decision rules. Mirrors `ScenePlanner.kt` exactly (the native side is authoritative
/// for the capture, this copy drives the on-screen badge). Change both together.
enum Scene {
  normal('Normal'),
  hdr('HDR'),
  night('Night');

  const Scene(this.label);
  final String label;
}

enum Shake { steady, moderate, shaky }

const double nightBelowEv = 5.0;
const double hdrFromEv = 12.0;
const double steadyRms = 0.03;
const double moderateRms = 0.12;

/// Live metering from the camera preview plus the gyroscope.
class MeterInput {
  const MeterInput({required this.iso, required this.exposureNs, required this.aperture, required this.shakeRms, required this.maxIso});

  final int iso;
  final int exposureNs;
  final double aperture;

  /// Gyroscope magnitude, rad/s, rolling RMS.
  final double shakeRms;
  final int maxIso;

  static MeterInput? fromEvent(Map<dynamic, dynamic> e) {
    final iso = e['iso'];
    final exp = e['exposureNs'];
    if (iso is! num || exp is! num) return null;
    return MeterInput(
      iso: iso.toInt(),
      exposureNs: exp.toInt(),
      aperture: (e['aperture'] as num?)?.toDouble() ?? 0,
      shakeRms: (e['shakeRms'] as num?)?.toDouble() ?? 0,
      maxIso: (e['maxIso'] as num?)?.toInt() ?? 800,
    );
  }
}

class BurstPlan {
  const BurstPlan({
    required this.scene,
    required this.frames,
    required this.exposureNs,
    required this.iso,
    required this.evGain,
    required this.ev,
    required this.shake,
  });

  final Scene scene;
  final int frames;

  /// Per-frame exposure, shorter than metered (highlight protection) and capped by hand shake.
  final int exposureNs;
  final int iso;

  /// Digital gain that restores the deliberate under-exposure.
  final double evGain;
  final double ev;
  final Shake shake;
}

double _log2(double x) => math.log(x) / math.ln2;

/// Exposure value at ISO 100.
double sceneEv(int iso, int exposureNs, double aperture) {
  final n = aperture > 0 ? aperture : 1.8;
  final t = math.max(exposureNs, 1) / 1e9;
  return _log2(n * n / t) - _log2(math.max(iso, 1) / 100.0);
}

Shake shakeOf(double rms) {
  if (rms < steadyRms) return Shake.steady;
  if (rms < moderateRms) return Shake.moderate;
  return Shake.shaky;
}

BurstPlan planBurst(MeterInput i) {
  final ev = sceneEv(i.iso, i.exposureNs, i.aperture);
  final shake = shakeOf(i.shakeRms);
  final Scene scene;
  if (ev < nightBelowEv) {
    scene = Scene.night;
  } else if (ev >= hdrFromEv) {
    scene = Scene.hdr;
  } else {
    scene = Scene.normal;
  }
  final int frames = switch (scene) {
    Scene.night => (8 + ((nightBelowEv - ev) * 1.5).round()).clamp(8, 12),
    Scene.hdr => 4,
    Scene.normal => ev >= 9.0 ? 3 : 5,
  };
  final double under = switch (scene) {
    Scene.hdr || Scene.night => 0.5,
    Scene.normal => ev >= 9.0 ? 0.8 : 0.7,
  };
  final int capNs = switch (shake) {
    Shake.steady => scene == Scene.night ? 250000000 : 125000000,
    Shake.moderate => 33000000,
    Shake.shaky => 16000000,
  };
  final desired = math.max((i.exposureNs * under).truncate(), 1);
  final t = math.min(desired, capNs);
  final iso = math.min(i.maxIso.toDouble(), (i.iso * desired / t).round().toDouble()).round();
  final evGain = math.max(1.0, (i.iso * i.exposureNs) / (iso * t));
  return BurstPlan(scene: scene, frames: frames, exposureNs: t, iso: iso, evGain: evGain, ev: ev, shake: shake);
}
