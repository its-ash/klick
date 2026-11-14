import 'package:flutter_test/flutter_test.dart';
import 'package:klick/core/scene.dart';

MeterInput meter({int iso = 100, int exposureNs = 10000000, double aperture = 1.8, double shake = 0.0, int maxIso = 3200}) =>
    MeterInput(iso: iso, exposureNs: exposureNs, aperture: aperture, shakeRms: shake, maxIso: maxIso);

void main() {
  test('scene value follows exposure, ISO and aperture', () {
    // f/1.8, 1/1000 s, ISO 100 -> about EV 11.7.
    expect(sceneEv(100, 1000000, 1.8), closeTo(11.66, 0.01));
    // Doubling the ISO lowers the scene EV by one stop.
    expect(sceneEv(200, 1000000, 1.8), closeTo(10.66, 0.01));
    // Unknown aperture falls back to f/1.8.
    expect(sceneEv(100, 1000000, 0), closeTo(11.66, 0.01));
  });

  test('bright sun is HDR with 4 underexposed frames', () {
    final p = planBurst(meter(iso: 50, exposureNs: 1000000));
    expect(p.scene, Scene.hdr);
    expect(p.frames, 4);
    expect(p.exposureNs, 500000);
    expect(p.iso, 50);
    expect(p.evGain, closeTo(2.0, 1e-9));
  });

  test('daylight without extreme contrast is normal with 3 frames', () {
    final p = planBurst(meter(iso: 100, exposureNs: 4000000));
    expect(p.scene, Scene.normal);
    expect(p.frames, 3);
    expect(p.evGain, closeTo(1 / 0.8, 1e-6));
  });

  test('indoor light is normal with 5 frames', () {
    final p = planBurst(meter(iso: 200, exposureNs: 20000000));
    expect(p.scene, Scene.normal);
    expect(p.frames, 5);
    expect(p.evGain, closeTo(1 / 0.7, 1e-6));
  });

  test('low light is night with 8 to 12 frames and half exposure', () {
    final p = planBurst(meter(iso: 1600, exposureNs: 100000000));
    expect(p.scene, Scene.night);
    expect(p.frames, 12);
    expect(p.exposureNs, 50000000);
    final dim = planBurst(meter(iso: 800, exposureNs: 33000000));
    expect(dim.scene, Scene.night);
    expect(dim.frames, inInclusiveRange(8, 12));
    final nearlyDark = planBurst(meter(iso: 400, exposureNs: 40000000));
    expect(nearlyDark.frames, inInclusiveRange(8, 12));
  });

  test('hand shake caps the exposure time and raises ISO to compensate', () {
    final steady = planBurst(meter(iso: 1600, exposureNs: 100000000, shake: 0.0));
    final shaky = planBurst(meter(iso: 1600, exposureNs: 100000000, shake: 0.5));
    expect(shaky.shake, Shake.shaky);
    expect(shaky.exposureNs, 16000000);
    expect(shaky.exposureNs, lessThan(steady.exposureNs));
    expect(shaky.iso, 3200); // clamped to the analog maximum
    // The remaining deficit is restored digitally.
    expect(shaky.evGain, closeTo((1600 * 100000000) / (3200 * 16000000), 1e-9));
    expect(shaky.evGain, greaterThan(steady.evGain));
  });

  test('a steady phone may expose longer at night than by day', () {
    final night = planBurst(meter(iso: 3200, exposureNs: 1000000000, maxIso: 6400));
    expect(night.scene, Scene.night);
    expect(night.exposureNs, 250000000);
    final day = planBurst(meter(iso: 50, exposureNs: 200000000, maxIso: 6400));
    expect(day.scene, Scene.normal);
    expect(day.exposureNs, 125000000);
  });

  test('shake levels', () {
    expect(shakeOf(0.0), Shake.steady);
    expect(shakeOf(0.05), Shake.moderate);
    expect(shakeOf(0.2), Shake.shaky);
    expect(planBurst(meter(shake: 0.05)).shake, Shake.moderate);
  });

  test('gain never drops below 1 and ISO never exceeds the analog maximum', () {
    for (final iso in [50, 100, 400, 1600, 6400]) {
      for (final exp in [100000, 4000000, 50000000, 500000000]) {
        for (final sh in [0.0, 0.05, 0.4]) {
          final p = planBurst(meter(iso: iso, exposureNs: exp, shake: sh, maxIso: 3200));
          expect(p.evGain, greaterThanOrEqualTo(1.0));
          expect(p.frames, inInclusiveRange(3, 12));
          expect(p.iso, lessThanOrEqualTo(iso > 3200 ? iso : 3200));
        }
      }
    }
  });

  test('metering events are parsed and bad ones ignored', () {
    final m = MeterInput.fromEvent({'iso': 400, 'exposureNs': 8000000, 'aperture': 1.7, 'shakeRms': 0.01, 'maxIso': 1600});
    expect(m!.iso, 400);
    expect(m.maxIso, 1600);
    expect(MeterInput.fromEvent({'type': 'scene'}), isNull);
  });
}
