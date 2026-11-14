import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:theme/theme.dart';

import '../core/queue.dart';
import '../core/scene.dart';
import '../platform/camera_channel.dart';
import '../settings/settings_store.dart';
import 'gallery_screen.dart';
import 'settings_screen.dart';

enum _Phase { starting, ready, denied, failed }

/// Viewfinder. Camera screens are dark by convention, so this screen paints black around the
/// preview; every other screen uses the shared theme.
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.camera, required this.settings});

  final CameraApi camera;
  final SettingsStore settings;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  static const _timers = [0, 3, 10];

  final ProcessingQueue _queue = ProcessingQueue();
  StreamSubscription<Map<dynamic, dynamic>>? _events;
  _Phase _phase = _Phase.starting;
  String? _error;
  CameraInfo? _info;
  EngineInfo? _engine;
  BurstPlan? _plan;
  bool _flashAuto = true;
  int _timerIndex = 0;
  int? _countdown;
  Timer? _countdownTimer;
  bool _capturing = false;
  Uint8List? _lastThumb;
  Offset? _focusPoint;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _queue.addListener(_onQueue);
    _events = widget.camera.events().listen(_onEvent);
    _boot();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _countdownTimer?.cancel();
    _events?.cancel();
    _queue.removeListener(_onQueue);
    _queue.dispose();
    widget.camera.stop();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      if (_phase == _Phase.ready || _phase == _Phase.failed) _start();
    } else if (state == AppLifecycleState.paused) {
      widget.camera.stop();
    }
  }

  Future<void> _boot() async {
    try {
      _engine = await widget.camera.engine();
    } catch (_) {
      _engine = const EngineInfo(available: false, version: '');
    }
    await _start();
  }

  Future<void> _start({String? lensId}) async {
    if (!mounted) return;
    setState(() => _phase = _Phase.starting);
    try {
      final ok = await widget.camera.permission();
      if (!mounted) return;
      if (!ok) {
        setState(() => _phase = _Phase.denied);
        return;
      }
      final info = await widget.camera.start(lensId: lensId);
      if (!mounted) return;
      setState(() {
        _info = info;
        _phase = _Phase.ready;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _phase = _Phase.failed;
        _error = '$e';
      });
    }
  }

  void _onEvent(Map<dynamic, dynamic> e) {
    switch (e['type']) {
      case 'scene':
        final input = MeterInput.fromEvent(e);
        if (input != null && mounted) setState(() => _plan = planBurst(input));
      case 'job':
        _queue.apply(e);
      case 'error':
        if (mounted) context.showToast('${e['message']}');
    }
  }

  void _onQueue() {
    if (!mounted) return;
    final done = _queue.lastDone;
    if (done?.uri != null) _loadThumb(done!.uri!);
    final err = _queue.lastError;
    if (err != null) {
      context.showToast(err);
      _queue.clearError();
    }
    setState(() {});
  }

  Future<void> _loadThumb(String uri) async {
    try {
      final b = await widget.camera.loadImage(uri, 200);
      if (mounted) setState(() => _lastThumb = b);
    } catch (_) {}
  }

  Future<void> _shutter() async {
    if (_capturing || _countdown != null || _phase != _Phase.ready) return;
    final seconds = _timers[_timerIndex];
    if (seconds == 0) return _capture();
    setState(() => _countdown = seconds);
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      final left = (_countdown ?? 1) - 1;
      if (left <= 0) {
        t.cancel();
        setState(() => _countdown = null);
        _capture();
      } else {
        setState(() => _countdown = left);
      }
    });
  }

  void _cancelTimer() {
    _countdownTimer?.cancel();
    setState(() => _countdown = null);
  }

  Future<void> _capture() async {
    setState(() => _capturing = true);
    final s = widget.settings;
    try {
      await widget.camera.capture(
        flash: _flashAuto ? 'auto' : 'off',
        quality: s.jpegQuality,
        keepDng: s.keepDng,
        sound: s.shutterSound,
        location: s.locationTag,
      );
    } catch (e) {
      if (mounted) context.showToast(_clean(e));
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  String _clean(Object e) => e.toString().replaceFirst(RegExp(r'^PlatformException\([^,]*, '), '').replaceFirst(RegExp(r', null, null\)$'), '');

  void _tapPreview(TapDownDetails d, Size size) {
    final nx = (d.localPosition.dx / size.width).clamp(0.0, 1.0);
    final ny = (d.localPosition.dy / size.height).clamp(0.0, 1.0);
    setState(() => _focusPoint = d.localPosition);
    widget.camera.tapFocus(nx, ny);
    Timer(const Duration(milliseconds: 1200), () {
      if (mounted) setState(() => _focusPoint = null);
    });
  }

  @override
  Widget build(BuildContext context) {
    return switch (_phase) {
      _Phase.denied => _Blocked(
          title: 'Camera access is off',
          message: 'Klick needs the camera to take photos. Allow it and try again.',
          icon: Icons.no_photography_outlined,
          action: UiButton('Allow camera', onPressed: _start),
        ),
      _Phase.failed => _Blocked(
          title: 'Camera not available',
          message: _error ?? 'Something went wrong opening the camera.',
          icon: Icons.error_outline_rounded,
          action: UiButton('Try again', onPressed: _start),
        ),
      _ => _viewfinder(context),
    };
  }

  Widget _viewfinder(BuildContext context) {
    final info = _info;
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            if (info != null && _phase == _Phase.ready) _preview(info) else const Center(child: UiSpinner()),
            Positioned(top: 0, left: 0, right: 0, child: _topBar(info)),
            Positioned(left: 0, right: 0, bottom: 0, child: _bottom(info)),
            if (_countdown != null)
              Positioned.fill(
                child: GestureDetector(
                  onTap: _cancelTimer,
                  child: Center(
                    child: Text('$_countdown', style: const TextStyle(color: Colors.white, fontSize: 96, fontWeight: FontWeight.w300)),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _preview(CameraInfo info) {
    final turns = info.sensorOrientation ~/ 90;
    return Center(
      child: AspectRatio(
        aspectRatio: info.previewHeight / info.previewWidth,
        child: LayoutBuilder(
          builder: (context, c) => GestureDetector(
            onTapDown: (d) => _tapPreview(d, c.biggest),
            child: Stack(
              fit: StackFit.expand,
              children: [
                RotatedBox(
                  quarterTurns: turns,
                  child: AspectRatio(
                    aspectRatio: info.previewWidth / info.previewHeight,
                    child: Texture(textureId: info.textureId),
                  ),
                ),
                if (widget.settings.grid) const IgnorePointer(child: CustomPaint(painter: _GridPainter())),
                if (_focusPoint != null)
                  Positioned(
                    left: _focusPoint!.dx - 32,
                    top: _focusPoint!.dy - 32,
                    child: IgnorePointer(
                      child: Container(
                        width: 64,
                        height: 64,
                        decoration: BoxDecoration(border: Border.all(color: Colors.amber, width: 2), borderRadius: BorderRadius.circular(8)),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _topBar(CameraInfo? info) {
    final plan = _plan;
    final queue = _queue.label;
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 4, 8, 0),
      child: Column(
        children: [
          Row(
            children: [
              IconButton(
                tooltip: _flashAuto ? 'Flash auto' : 'Flash off',
                onPressed: info?.flash == true ? () => setState(() => _flashAuto = !_flashAuto) : null,
                icon: Icon(_flashAuto ? Icons.flash_auto_rounded : Icons.flash_off_rounded, color: Colors.white),
              ),
              IconButton(
                tooltip: widget.settings.grid ? 'Hide grid' : 'Show grid',
                onPressed: () => setState(() => widget.settings.setGrid(!widget.settings.grid)),
                icon: Icon(Icons.grid_3x3_rounded, color: widget.settings.grid ? Colors.amber : Colors.white),
              ),
              IconButton(
                tooltip: _timerIndex == 0 ? 'Timer off' : 'Timer ${_timers[_timerIndex]} seconds',
                onPressed: () => setState(() => _timerIndex = (_timerIndex + 1) % _timers.length),
                icon: Icon(_timerIndex == 0 ? Icons.timer_off_outlined : Icons.timer_rounded, color: _timerIndex == 0 ? Colors.white : Colors.amber),
              ),
              const Spacer(),
              if (plan != null) _SceneBadge(plan: plan, raw: info?.raw ?? true),
              const Spacer(),
              IconButton(
                tooltip: 'Settings',
                icon: const Icon(Icons.settings_rounded, color: Colors.white),
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => SettingsScreen(settings: widget.settings, camera: widget.camera)),
                ),
              ),
            ],
          ),
          if (_engine != null && !_engine!.available)
            const _Pill(text: 'Processing engine missing in this build', color: Colors.redAccent),
          if (queue != null) _Pill(text: queue, color: Colors.white24, spinner: true),
          if (info != null && !info.raw) const _Pill(text: 'No RAW on this camera: using YUV bursts', color: Colors.white24),
        ],
      ),
    );
  }

  Widget _bottom(CameraInfo? info) {
    final lenses = info?.lenses ?? const <Lens>[];
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (lenses.length > 1)
            Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  for (final l in lenses)
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      child: ChoiceChip(
                        label: Text(l.label),
                        selected: l.id == info?.lensId,
                        showCheckmark: false,
                        onSelected: _capturing ? null : (_) => _start(lensId: l.id),
                      ),
                    ),
                ],
              ),
            ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _GalleryButton(thumb: _lastThumb, onTap: _openGallery),
              _ShutterButton(busy: _capturing, counting: _countdown != null, onTap: _shutter),
              const SizedBox(width: 56, height: 56),
            ],
          ),
          if (_capturing)
            const Padding(
              padding: EdgeInsets.only(top: 12),
              child: Text('Hold still…', style: TextStyle(color: Colors.white70)),
            ),
        ],
      ),
    );
  }

  Future<void> _openGallery() async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => GalleryScreen(camera: widget.camera)));
    final l = _queue.lastDone?.uri;
    if (l != null) _loadThumb(l);
  }
}

class _SceneBadge extends StatelessWidget {
  const _SceneBadge({required this.plan, required this.raw});

  final BurstPlan plan;
  final bool raw;

  @override
  Widget build(BuildContext context) {
    final color = switch (plan.scene) {
      Scene.night => Colors.indigoAccent,
      Scene.hdr => Colors.orangeAccent,
      Scene.normal => Colors.white,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(color: Colors.black54, border: Border.all(color: color), borderRadius: BorderRadius.circular(999)),
      child: Text(
        '${plan.scene.label} · ${plan.frames} frames',
        style: TextStyle(color: color, fontWeight: FontWeight.w600, fontSize: 13),
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.text, required this.color, this.spinner = false});

  final String text;
  final Color color;
  final bool spinner;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
        decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(999)),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (spinner) const Padding(padding: EdgeInsets.only(right: 8), child: SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))),
            Text(text, style: const TextStyle(color: Colors.white, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

class _ShutterButton extends StatelessWidget {
  const _ShutterButton({required this.busy, required this.counting, required this.onTap});

  final bool busy;
  final bool counting;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Take photo',
      child: GestureDetector(
        onTap: busy ? null : onTap,
        child: Container(
          width: 80,
          height: 80,
          padding: const EdgeInsets.all(5),
          decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 4)),
          child: DecoratedBox(
            decoration: BoxDecoration(shape: BoxShape.circle, color: busy || counting ? Colors.white54 : Colors.white),
            child: busy ? const Center(child: SizedBox(width: 28, height: 28, child: CircularProgressIndicator(strokeWidth: 3, color: Colors.black54))) : null,
          ),
        ),
      ),
    );
  }
}

class _GalleryButton extends StatelessWidget {
  const _GalleryButton({required this.thumb, required this.onTap});

  final Uint8List? thumb;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Open photos',
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: 56,
          height: 56,
          decoration: BoxDecoration(
            color: Colors.white12,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.white54),
            image: thumb == null ? null : DecorationImage(image: MemoryImage(thumb!), fit: BoxFit.cover),
          ),
          child: thumb == null ? const Icon(Icons.photo_library_outlined, color: Colors.white70) : null,
        ),
      ),
    );
  }
}

class _GridPainter extends CustomPainter {
  const _GridPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = Colors.white38
      ..strokeWidth = 1;
    for (var i = 1; i < 3; i++) {
      canvas.drawLine(Offset(size.width * i / 3, 0), Offset(size.width * i / 3, size.height), p);
      canvas.drawLine(Offset(0, size.height * i / 3), Offset(size.width, size.height * i / 3), p);
    }
  }

  @override
  bool shouldRepaint(covariant _GridPainter old) => false;
}

/// Themed full-screen message used when the camera cannot start.
class _Blocked extends StatelessWidget {
  const _Blocked({required this.title, required this.message, required this.icon, required this.action});

  final String title;
  final String message;
  final IconData icon;
  final Widget action;

  @override
  Widget build(BuildContext context) {
    return UiScaffold(
      body: UiEmptyState(title: title, message: message, icon: icon, action: action),
    );
  }
}
