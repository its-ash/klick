import 'dart:async';
import 'dart:math' as math;
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

  /// True while the close-focusing lens was chosen automatically (not by the user).
  bool _autoMacro = false;

  /// Device roll in degrees (clockwise from upright), or null while lying flat or unknown.
  final ValueNotifier<double?> _roll = ValueNotifier(null);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    widget.settings.addListener(_onSettings);
    _queue.addListener(_onQueue);
    _events = widget.camera.events().listen(_onEvent);
    _boot();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    widget.settings.removeListener(_onSettings);
    _countdownTimer?.cancel();
    _events?.cancel();
    _roll.dispose();
    _queue.removeListener(_onQueue);
    _queue.dispose();
    widget.camera.stop();
    super.dispose();
  }

  void _onSettings() {
    if (mounted) setState(() {});
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
      final info = await widget.camera.start(lensId: lensId, autoMacro: widget.settings.autoMacro);
      if (!mounted) return;
      setState(() {
        _info = info;
        _phase = _Phase.ready;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      if (lensId != null) {
        // That lens is not open to apps on this phone; it is dropped from the list natively.
        _autoMacro = false;
        context.showToast('That lens is not available on this phone');
        return _start();
      }
      setState(() {
        _phase = _Phase.failed;
        _error = '$e';
      });
    }
  }

  /// A lens chip: instant when that lens already runs as the standby camera.
  Future<void> _selectLens(String id) async {
    _autoMacro = false;
    try {
      final info = await widget.camera.selectLens(id);
      if (mounted) setState(() => _info = info);
    } catch (_) {
      if (!mounted) return;
      context.showToast('That lens is not available on this phone');
      await _start();
    }
  }

  void _onEvent(Map<dynamic, dynamic> e) {
    switch (e['type']) {
      case 'scene':
        final input = MeterInput.fromEvent(e);
        if (input != null && mounted) setState(() => _plan = planBurst(input));
      case 'job':
        _queue.apply(e);
      case 'lens':
        // Auto macro switched lenses natively (both cameras run, so it is instant).
        final m = e['info'];
        if (m is! Map || !mounted) return;
        setState(() {
          _info = CameraInfo.fromMap(m);
          _autoMacro = e['macro'] == true;
        });
      case 'macro':
        // Single-camera phones: auto macro asks for a restart on the other lens.
        final id = e['lensId'];
        if (id is! String || _capturing || _phase != _Phase.ready || !mounted) return;
        if (e['on'] == true && widget.settings.autoMacro && !_autoMacro) {
          _autoMacro = true;
          _start(lensId: id);
        } else if (e['on'] == false && _autoMacro) {
          _autoMacro = false;
          _start(lensId: id);
        }
      case 'level':
        final roll = e['roll'];
        _roll.value = e['flat'] == true || roll is! num ? null : roll.toDouble();
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
        look: s.look.name,
      );
    } catch (e) {
      if (mounted) context.showToast(_clean(e));
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  String _clean(Object e) =>
      e.toString().replaceFirst(RegExp(r'^PlatformException\([^,]*, '), '').replaceFirst(RegExp(r', null, null\)$'), '');

  /// [origin] and [shown] place the whole (cropped) preview frame in screen coordinates.
  void _tapPreview(TapDownDetails d, Offset origin, Size shown) {
    final nx = ((d.localPosition.dx - origin.dx) / shown.width).clamp(0.0, 1.0);
    final ny = ((d.localPosition.dy - origin.dy) / shown.height).clamp(0.0, 1.0);
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
      body: Stack(
        children: [
          Positioned.fill(child: info != null && _phase == _Phase.ready ? _preview(info) : const Center(child: UiSpinner())),
          // Scrims keep the controls readable over a bright full-screen preview.
          const Positioned(top: 0, left: 0, right: 0, height: 140, child: IgnorePointer(child: _Scrim(top: true))),
          const Positioned(bottom: 0, left: 0, right: 0, height: 220, child: IgnorePointer(child: _Scrim(top: false))),
          SafeArea(
            child: Stack(
              children: [
                Positioned(top: 0, left: 0, right: 0, child: _topBar(info)),
                Positioned(left: 0, right: 0, bottom: 0, child: _bottom(info)),
                if (_countdown != null)
                  Positioned.fill(
                    child: GestureDetector(
                      onTap: _cancelTimer,
                      child: Center(
                        child: Text(
                          '$_countdown',
                          style: const TextStyle(color: Colors.white, fontSize: 96, fontWeight: FontWeight.w300),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _preview(CameraInfo info) {
    // Camera2 already rotates SurfaceTexture frames upright for the device's natural (portrait)
    // orientation and Flutter applies that transform, so the texture fills a portrait box as is.
    // Full screen: scale the portrait preview to cover the screen and crop the overflow.
    return LayoutBuilder(
      builder: (context, c) {
        final screen = c.biggest;
        final pw = info.previewHeight.toDouble();
        final ph = info.previewWidth.toDouble();
        final scale = math.max(screen.width / pw, screen.height / ph);
        final shown = Size(pw * scale, ph * scale);
        final origin = Offset((screen.width - shown.width) / 2, (screen.height - shown.height) / 2);
        return GestureDetector(
          onTapDown: (d) => _tapPreview(d, origin, shown),
          child: Stack(
            fit: StackFit.expand,
            children: [
              ClipRect(
                child: OverflowBox(
                  maxWidth: shown.width,
                  maxHeight: shown.height,
                  child: SizedBox.fromSize(
                    size: shown,
                    child: Texture(textureId: info.textureId),
                  ),
                ),
              ),
              if (widget.settings.grid) const IgnorePointer(child: CustomPaint(painter: _GridPainter())),
              if (widget.settings.grid)
                IgnorePointer(
                  child: ValueListenableBuilder<double?>(
                    valueListenable: _roll,
                    builder: (context, roll, _) => roll == null ? const SizedBox.shrink() : CustomPaint(painter: _LevelPainter(roll)),
                  ),
                ),
              if (_focusPoint != null)
                Positioned(
                  left: _focusPoint!.dx - 32,
                  top: _focusPoint!.dy - 32,
                  child: IgnorePointer(
                    child: Container(
                      width: 64,
                      height: 64,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.amber, width: 2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
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
                icon: Icon(
                  _timerIndex == 0 ? Icons.timer_off_outlined : Icons.timer_rounded,
                  color: _timerIndex == 0 ? Colors.white : Colors.amber,
                ),
              ),
              const Spacer(),
              if (plan != null) _SceneBadge(plan: plan, raw: info?.raw ?? true),
              const Spacer(),
              IconButton(
                tooltip: 'Settings',
                icon: const Icon(Icons.settings_rounded, color: Colors.white),
                onPressed: () => Navigator.of(context)
                    .push(
                      MaterialPageRoute(
                        builder: (_) => SettingsScreen(settings: widget.settings, camera: widget.camera),
                      ),
                    )
                    .then((_) => widget.camera.setAutoMacro(widget.settings.autoMacro)),
              ),
            ],
          ),
          if (_engine != null && !_engine!.available) const _Pill(text: 'Processing engine missing in this build', color: Colors.redAccent),
          if (_autoMacro) const _Pill(text: 'Macro', color: Colors.green),
          if (queue != null) _Pill(text: queue, color: Colors.white24, spinner: true),
          if (info != null && !info.raw) const _Pill(text: 'No RAW on this camera: using YUV bursts', color: Colors.white24),
        ],
      ),
    );
  }

  Widget _bottom(CameraInfo? info) {
    final lenses = info?.lenses ?? const <Lens>[];
    final mainId = info?.mainLensId;
    final macroId = info?.macroLensId;
    // A dedicated Normal | Macro switch when the phone has a close-focusing lens; the zoom chips then
    // list the other lenses only.
    final tabs = mainId != null && macroId != null && lenses.any((l) => l.id == macroId);
    final chips = tabs ? lenses.where((l) => l.id != macroId).toList() : lenses;
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (tabs)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: SizedBox(
                width: 240,
                child: UiSegmentedControl<String>(
                  [UiSegment(mainId, label: 'Normal'), UiSegment(macroId, label: 'Macro')],
                  value: info!.lensId == macroId ? macroId : mainId,
                  expand: true,
                  onChanged: (id) {
                    if (!_capturing && id != info.lensId) _selectLens(id);
                  },
                ),
              ),
            ),
          if (chips.length > 1)
            Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  for (final l in chips)
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      child: ChoiceChip(
                        label: Text(l.label),
                        selected: l.id == info?.lensId,
                        showCheckmark: false,
                        onSelected: _capturing
                            ? null
                            : (_) {
                                _selectLens(l.id);
                              },
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
              _LookButton(look: widget.settings.look, onTap: _capturing ? null : widget.settings.cycleLook),
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
      decoration: BoxDecoration(
        color: Colors.black54,
        border: Border.all(color: color),
        borderRadius: BorderRadius.circular(999),
      ),
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
            if (spinner)
              const Padding(
                padding: EdgeInsets.only(right: 8),
                child: SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
              ),
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
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 4),
          ),
          child: DecoratedBox(
            decoration: BoxDecoration(shape: BoxShape.circle, color: busy || counting ? Colors.white54 : Colors.white),
            child: busy
                ? const Center(
                    child: SizedBox(width: 28, height: 28, child: CircularProgressIndicator(strokeWidth: 3, color: Colors.black54)),
                  )
                : null,
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

class _LookButton extends StatelessWidget {
  const _LookButton({required this.look, required this.onTap});

  final PhotoLook look;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final accent = switch (look) {
      PhotoLook.pro => Colors.amberAccent,
      PhotoLook.natural => Colors.white,
      PhotoLook.vivid => Colors.orangeAccent,
      PhotoLook.film => const Color(0xFFFFD180),
      PhotoLook.mono => Colors.white70,
    };
    return Semantics(
      button: true,
      excludeSemantics: true,
      label: 'Photo look: ${look.label}',
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: 56,
          height: 56,
          decoration: BoxDecoration(
            color: Colors.white12,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: accent.withValues(alpha: 0.7)),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.auto_fix_high_rounded, size: 20, color: accent),
              const SizedBox(height: 2),
              Text(
                look.label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(color: accent, fontSize: 11, fontWeight: FontWeight.w600),
              ),
            ],
          ),
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

class _Scrim extends StatelessWidget {
  const _Scrim({required this.top});

  final bool top;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: top ? Alignment.topCenter : Alignment.bottomCenter,
          end: top ? Alignment.bottomCenter : Alignment.topCenter,
          colors: const [Colors.black54, Colors.transparent],
        ),
      ),
    );
  }
}

/// Horizon level: fixed ticks mark the way the phone is held (portrait or landscape), and the
/// center line shows the real horizon. Both turn amber within 1 degree of level.
class _LevelPainter extends CustomPainter {
  const _LevelPainter(this.roll);

  /// Degrees clockwise from natural upright.
  final double roll;

  @override
  void paint(Canvas canvas, Size size) {
    final snap = (roll / 90).round() * 90.0;
    final tilt = roll - snap;
    final level = tilt.abs() < 1;
    final half = size.shortestSide * 0.18;
    final ticks = Paint()
      ..color = level ? Colors.amber : Colors.white70
      ..strokeWidth = 2;
    final horizon = Paint()
      ..color = level ? Colors.amber : Colors.white
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;
    final c = size.center(Offset.zero);
    canvas.save();
    canvas.translate(c.dx, c.dy);
    canvas.rotate(-snap * math.pi / 180);
    canvas.drawLine(Offset(-half - 28, 0), Offset(-half - 8, 0), ticks);
    canvas.drawLine(Offset(half + 8, 0), Offset(half + 28, 0), ticks);
    canvas.rotate(-tilt * math.pi / 180);
    canvas.drawLine(Offset(-half, 0), Offset(-12, 0), horizon);
    canvas.drawLine(const Offset(12, 0), Offset(half, 0), horizon);
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _LevelPainter old) => old.roll != roll;
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
