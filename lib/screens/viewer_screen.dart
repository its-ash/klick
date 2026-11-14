import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:theme/theme.dart';

import '../core/photo.dart';
import '../platform/camera_channel.dart';

/// Full-screen viewer with swipe between photos, a before/after compare slider, share and delete.
class ViewerScreen extends StatefulWidget {
  const ViewerScreen({super.key, required this.photos, required this.index, required this.camera});

  final List<Photo> photos;
  final int index;
  final CameraApi camera;

  @override
  State<ViewerScreen> createState() => _ViewerScreenState();
}

class _ViewerScreenState extends State<ViewerScreen> {
  late final PageController _pages = PageController(initialPage: widget.index);
  late final List<Photo> _photos = List.of(widget.photos);
  late int _index = widget.index;

  /// Before/after slider on the current photo; off by default, so the processed photo shows alone.
  bool _compare = false;

  Photo get _photo => _photos[_index];

  @override
  void dispose() {
    _pages.dispose();
    super.dispose();
  }

  Future<void> _delete() async {
    final ok = await UiDialog.confirm(
      context,
      title: 'Delete photo?',
      message: 'This removes the photo from your phone. A saved RAW (DNG) copy is kept.',
      destructive: true,
      confirmLabel: 'Delete',
    );
    if (!ok || !mounted) return;
    final removed = await widget.camera.delete(_photo.uri);
    if (!mounted) return;
    if (!removed) {
      context.showToast('Could not delete this photo');
      return;
    }
    if (_photos.length == 1) {
      Navigator.of(context).pop(true);
      return;
    }
    setState(() {
      _photos.removeAt(_index);
      _index = _index.clamp(0, _photos.length - 1);
      _compare = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final photo = _photo;
    return UiScaffold(
      padding: EdgeInsets.zero,
      appBar: UiAppBar(
        title: '${_index + 1} / ${_photos.length}',
        back: true,
        actions: [
          if (photo.hasBefore)
            UiIconButton(
              _compare ? Icons.photo_outlined : Icons.compare_rounded,
              tooltip: _compare ? 'Show processed photo' : 'Compare before and after',
              onPressed: () => setState(() => _compare = !_compare),
            ),
          UiIconButton(Icons.ios_share_rounded, tooltip: 'Share', onPressed: () => widget.camera.share(photo.uri)),
          UiIconButton(Icons.delete_outline_rounded, tooltip: 'Delete', onPressed: _delete),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: PageView.builder(
              controller: _pages,
              itemCount: _photos.length,
              onPageChanged: (i) => setState(() {
                _index = i;
                _compare = false;
              }),
              itemBuilder: (context, i) =>
                  _Page(key: ValueKey(_photos[i].uri), photo: _photos[i], camera: widget.camera, compare: _compare && i == _index),
            ),
          ),
          if (_compare && photo.hasBefore)
            Padding(
              padding: const EdgeInsets.all(UiSpace.lg),
              child: Text(
                'Drag the line to compare the single frame with the processed photo',
                style: context.text.bodySmall,
                textAlign: TextAlign.center,
              ),
            ),
        ],
      ),
    );
  }
}

class _Page extends StatefulWidget {
  const _Page({super.key, required this.photo, required this.camera, required this.compare});

  final Photo photo;
  final CameraApi camera;

  /// Show the before/after slider instead of the processed photo alone.
  final bool compare;

  @override
  State<_Page> createState() => _PageState();
}

class _PageState extends State<_Page> {
  late final Future<Uint8List> _after = widget.camera.loadImage(widget.photo.uri, 2048);

  /// Shared by both layers so pinch-zoom keeps before and after lined up.
  final TransformationController _zoom = TransformationController();

  /// Divider position, 0 (all after) to 1 (all before), as a fraction of the page width.
  double _split = 0.5;

  @override
  void dispose() {
    _zoom.dispose();
    super.dispose();
  }

  Widget get _afterImage => FutureBuilder<Uint8List>(
    future: _after,
    builder: (context, snap) {
      if (snap.hasError) return const _Missing();
      if (!snap.hasData) return const UiSpinner();
      return Image.memory(snap.data!, fit: BoxFit.contain, gaplessPlayback: true);
    },
  );

  @override
  Widget build(BuildContext context) {
    final ref = widget.compare ? widget.photo.refPath : null;
    final after = InteractiveViewer(
      transformationController: _zoom,
      minScale: 1,
      maxScale: 5,
      child: Center(child: _afterImage),
    );
    // One tree for both modes, so turning the slider on or off never reloads the processed photo.
    return LayoutBuilder(
      builder: (context, c) {
        final width = c.maxWidth;
        final x = width * _split;
        return Stack(
          fit: StackFit.expand,
          children: [
            after,
            if (ref != null) ...[
              // The single reference frame, shown left of the divider, following the same zoom.
              IgnorePointer(
                child: ClipRect(
                  clipper: _LeftOf(x),
                  child: InteractiveViewer(
                    transformationController: _zoom,
                    minScale: 1,
                    maxScale: 5,
                    child: Center(
                      child: Image.file(File(ref), fit: BoxFit.contain, errorBuilder: (_, _, _) => const _Missing()),
                    ),
                  ),
                ),
              ),
              const Positioned(
                top: UiSpace.sm,
                left: UiSpace.sm,
                child: IgnorePointer(child: _Tag('Before')),
              ),
              const Positioned(
                top: UiSpace.sm,
                right: UiSpace.sm,
                child: IgnorePointer(child: _Tag('After')),
              ),
              // Drag handle: a wide touch strip around the line, so page swipes still work elsewhere.
              Positioned(
                key: const Key('compare-handle'),
                left: x - 28,
                top: 0,
                bottom: 0,
                width: 56,
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onHorizontalDragUpdate: (d) => setState(() => _split = ((_split * width + d.delta.dx) / width).clamp(0.0, 1.0)),
                  child: const _Handle(),
                ),
              ),
            ],
          ],
        );
      },
    );
  }
}

class _LeftOf extends CustomClipper<Rect> {
  const _LeftOf(this.x);

  final double x;

  @override
  Rect getClip(Size size) => Rect.fromLTWH(0, 0, x.clamp(0.0, size.width), size.height);

  @override
  bool shouldReclip(covariant _LeftOf old) => old.x != x;
}

class _Handle extends StatelessWidget {
  const _Handle();

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        Center(child: Container(width: 2, color: Colors.white)),
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            boxShadow: const [BoxShadow(color: Colors.black38, blurRadius: 6)],
          ),
          child: const Icon(Icons.compare_arrows_rounded, color: Colors.black87, size: 22),
        ),
      ],
    );
  }
}

class _Tag extends StatelessWidget {
  const _Tag(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(999)),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
      ),
    );
  }
}

class _Missing extends StatelessWidget {
  const _Missing();

  @override
  Widget build(BuildContext context) =>
      const UiEmptyState(title: 'Cannot show this photo', icon: Icons.broken_image_outlined, compact: true);
}
