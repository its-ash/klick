import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:theme/theme.dart';

import '../core/photo.dart';
import '../platform/camera_channel.dart';

/// Full-screen viewer with swipe between photos, before/after toggle, share and delete.
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
  bool _before = false;

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
      _before = false;
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
                _before = false;
              }),
              itemBuilder: (context, i) => _Page(
                key: ValueKey(_photos[i].uri),
                photo: _photos[i],
                camera: widget.camera,
                before: _before && i == _index,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(UiSpace.lg),
            child: photo.hasBefore
                ? UiSegmentedControl<bool>(
                    const [UiSegment(false, label: 'After'), UiSegment(true, label: 'Before')],
                    value: _before,
                    expand: true,
                    onChanged: (v) => setState(() => _before = v),
                  )
                : Text('No reference frame saved for this photo', style: context.text.bodySmall),
          ),
        ],
      ),
    );
  }
}

class _Page extends StatefulWidget {
  const _Page({super.key, required this.photo, required this.camera, required this.before});

  final Photo photo;
  final CameraApi camera;
  final bool before;

  @override
  State<_Page> createState() => _PageState();
}

class _PageState extends State<_Page> {
  late final Future<Uint8List> _after = widget.camera.loadImage(widget.photo.uri, 2048);

  @override
  Widget build(BuildContext context) {
    final ref = widget.photo.refPath;
    return InteractiveViewer(
      minScale: 1,
      maxScale: 5,
      child: Center(
        child: widget.before && ref != null
            ? Image.file(File(ref), fit: BoxFit.contain, errorBuilder: (_, _, _) => const _Missing())
            : FutureBuilder<Uint8List>(
                future: _after,
                builder: (context, snap) {
                  if (snap.hasError) return const _Missing();
                  if (!snap.hasData) return const UiSpinner();
                  return Image.memory(snap.data!, fit: BoxFit.contain, gaplessPlayback: true);
                },
              ),
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
