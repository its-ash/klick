import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:theme/theme.dart';

import '../core/photo.dart';
import '../platform/camera_channel.dart';
import 'viewer_screen.dart';

/// Grid of the photos Klick saved in Pictures/Klick.
class GalleryScreen extends StatefulWidget {
  const GalleryScreen({super.key, required this.camera});

  final CameraApi camera;

  @override
  State<GalleryScreen> createState() => _GalleryScreenState();
}

class _GalleryScreenState extends State<GalleryScreen> {
  late Future<List<Photo>> _photos = widget.camera.listPhotos();

  void _reload() => setState(() => _photos = widget.camera.listPhotos());

  @override
  Widget build(BuildContext context) {
    return UiScaffold(
      padding: EdgeInsets.zero,
      appBar: const UiAppBar(title: 'Photos', back: true),
      body: FutureBuilder<List<Photo>>(
        future: _photos,
        builder: (context, snap) {
          if (snap.hasError) return UiErrorState(message: '${snap.error}', onRetry: _reload);
          if (!snap.hasData) return const Center(child: UiSpinner());
          final photos = snap.data!;
          if (photos.isEmpty) {
            return const UiEmptyState(
              title: 'No photos yet',
              message: 'Photos you take with Klick appear here.',
              icon: Icons.photo_library_outlined,
            );
          }
          return GridView.builder(
            padding: const EdgeInsets.all(UiSpace.sm),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
              mainAxisSpacing: UiSpace.xs,
              crossAxisSpacing: UiSpace.xs,
            ),
            itemCount: photos.length,
            itemBuilder: (context, i) => _Thumb(
              photo: photos[i],
              camera: widget.camera,
              onTap: () async {
                await Navigator.of(context).push<bool>(
                  MaterialPageRoute(builder: (_) => ViewerScreen(photos: photos, index: i, camera: widget.camera)),
                );
                if (mounted) _reload();
              },
            ),
          );
        },
      ),
    );
  }
}

class _Thumb extends StatefulWidget {
  const _Thumb({required this.photo, required this.camera, required this.onTap});

  final Photo photo;
  final CameraApi camera;
  final VoidCallback onTap;

  @override
  State<_Thumb> createState() => _ThumbState();
}

class _ThumbState extends State<_Thumb> {
  late final Future<Uint8List> _bytes = widget.camera.loadImage(widget.photo.uri, 360);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onTap,
      child: ClipRRect(
        borderRadius: UiRadius.all(UiRadius.xs),
        child: ColoredBox(
          color: context.colors.surfaceMuted,
          child: FutureBuilder<Uint8List>(
            future: _bytes,
            builder: (context, snap) => snap.hasData
                ? Image.memory(snap.data!, fit: BoxFit.cover, gaplessPlayback: true)
                : const SizedBox.expand(),
          ),
        ),
      ),
    );
  }
}
