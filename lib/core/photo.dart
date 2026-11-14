/// A processed Klick photo in MediaStore `Pictures/Klick`.
class Photo {
  const Photo({
    required this.uri,
    required this.name,
    required this.added,
    this.width = 0,
    this.height = 0,
    this.refPath,
  });

  final String uri;
  final String name;
  final DateTime added;
  final int width;
  final int height;

  /// Path of the reference frame render (the "before" image), when it still exists.
  final String? refPath;

  bool get hasBefore => refPath != null;

  static Photo fromMap(Map<dynamic, dynamic> m) => Photo(
        uri: m['uri'] as String,
        name: (m['name'] as String?) ?? '',
        added: DateTime.fromMillisecondsSinceEpoch((m['added'] as num?)?.toInt() ?? 0),
        width: (m['width'] as num?)?.toInt() ?? 0,
        height: (m['height'] as num?)?.toInt() ?? 0,
        refPath: m['ref'] as String?,
      );
}
