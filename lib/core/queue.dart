import 'package:flutter/foundation.dart';

enum JobState { queued, processing, done, failed }

class JobInfo {
  const JobInfo({required this.id, required this.state, this.scene = 'normal', this.frames = 0, this.uri, this.name, this.message});

  final int id;
  final JobState state;
  final String scene;
  final int frames;
  final String? uri;
  final String? name;
  final String? message;

  bool get active => state == JobState.queued || state == JobState.processing;

  JobInfo copyWith({JobState? state, String? uri, String? name, String? message}) => JobInfo(
    id: id,
    state: state ?? this.state,
    scene: scene,
    frames: frames,
    uri: uri ?? this.uri,
    name: name ?? this.name,
    message: message ?? this.message,
  );
}

/// Tracks photos being processed natively so the user can keep shooting.
/// Fed with the `job` events of the native event channel.
class ProcessingQueue extends ChangeNotifier {
  final Map<int, JobInfo> _jobs = {};
  JobInfo? _lastDone;
  String? _lastError;

  List<JobInfo> get jobs => List.unmodifiable(_jobs.values);
  int get activeCount => _jobs.values.where((j) => j.active).length;
  bool get busy => activeCount > 0;
  JobInfo? get lastDone => _lastDone;
  String? get lastError => _lastError;

  /// Text for the "processing" pill; null when idle.
  String? get label {
    final n = activeCount;
    if (n == 0) return null;
    final processing = _jobs.values.where((j) => j.state == JobState.processing).length;
    if (n == 1) return processing == 1 ? 'Processing photo…' : 'Photo queued…';
    return 'Processing $n photos…';
  }

  void clearError() {
    if (_lastError == null) return;
    _lastError = null;
    notifyListeners();
  }

  /// Applies one native event. Unknown events are ignored.
  void apply(Map<dynamic, dynamic> e) {
    if (e['type'] != 'job') return;
    final id = (e['id'] as num?)?.toInt();
    final state = switch (e['state']) {
      'queued' => JobState.queued,
      'processing' => JobState.processing,
      'done' => JobState.done,
      'failed' => JobState.failed,
      _ => null,
    };
    if (id == null || state == null) return;
    final old = _jobs[id];
    var job = old ?? JobInfo(id: id, state: state, scene: (e['scene'] as String?) ?? 'normal', frames: (e['frames'] as num?)?.toInt() ?? 0);
    // States only move forward.
    if (old != null && state.index < old.state.index) return;
    job = job.copyWith(state: state, uri: e['uri'] as String?, name: e['name'] as String?, message: e['message'] as String?);
    _jobs[id] = job;
    if (state == JobState.done) _lastDone = job;
    if (state == JobState.failed) _lastError = job.message ?? 'Processing failed';
    // Keep the map small: finished jobs beyond the latest few are dropped.
    final finished = _jobs.values.where((j) => !j.active).toList()..sort((a, b) => a.id.compareTo(b.id));
    for (final j in finished.take((finished.length - 5).clamp(0, finished.length))) {
      _jobs.remove(j.id);
    }
    notifyListeners();
  }
}
