import 'package:flutter_test/flutter_test.dart';
import 'package:klick/core/queue.dart';

Map<String, Object?> job(int id, String state, {String? uri, String? message}) =>
    {'type': 'job', 'id': id, 'state': state, 'scene': 'night', 'frames': 9, 'uri': uri, 'message': message};

void main() {
  test('starts idle', () {
    final q = ProcessingQueue();
    expect(q.busy, isFalse);
    expect(q.label, isNull);
    expect(q.activeCount, 0);
  });

  test('one job moves queued, processing, done', () {
    final q = ProcessingQueue();
    q.apply(job(1, 'queued'));
    expect(q.activeCount, 1);
    expect(q.label, 'Photo queued…');
    q.apply(job(1, 'processing'));
    expect(q.label, 'Processing photo…');
    q.apply(job(1, 'done', uri: 'content://x/1'));
    expect(q.busy, isFalse);
    expect(q.lastDone!.uri, 'content://x/1');
    expect(q.jobs.single.scene, 'night');
    expect(q.jobs.single.frames, 9);
  });

  test('the user can keep shooting while earlier photos process', () {
    final q = ProcessingQueue();
    q.apply(job(1, 'queued'));
    q.apply(job(1, 'processing'));
    q.apply(job(2, 'queued'));
    q.apply(job(3, 'queued'));
    expect(q.activeCount, 3);
    expect(q.label, 'Processing 3 photos…');
    q.apply(job(1, 'done', uri: 'a'));
    expect(q.activeCount, 2);
    expect(q.label, 'Processing 2 photos…');
  });

  test('states never move backwards', () {
    final q = ProcessingQueue();
    q.apply(job(1, 'done', uri: 'a'));
    q.apply(job(1, 'processing'));
    expect(q.jobs.single.state, JobState.done);
  });

  test('failures are reported once and clear', () {
    final q = ProcessingQueue();
    var notified = 0;
    q.addListener(() => notified++);
    q.apply(job(1, 'queued'));
    q.apply(job(1, 'failed', message: 'engine missing'));
    expect(q.busy, isFalse);
    expect(q.lastError, 'engine missing');
    q.clearError();
    expect(q.lastError, isNull);
    expect(notified, 3);
    q.clearError();
    expect(notified, 3);
  });

  test('ignores foreign and malformed events', () {
    final q = ProcessingQueue();
    q.apply({'type': 'scene'});
    q.apply({'type': 'job', 'id': 1, 'state': 'weird'});
    q.apply({'type': 'job', 'state': 'queued'});
    expect(q.jobs, isEmpty);
  });

  test('keeps only the latest finished jobs', () {
    final q = ProcessingQueue();
    for (var i = 1; i <= 12; i++) {
      q.apply(job(i, 'queued'));
      q.apply(job(i, 'done', uri: 'u$i'));
    }
    expect(q.jobs.length, 5);
    expect(q.lastDone!.id, 12);
  });
}
