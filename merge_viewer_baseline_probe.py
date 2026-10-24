#!/usr/bin/env python3
"""PC pilot instrumentation; source behavior is preserved, overhead is not zero.

Usage: python merge_viewer_baseline_probe.py --check
       python merge_viewer_baseline_probe.py merge --run-id pilot01 --seconds 30
       python merge_viewer_baseline_probe.py viewer --run-id pilot01 --seconds 40

Run both processes in the SAME WSL instance/boot. Start viewer first.
Do not run another Merge or viewer concurrently. Start Frame and Gesture only.
--seconds starts after node creation; shutdown may exceed it by a callback.

This probe subclasses the existing Merge; it does not fix cache/fusion behavior,
disable original disk writes/printing, or enable live video. Annotated output
header.frame_id is an instrumentation token, NOT an optical coordinate frame.
Use only in this isolated pilot with this viewer, not a production geometry chain.
JPEG payload, output topics, QoS and fusion logic remain the original ones.

CSV clocks: time.monotonic_ns(). image_publish.start_ns is immediately before
calling publish; viewer.end_ns is after imshow/waitKey returns (software only).
Join rows on run_id AND output_id. The delta includes publish/transport/queue/
decode/viewer work; it is NOT pure network time, camera E2E, or monitor light.
Repeated source sequence numbers can have distinct output IDs.

Rows buffer in memory and flush OUTSIDE callbacks every 5 s and at shutdown.
Nested event bookkeeping may be inside outer callback durations. Overhead is
not isolated; abrupt termination can lose the last buffered interval.
Viewer gap is an update interval, not stalling without a predefined threshold.
No camera-acquisition timestamps or CV accuracy are measured by this probe.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time
import traceback

EXPECTED = 'acfbac85ca00d717346b3adb9a081714bea1860cc4980ef9d9d0c8737fb87404'
FIELDS = ['run_id', 'event', 'output_id', 'frame_sequence', 'start_ns',
          'end_ns', 'duration_ms', 'status', 'details_json']


def source_info():
    spec = importlib.util.find_spec('smartclass_cv.merge_node_test')
    if spec is None or spec.origin is None:
        raise RuntimeError('Source Merge tidak ditemukan. Source lingkungan ROS/workspace dahulu.')
    path = Path(spec.origin)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise RuntimeError(f'Versi Merge berubah: {digest}. Review source sebelum menjalankan probe.')
    return {'module_path': str(path), 'source_path': str(path.resolve()), 'sha256': digest}


class Ledger:
    def __init__(self, args, source):
        directory = Path(args.output_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        stem = f'{args.run_id}_{args.mode}_{time.monotonic_ns()}'
        self.run_id = args.run_id
        self.path = directory / (stem + '.csv')
        self.file = self.path.open('x', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self.rows = []
        self.count = 0
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        metadata = {'run_id': args.run_id, 'mode': args.mode, 'seconds_requested': args.seconds,
                    'source': source, 'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'boot_id': boot, 'clock': 'time.monotonic_ns',
                    'clock_info': vars(time.get_clock_info('monotonic')),
                    'fusion_behavior': 'original; live disabled; original disk writes/prints retained',
                    'measurement_limit': 'software Merge-to-viewer; no camera E2E or physical display measurement',
                    'logging': 'memory rows; CSV flush outside callbacks every 5 s; overhead not isolated'}
        with (directory / (stem + '.metadata.json')).open('x', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print('CSV:', self.path, flush=True)

    def emit(self, event, start, end, seq='', output_id='', status='ok', **details):
        self.rows.append(dict(run_id=self.run_id, event=event, output_id=output_id,
                              frame_sequence=seq, start_ns=start, end_ns=end,
                              duration_ms=(end-start)/1e6, status=status,
                              details_json=json.dumps(details, ensure_ascii=False)))
        self.count += 1

    def flush(self):
        self.writer.writerows(self.rows)
        self.rows.clear()
        self.file.flush()

    def close(self):
        self.flush()
        self.file.close()


def make_merge(ledger):
    from smartclass_cv.merge_node_test import BondingBoxMerger

    class ImagePublisherProbe:
        def __init__(self, original, owner):
            self.original = original
            self.owner = owner

        def publish(self, msg):
            owner = self.owner
            if owner.active_sequence is None:
                raise RuntimeError('Unexpected image publish outside exact fusion; review source.')
            owner.output_count += 1
            output_id = str(owner.output_count)
            seq = owner.active_sequence
            msg.header.frame_id = f'baseline_probe/{ledger.run_id}/{output_id}/{seq}'
            start = time.monotonic_ns()
            status = 'ok'
            try:
                self.original.publish(msg)
            except Exception:
                status = 'error'
                raise
            finally:
                end = time.monotonic_ns()
                ledger.emit('image_publish', start, end, seq, output_id, status,
                            bytes=len(msg.data), meaning='publish call; not subscriber receipt')

    class MergeProbe(BondingBoxMerger):
        def __init__(self):
            self.active_sequence = None
            self.output_count = 0
            super().__init__()
            self.pub_foxglove = ImagePublisherProbe(self.pub_foxglove, self)

        def frame_callback(self, msg):
            seq = int(msg.frame_sequence.data)
            before = set(self.frame_cache)
            start = time.monotonic_ns()
            status = 'ok'
            try:
                super().frame_callback(msg)
            except Exception:
                status = 'error'
                raise
            finally:
                end = time.monotonic_ns()
                ledger.emit('frame_callback', start, end, seq, status=status,
                            cached=seq in self.frame_cache, cache_count=len(self.frame_cache),
                            evicted_sequences=sorted(before-set(self.frame_cache)),
                            pending_result_keys=len(self.QBox))

        def _store_ai_result(self, msg, name):
            seq = int(msg.frame_sequence.data)
            before = set(self.QBox)
            start = time.monotonic_ns()
            status = 'ok'
            try:
                super()._store_ai_result(msg, name)
            except Exception:
                status = 'error'
                raise
            finally:
                end = time.monotonic_ns()
                ledger.emit('ai_callback', start, end, seq, status=status, branch=name,
                            box_id=int(msg.bondingbox_id.data), label=msg.label,
                            cache_count=len(self.frame_cache), pending_result_keys=len(self.QBox),
                            removed_pending_keys=sorted(before-set(self.QBox)))

        def _publish_exact_fusion(self, seq):
            previous = self.active_sequence
            self.active_sequence = seq
            cached = seq in self.frame_cache
            n_boxes = len(self.QBox.get(str(seq), []))
            count_before = self.output_count
            start = time.monotonic_ns()
            success = False
            status = 'ok'
            try:
                success = super()._publish_exact_fusion(seq)
                return success
            except Exception:
                status = 'error'
                raise
            finally:
                end = time.monotonic_ns()
                output_id = str(self.output_count) if self.output_count > count_before else ''
                ledger.emit('fusion_attempt', start, end, seq, output_id, status,
                            cached_at_start=cached, boxes_at_start=n_boxes, success=success)
                self.active_sequence = previous

    return MergeProbe()


def make_viewer(ledger):
    import cv2
    import numpy as np
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    class ViewerProbe(Node):
        def __init__(self):
            super().__init__('merge_viewer_probe')
            self.last_update = None
            self.last_sequence = None
            self.sub = self.create_subscription(CompressedImage, '/cv/frame/image', self.callback, 10)

        def callback(self, msg):
            start = time.monotonic_ns()
            parts = msg.header.frame_id.split('/')
            if (len(parts) != 4 or parts[:2] != ['baseline_probe', ledger.run_id]
                    or not parts[2].isdigit() or not parts[3].lstrip('-').isdigit()):
                ledger.emit('viewer_unmatched', start, time.monotonic_ns(), status='unmatched',
                            header_frame_id=msg.header.frame_id)
                return
            output_id, seq = parts[2], int(parts[3])
            decode_end = None
            status = 'ok'
            displayed = False
            try:
                frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                decode_end = time.monotonic_ns()
                if frame is None:
                    status = 'decode_failed'
                else:
                    cv2.imshow('MERGE BASELINE PROBE', frame)
                    cv2.waitKey(1)
                    displayed = True
            except Exception:
                status = 'error'
                raise
            finally:
                end = time.monotonic_ns()
                ledger.emit('viewer_update', start, end, seq, output_id, status,
                            decoded_at_ns=decode_end, display_call_completed=displayed,
                            update_gap_ms=((end-self.last_update)/1e6
                                           if displayed and self.last_update is not None else None),
                            sequence_delta=(seq-self.last_sequence if self.last_sequence is not None else None),
                            bytes=len(msg.data))
                if displayed:
                    self.last_update, self.last_sequence = end, seq

    return ViewerProbe()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=['merge', 'viewer'])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output-dir', default='/home/hafidz/smartclass_ws/experiments')
    args = parser.parse_args()
    source = source_info()
    import rclpy
    import cv2
    import numpy
    from smartclass_cv.merge_node_test import BondingBoxMerger
    from smartclass_msgs.msg import FrameCv, BondingBoxCv, ResultCv
    if args.check:
        print('CHECK OK: imports dan hash source sesuai. Belum menguji ROS runtime, GUI atau timing.')
        print(json.dumps(source, indent=2))
        return
    if (not args.mode or not args.run_id or not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id)
            or not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('mode, --run-id (huruf/angka/_/-), dan --seconds positif diperlukan')
    ledger = Ledger(args, source)
    node = None
    status = 'normal_duration'
    start = time.monotonic_ns()
    try:
        rclpy.init()
        node = make_merge(ledger) if args.mode == 'merge' else make_viewer(ledger)
        start = time.monotonic_ns()
        deadline = start + int(args.seconds*1e9)
        last_flush = start
        print(f'READY {args.mode}; run_id={args.run_id}; seconds={args.seconds}', flush=True)
        while rclpy.ok() and time.monotonic_ns() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic_ns()
            if now-last_flush >= 5_000_000_000:
                ledger.flush()
                last_flush = now
        if not rclpy.ok():
            status = 'context_shutdown'
    except KeyboardInterrupt:
        status = 'keyboard_interrupt'
    except Exception:
        status = 'error'
        traceback.print_exc()
        raise
    finally:
        try:
            if node is not None:
                node.destroy_node()
            if args.mode == 'viewer':
                cv2.destroyAllWindows()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            status += '_cleanup_error'
            traceback.print_exc()
        finally:
            ledger.emit('run_end', start, time.monotonic_ns(), status=status)
            ledger.close()
            print(f'END status={status}; rows={ledger.count}; CSV={ledger.path}', flush=True)


if __name__ == '__main__':
    main()
