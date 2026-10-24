"""Instrument the original expression node, preserving model, QoS and output.

One row is one face crop. Model duration is the entire YOLO call.
Total includes original prints; CSV/progress logging is outside timing.
Publish timing does not measure delivery. Empty detections are recorded.
"""
import argparse
import csv
import hashlib
import inspect
import time
from datetime import datetime
from pathlib import Path

import rclpy
from smartclass_cv.expression_node import Expression

EXPECTED_SHA256 = 'b11f9004549363717b828adac6da728ac333d43ee6c94e7f5566447ef90b3c6f'
FIELDS = [
    'callback_index', 'frame_id', 'camera_id', 'frame_sequence',
    'bondingbox_id', 'sequence_delta', 'input_width', 'input_height',
    'callback_start_monotonic_ns', 'callback_end_monotonic_ns',
    'callback_gap_ms', 'n_model_boxes', 'n_published',
    't_cv2_ms', 't_model_call_ms', 't_postprocess_and_print_ms',
    't_publish_ms', 't_total_ms', 'status', 'error', 'result_labels',
]

def ms(a, b):
    return '' if a is None or b is None else (b-a)/1e6

class ExpressionBaselineProbe(Expression):
    def __init__(self, output):
        self._count = 0
        self._previous_start = self._previous_sequence = None
        self._file = output.open('x', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(FIELDS)
        self._file.flush()
        try:
            super().__init__()
        except BaseException:
            self._file.close()
            raise

    def listener_callback(self, msg):
        t0 = time.monotonic_ns()
        t1 = t2 = None
        publish_ns = 0
        detected = published = 0
        labels = []
        status, error = 'ok', ''
        try:
            frame = self.br.imgmsg_to_cv2(msg.cropped_image)
            t1 = time.monotonic_ns()
            results = self.model(frame, conf=0.5, verbose=False)
            t2 = time.monotonic_ns()
            print('Terima')
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    detected += 1
                    cls_id = int(box.cls[0])
                    emotion = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    waktu = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f'emosi - {emotion} frame {msg.frame_id} time {waktu} ')
                    msg.label = emotion
                    msg.confident_level = round(conf, 2)
                    msg.timestamp = waktu
                    before = time.monotonic_ns()
                    try:
                        self.pub.publish(msg)
                    finally:
                        publish_ns += time.monotonic_ns()-before
                    published += 1
                    labels.append(emotion)
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
            self.get_logger().error(f'❌ Error: {exc}')
        end = time.monotonic_ns()
        seq = msg.frame_sequence.data
        delta = '' if self._previous_sequence is None else seq-self._previous_sequence
        gap = ms(self._previous_start, t0)
        self._previous_start, self._previous_sequence = t0, seq
        self._count += 1
        post_ms = '' if t2 is None else (end-t2-publish_ns)/1e6
        self._writer.writerow([
            self._count, msg.frame_id, msg.camera_id, seq,
            msg.bondingbox_id.data, delta,
            msg.cropped_image.width, msg.cropped_image.height,
            t0, end, gap, detected, published,
            ms(t0, t1), ms(t1, t2), post_ms, publish_ns/1e6,
            ms(t0, end), status, error, '|'.join(labels),
        ])
        if self._count % 10 == 0 or status == 'error':
            self._file.flush()
        if self._count % 10 == 0:
            print(f'[Callback #{self._count}] seq={seq} boxes={detected} published={published} total={ms(t0,end):.2f} ms status={status}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not (0 < args.seconds < float('inf')):
        parser.error('--seconds must be finite and positive')
    original = Path(inspect.getsourcefile(Expression)).resolve()
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f'Expression source differs: {original}\nSHA256: {digest}')
    output = (args.output or Path.home() / 'smartclass_ws' / 'experiments' /
              f'expression_pc_hikvision_{time.monotonic_ns()}.csv').expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Original expression source: {original}\nSHA256: {digest}\nCSV: {output}', flush=True)
    rclpy.init(args=[])
    node = None
    try:
        init_start = time.monotonic_ns()
        node = ExpressionBaselineProbe(output)
        print(f'Node initialization: {ms(init_start,time.monotonic_ns()):.2f} ms\nModel: {node.model_path}', flush=True)
        start = time.monotonic_ns()
        deadline = start + int(args.seconds*1e9)
        print(f'Recording about {args.seconds:g} seconds; each row is one face crop.', flush=True)
        try:
            while rclpy.ok() and time.monotonic_ns() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            print('Stopped early by Ctrl+C.', flush=True)
        print(f'Recorded {node._count} callbacks over {(time.monotonic_ns()-start)/1e9:.3f} monotonic seconds.', flush=True)
    finally:
        if node is not None:
            node._file.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f'CSV saved: {output}', flush=True)

if __name__ == '__main__':
    main()
