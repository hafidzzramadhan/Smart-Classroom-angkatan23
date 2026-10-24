"""Measure the existing activity callback; CSV logging is outside total timing.

One callback is one person crop, not necessarily one full camera frame.
Publish duration measures publish() only, not transport or delivery.
The original per-callback ROS log is retained and timed separately.
"""
import argparse
import csv
import datetime
import hashlib
import inspect
import time
from pathlib import Path

import numpy as np
import rclpy
from smartclass_cv.activity_node import ActivityNode

EXPECTED_SHA256 = '20a97f5a5371f333c0dde63f6c38a65b1054fe7c7b3ed9ac79d94c97a9aa5c67'
FIELDS = [
    'callback_index', 'frame_id', 'camera_id', 'frame_sequence',
    'bondingbox_id', 'sequence_delta', 'input_width', 'input_height',
    'callback_start_monotonic_ns', 'callback_end_monotonic_ns',
    'callback_gap_ms', 'n_model_boxes', 'n_published',
    't_cv2_ms', 't_model_call_ms', 't_postprocess_ms', 't_publish_ms',
    't_original_log_ms', 't_total_ms', 'status', 'error', 'result_label',
]

def ms(a, b):
    return '' if a is None or b is None else (b - a) / 1e6

class ActivityBaselineProbe(ActivityNode):
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

    def image_callback(self, msg):
        t0 = time.monotonic_ns()
        t1 = t2 = t3 = t4 = t5 = None
        detected = published = 0
        status, error, result_text = 'ok', '', ''
        try:
            frame = self.bridge.imgmsg_to_cv2(msg.cropped_image, desired_encoding='bgr8')
            t1 = time.monotonic_ns()
            results = self.model(frame, conf=0.4, verbose=False)[0]
            t2 = time.monotonic_ns()
            counts = {c: 0 for c in self.class_names}
            phone_detected = False
            phone_center = None
            person_boxes = []
            for box in results.boxes:
                detected += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.class_names[cls]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if label == 'main_hp' or (label == 'belajar' and (x2-x1) < 300 and (y2-y1) < 500):
                    phone_detected = True
                    phone_center = (cx, cy)
                    counts['main_hp'] += 1
                else:
                    person_boxes.append((label, x1, y1, x2, y2, cx, cy, conf))
            for label, x1, y1, x2, y2, cx, cy, conf in person_boxes:
                if phone_detected:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array(phone_center))
                    if dist < 400:
                        label = 'main_hp'
                counts[label] += 1
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            result_text = timestamp.join([f'{k}:{v}' for k, v in counts.items() if v > 0])
            msg.label = result_text
            t3 = time.monotonic_ns()
            self.pub.publish(msg)
            t4 = time.monotonic_ns()
            published = 1
            self.get_logger().info(f'📤 {result_text}')
            t5 = time.monotonic_ns()
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
        end = time.monotonic_ns()
        seq = msg.frame_sequence.data
        delta = '' if self._previous_sequence is None else seq - self._previous_sequence
        gap = ms(self._previous_start, t0)
        self._previous_start, self._previous_sequence = t0, seq
        self._count += 1
        self._writer.writerow([
            self._count, msg.frame_id, msg.camera_id, seq,
            msg.bondingbox_id.data, delta,
            msg.cropped_image.width, msg.cropped_image.height,
            t0, end, gap, detected, published,
            ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4),
            ms(t4, t5), ms(t0, end), status, error, result_text,
        ])
        if self._count % 10 == 0 or status == 'error':
            self._file.flush()
        if status == 'error':
            # Preserve the original callback's fail-fast behavior after recording.
            raise RuntimeError(error)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not (0 < args.seconds < float('inf')):
        parser.error('--seconds must be finite and positive')
    original = Path(inspect.getsourcefile(ActivityNode)).resolve()
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f'Activity source differs: {original}\nSHA256: {digest}')
    output = (args.output or Path.home() / 'smartclass_ws' / 'experiments' /
              f'activity_pc_hikvision_{time.monotonic_ns()}.csv').expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Original activity source: {original}\nSHA256: {digest}\nCSV: {output}', flush=True)
    rclpy.init(args=[])
    node = None
    try:
        init_start = time.monotonic_ns()
        node = ActivityBaselineProbe(output)
        print(f'Node initialization: {ms(init_start, time.monotonic_ns()):.2f} ms', flush=True)
        print('Model: /home/hafidz/smartclass_ws/src/smartclass_cv/models/best.onnx', flush=True)
        start = time.monotonic_ns()
        deadline = start + int(args.seconds * 1e9)
        print(f'Recording about {args.seconds:g} seconds; each row is one person crop.', flush=True)
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
