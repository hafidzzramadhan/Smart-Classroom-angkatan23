"""Time the existing face pipeline without changing its model or ROS interfaces.

Run in the sourced smartclass workspace. Durations exclude CSV/progress logging.
Publish timing covers the publish() call, not delivery to another node.
"""

import argparse
import csv
import hashlib
import inspect
import time
from datetime import datetime
from pathlib import Path

import rclpy
from smartclass_cv.face_node import FaceNode
from smartclass_msgs.msg import BondingBoxCv


EXPECTED_FACE_SHA256 = (
    "190f71d4b08617cbc23d24c7a8811ce56725db91f552d2f261882cf2a9ff6cb7"
)
FIELDS = [
    "callback_index", "frame_id", "camera_id", "frame_sequence",
    "sequence_gap", "frame_sampling", "input_width", "input_height",
    "callback_start_monotonic_ns", "callback_end_monotonic_ns",
    "callback_gap_ms", "n_boxes_detected", "n_boxes_published",
    "t_cv2_ms", "t_inference_ms", "t_prepare_ms", "t_publish_ms",
    "t_total_ms", "status", "error",
]


def elapsed_ms(start, end):
    return "" if start is None or end is None else (end - start) / 1e6


class FaceBaselineProbe(FaceNode):
    def __init__(self, output):
        self._callback_index = 0
        self._previous_start = None
        self._previous_sequence = None
        self._timing_file = output.open("x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._timing_file)
        self._writer.writerow(FIELDS)
        self._timing_file.flush()
        try:
            # Reuse the original model loading, subscription, publisher and QoS.
            super().__init__()
        except BaseException:
            self._timing_file.close()
            raise

    def listener_callback(self, msg):
        t0 = time.monotonic_ns()
        t1 = t2 = None
        publish_ns = 0
        detected = published = 0
        status = "sampling_skipped"
        error = ""
        try:
            if msg.frame_sampling.data > 0:
                status = "ok"
                frame = self.br.imgmsg_to_cv2(
                    msg.frame_image, desired_encoding="bgr8"
                )
                t1 = time.monotonic_ns()
                # Preserve the original call, including default verbosity/device.
                results = self.model(frame)
                t2 = time.monotonic_ns()
                for result in results:
                    detections = result.boxes.xyxy
                    bboxes = (
                        detections.cpu().numpy()
                        if hasattr(detections, "cpu") else detections
                    )
                    detected += len(bboxes)
                    bondingboxid = 1
                    for bbox in bboxes:
                        x1, y1, x2, y2 = map(int, bbox)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cropped_image = frame[y1:y2, x1:x2]
                        cropped_ros_image = self.br.cv2_to_imgmsg(
                            cropped_image, encoding="bgr8"
                        )
                        cropped_msg = BondingBoxCv()
                        cropped_msg.camera_id = msg.camera_id
                        cropped_msg.frame_id = msg.frame_id
                        cropped_msg.frame_sampling = msg.frame_sampling
                        cropped_msg.frame_sequence.data = msg.frame_sequence.data
                        cropped_msg.bondingbox_id.data = bondingboxid
                        cropped_msg.cropped_image = cropped_ros_image
                        cropped_msg.bondingbox = self.bondingbox2d(x1, y1, x2, y2)
                        cropped_msg.timestamp = timestamp
                        before_publish = time.monotonic_ns()
                        try:
                            self.pub.publish(cropped_msg)
                        finally:
                            publish_ns += time.monotonic_ns() - before_publish
                        published += 1
                        bondingboxid += 1
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"❌ Error: {exc}")
        t3 = time.monotonic_ns()

        sequence = msg.frame_sequence.data
        sequence_gap = (
            "" if self._previous_sequence is None
            else sequence - self._previous_sequence
        )
        callback_gap = elapsed_ms(self._previous_start, t0)
        self._previous_start = t0
        self._previous_sequence = sequence
        self._callback_index += 1
        prepare_ms = "" if t2 is None else (t3 - t2 - publish_ns) / 1e6
        self._writer.writerow([
            self._callback_index, msg.frame_id, msg.camera_id, sequence,
            sequence_gap, msg.frame_sampling.data,
            msg.frame_image.width, msg.frame_image.height,
            t0, t3, callback_gap, detected, published,
            elapsed_ms(t0, t1), elapsed_ms(t1, t2), prepare_ms,
            publish_ns / 1e6, elapsed_ms(t0, t3), status, error,
        ])
        if self._callback_index % 10 == 0:
            self._timing_file.flush()
            self.get_logger().info(
                f"[Callback #{self._callback_index}] seq={sequence} "
                f"boxes={detected} published={published} "
                f"total={elapsed_ms(t0, t3):.2f} ms status={status}"
            )

    def close_timing_log(self):
        self._timing_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (0 < args.seconds < float("inf")):
        parser.error("--seconds must be a finite positive number")

    # Refuse to silently instrument a different installed implementation.
    original = Path(inspect.getsourcefile(FaceNode)).resolve()
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != EXPECTED_FACE_SHA256:
        raise SystemExit(
            f"Face source differs from the reviewed baseline: {original}\n"
            f"SHA256: {digest}"
        )
    output = args.output or (
        Path.home() / "smartclass_ws" / "experiments"
        / f"face_pc_hikvision_{time.monotonic_ns()}.csv"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Original face source: {original}", flush=True)
    print(f"Original SHA256: {digest}", flush=True)
    print(f"CSV: {output}", flush=True)

    rclpy.init(args=[])
    node = None
    try:
        init_start = time.monotonic_ns()
        node = FaceBaselineProbe(output)
        print(
            f"Node initialization including model loading: "
            f"{elapsed_ms(init_start, time.monotonic_ns()):.2f} ms",
            flush=True,
        )
        print(f"Model: {node.model_path}", flush=True)
        start = time.monotonic_ns()
        deadline = start + int(args.seconds * 1e9)
        print(f"Recording for about {args.seconds:g} seconds.", flush=True)
        try:
            while rclpy.ok() and time.monotonic_ns() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            print("Stopped early by Ctrl+C.", flush=True)
        print(
            f"Recorded {node._callback_index} callbacks over "
            f"{(time.monotonic_ns() - start) / 1e9:.3f} monotonic seconds.",
            flush=True,
        )
    finally:
        if node is not None:
            node.close_timing_log()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"CSV saved: {output}", flush=True)


if __name__ == "__main__":
    main()
