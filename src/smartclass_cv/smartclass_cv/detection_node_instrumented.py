import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from vision_msgs.msg import Pose2D
from vision_msgs.msg import BoundingBox2D
from std_msgs.msg import String
from smartclass_msgs.msg import FrameCv
from smartclass_msgs.msg import BondingBoxCv
import datetime
import json
import cv2
import numpy as np
import os
import time
import csv

from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


# Konfigurasi ByteTrack khusus (ditulis otomatis ke ~/.ros/smartclass saat node start).
# - conf detektor dibuat rendah supaya asosiasi tahap-2 ByteTrack (deteksi skor rendah)
#   bisa menyambung track yang sedang ada, sehingga box tidak putus-putus.
# - new_track_thresh lebih tinggi supaya deteksi lemah tidak membuat ID baru.
# - track_buffer lebih besar supaya ID bertahan saat objek tertutup sebentar.
TRACKER_YAML = """tracker_type: bytetrack
track_high_thresh: 0.4
track_low_thresh: 0.1
new_track_thresh: 0.45
track_buffer: 45
match_thresh: 0.8
fuse_score: True
"""


class DetectionNodeInstrumented(Node):
    def __init__(self):
        # Nama node disamakan dengan yang berjalan di sistem (/DetectionNode)
        super().__init__('DetectionNode')
        self.bridge = CvBridge()

        # === PARAMETER (bisa diubah lewat --ros-args -p nama:=nilai) ===
        self.declare_parameter('model_path', '')     # kosong -> models/best.onnx
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.5)
        self.declare_parameter('alpha', 0.4)         # smoothing dasar saat objek diam (1.0 = tanpa smoothing)
        self.declare_parameter('fast_move_ratio', 0.15)  # pergeseran pusat (relatif ukuran box) yang dianggap gerakan cepat
        self.declare_parameter('timing_log_path', '~/ros2_ws/detection_timing_log.csv')

        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf_thresh = float(self.get_parameter('conf').value)
        self.iou_thresh = float(self.get_parameter('iou').value)
        self.alpha = float(self.get_parameter('alpha').value)
        self.fast_move_ratio = max(1e-3, float(self.get_parameter('fast_move_ratio').value))

        # Depth 1: selalu proses frame terbaru, frame basi dibuang (tidak menumpuk)
        self.sub = self.create_subscription(
            FrameCv, '/cv/frame', self.callback, 1
        )

        self.pub_counting = self.create_publisher(String, '/cv/counting', 10)
        self.pub_delete_frame = self.create_publisher(String, '/cv/frame/delete', 20)
        self.pub_person = self.create_publisher(BondingBoxCv, '/cv/bondingbox/person', 10)
        self.pub_nonperson = self.create_publisher(BondingBoxCv, '/cv/bondingbox/nonperson', 10)

        model_path = self.get_parameter('model_path').value
        if not model_path:
            pkg_path = get_package_share_directory('smartclass_cv')
            model_path = os.path.join(pkg_path, 'models', 'best.onnx')
        self.get_logger().info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path, task='detect')

        self.tracker_cfg = self._ensure_tracker_cfg()
        self.get_logger().info(f"Tracker config: {self.tracker_cfg}")
        self.get_logger().info(
            f"Detection Node started (imgsz={self.imgsz}, conf={self.conf_thresh}, "
            f"iou={self.iou_thresh}, alpha={self.alpha})"
        )

        # === SMOOTHING PER TRACK_ID ===
        self.prev_ttl_frames = 30   # hapus state track yang sudah lama tidak terlihat
        self.prev = {}              # (label, track_id) -> {"box": (x1,y1,x2,y2), "last_seen": n}

        # === LOGGING TIMING ===
        self.csv_path = os.path.expanduser(self.get_parameter('timing_log_path').value)
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame_id', 'frame_sequence',
            't_recv_to_cv2_ms', 't_inference_ms', 't_postprocess_ms',
            't_publish_ms', 't_total_ms'
        ])

        self.frame_counter = 0
        self.last_recv_time = None

    @staticmethod
    def _ensure_tracker_cfg():
        cfg_dir = os.path.expanduser('~/.ros/smartclass')
        os.makedirs(cfg_dir, exist_ok=True)
        path = os.path.join(cfg_dir, 'bytetrack_custom.yaml')
        with open(path, 'w') as f:
            f.write(TRACKER_YAML)
        return path

    def _smooth_box(self, label, tid, box):
        """EMA adaptif per (label, track_id).

        Objek diam  -> alpha dasar (halus, box stabil).
        Objek cepat -> alpha naik mendekati 1.0 (box langsung ikut, tidak tertinggal).
        """
        x1, y1, x2, y2 = box
        entry = self.prev.get((label, tid))
        if entry is not None:
            px1, py1, px2, py2 = entry["box"]
            # seberapa jauh pusat box bergeser dibanding ukuran box
            diag = max(((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5, 1.0)
            dx = ((x1 + x2) - (px1 + px2)) / 2.0
            dy = ((y1 + y2) - (py1 + py2)) / 2.0
            ratio = ((dx ** 2 + dy ** 2) ** 0.5) / diag
            a = self.alpha + (1.0 - self.alpha) * min(1.0, ratio / self.fast_move_ratio)
            x1 = a * x1 + (1 - a) * px1
            y1 = a * y1 + (1 - a) * py1
            x2 = a * x2 + (1 - a) * px2
            y2 = a * y2 + (1 - a) * py2
        self.prev[(label, tid)] = {
            "box": (x1, y1, x2, y2),
            "last_seen": self.frame_counter,
        }
        return x1, y1, x2, y2

    def _cleanup_prev(self):
        """Buang state track lama supaya dictionary tidak membesar terus."""
        limit = self.frame_counter - self.prev_ttl_frames
        stale = [k for k, v in self.prev.items() if v["last_seen"] < limit]
        for k in stale:
            del self.prev[k]

    def callback(self, msg: FrameCv):
        t_start = time.perf_counter()

        if self.last_recv_time is not None:
            gap = t_start - self.last_recv_time
        else:
            gap = 0.0
        self.last_recv_time = t_start

        if msg.frame_sampling.data > 0:
            frame_id = msg.frame_id

            # === TAHAP 1: konversi image ROS -> OpenCV ===
            t0 = time.perf_counter()
            frame = self.bridge.imgmsg_to_cv2(msg.frame_image, desired_encoding='bgr8')
            t1 = time.perf_counter()

            # === TAHAP 2: YOLO INFERENCE + TRACKING ===
            results = self.model.track(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                persist=True,
                tracker=self.tracker_cfg,
                verbose=False,
            )
            t2 = time.perf_counter()

            # === TAHAP 3: POSTPROCESSING (smoothing, clamp, hitung) ===
            H, W = frame.shape[:2]
            detections = []
            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue
                ids = r.boxes.id.int().cpu().tolist()
                for box, cls, conf, tid in zip(
                    r.boxes.xyxy, r.boxes.cls, r.boxes.conf, ids
                ):
                    label = self.model.names[int(cls)]
                    raw = tuple(float(v) for v in box.cpu().numpy())

                    # smoothing ringan per track_id
                    x1, y1, x2, y2 = self._smooth_box(label, tid, raw)

                    # clamp supaya crop tidak salah
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(W, int(x2))
                    y2 = min(H, int(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    detections.append({
                        "label": label,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "track_id": int(tid),
                    })

            self._cleanup_prev()

            counts = {}
            for det in detections:
                label = det["label"]
                counts[label] = counts.get(label, 0) + 1

            out = String()
            out.data = json.dumps({"frame_id": frame_id, "counts": counts})
            t3 = time.perf_counter()

            # === TAHAP 4: PUBLISH ===
            self.pub_counting.publish(out)

            for box in detections:
                msg_pub = BondingBoxCv()
                msg_pub.frame_id = frame_id
                msg_pub.frame_sampling = msg.frame_sampling
                msg_pub.frame_sequence.data = msg.frame_sequence.data
                # ID dari tracker (konsisten antar frame), bukan counter urutan
                msg_pub.bondingbox_id.data = box["track_id"]
                msg_pub.bondingbox = self.bondingbox2d(*box["bbox"])
                # crop tetap dikirim: dibutuhkan activity_node
                croppedimage = frame[box["bbox"][1]:box["bbox"][3], box["bbox"][0]:box["bbox"][2]]
                msg_pub.cropped_image = self.bridge.cv2_to_imgmsg(croppedimage, encoding="bgr8")
                msg_pub.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                label_lower = box["label"].strip().lower()
                if label_lower == 'person':
                    msg_pub.label = 'Person'
                    msg_pub.confident_level = box["confidence"]
                    self.pub_person.publish(msg_pub)
                elif label_lower == 'chair':
                    msg_pub.label = 'Chair'
                    msg_pub.confident_level = box["confidence"]
                    self.pub_nonperson.publish(msg_pub)

            t4 = time.perf_counter()

            # === CATAT SEMUA WAKTU (dalam milidetik) ===
            t_cv2 = (t1 - t0) * 1000
            t_infer = (t2 - t1) * 1000
            t_post = (t3 - t2) * 1000
            t_pub = (t4 - t3) * 1000
            t_total = (t4 - t_start) * 1000

            self.csv_writer.writerow([
                frame_id, msg.frame_sequence.data,
                round(t_cv2, 2), round(t_infer, 2),
                round(t_post, 2), round(t_pub, 2), round(t_total, 2)
            ])
            self.csv_file.flush()

            self.frame_counter += 1
            if self.frame_counter % 10 == 0:
                self.get_logger().info(
                    f"[Frame #{self.frame_counter}] gap_antar_frame={gap*1000:.1f}ms | "
                    f"cv2={t_cv2:.1f}ms | inference={t_infer:.1f}ms | "
                    f"postproc={t_post:.1f}ms | publish={t_pub:.1f}ms | TOTAL={t_total:.1f}ms"
                )

    def bondingbox2d(self, x1, y1, x2, y2):
        min_x, min_y = min(x1, x2), min(y1, y2)
        max_x, max_y = max(x1, x2), max(y1, y2)
        bbox = BoundingBox2D()
        center = Pose2D()
        center.position.x = float((min_x + max_x) / 2)
        center.position.y = float((min_y + max_y) / 2)
        center.theta = 0.0
        bbox.center = center
        bbox.size_x = float(max_x - min_x)
        bbox.size_y = float(max_y - min_y)
        return bbox

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f"Timing log tersimpan di: {self.csv_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNodeInstrumented()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()