import os
import time
import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from smartclass_msgs.msg import BondingBoxCv
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from smartclass_cv.remote_client import RemoteError, RemoteModel, RttStats


class ActivityNode(Node):
    def __init__(self):
        super().__init__('ActivityNode')

        # ===== PARAMETER =====
        # mode 'local'  -> model dijalankan di node ini (perilaku lama)
        # mode 'remote' -> model dipanggil di server lewat HTTP
        self.declare_parameter('mode', 'local')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000')
        self.declare_parameter('timeout_s', 1.0)
        self.declare_parameter('image_format', 'jpg')   # 'jpg' atau 'png'
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('model_path', '')        # kosong -> models/best.onnx

        self.mode = self.get_parameter('mode').value
        if self.mode not in ('local', 'remote'):
            raise ValueError("parameter 'mode' harus 'local' atau 'remote'")

        # ===== ROS =====
        self.sub = self.create_subscription(
            BondingBoxCv,
            '/cv/bondingbox/person',
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(
            BondingBoxCv,
            '/cv/bondingbox/activity',
            10
        )

        self.bridge = CvBridge()

        # ===== MODEL =====
        if self.mode == 'local':
            from ultralytics import YOLO  # hanya dibutuhkan di mode local

            model_path = self.get_parameter('model_path').value
            if not model_path:
                # NOTE: perilaku lama memuat models/best.onnx. Putuskan apakah
                # seharusnya activity.onnx, lalu ubah di sini / lewat parameter.
                model_path = os.path.join(
                    get_package_share_directory('smartclass_cv'),
                    'models', 'best.onnx')
            self.get_logger().info(f"📦 Model path: {model_path}")
            self.model = YOLO(model_path)
        else:
            url = self.get_parameter('server_url').value.rstrip('/') + '/activity'
            self.remote = RemoteModel(
                url,
                timeout=self.get_parameter('timeout_s').value,
                image_format=self.get_parameter('image_format').value,
                jpeg_quality=self.get_parameter('jpeg_quality').value,
            )
            self.stats = RttStats(self.get_logger(), 'activity')
            self.get_logger().info(f"🌐 Remote model: {url}")

        self.class_names = ['belajar', 'tidur', 'main_hp', 'berdiri', 'berjalan']
        self.alert_classes = ['tidur', 'main_hp']

        self.get_logger().info(
            f"🟢 SMART CLASSROOM ROS NODE AKTIF (mode={self.mode})")

        self.last_log_time = time.time()

    # ------------------------------------------------------------------
    # Inferensi: hasilnya selalu list[(cls, conf, x1, y1, x2, y2)]
    # ------------------------------------------------------------------
    def _infer_local(self, frame):
        results = self.model(frame, conf=0.4, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.append((int(box.cls[0]), float(box.conf[0]), x1, y1, x2, y2))
        return out

    def _infer_remote(self, frame):
        """Return list deteksi, atau None kalau server gagal."""
        try:
            res = self.remote.infer(frame)
        except RemoteError as e:
            self.stats.add_fail(e)
            return None
        self.stats.add(res)
        return [
            (int(d['cls']), float(d['conf']),
             int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2']))
            for d in res['detections']
        ]

    def image_callback(self, msg: BondingBoxCv):
        frame = self.bridge.imgmsg_to_cv2(msg.cropped_image, desired_encoding='bgr8')

        if self.mode == 'local':
            detections = self._infer_local(frame)
        else:
            detections = self._infer_remote(frame)
            if detections is None:
                return  # server gagal: lewati frame ini

        counts = {c: 0 for c in self.class_names}
        phone_detected = False
        phone_center = None
        person_boxes = []

        # ===== PASS 1: HP & PERSON =====  (aturan tetap di node)
        for cls, conf, x1, y1, x2, y2 in detections:
            label = self.class_names[cls]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # PAKSA HP
            if label == 'main_hp' or (label == 'belajar' and (x2-x1) < 300 and (y2-y1) < 500):
                phone_detected = True
                phone_center = (cx, cy)
                counts['main_hp'] += 1
            else:
                person_boxes.append((label, x1, y1, x2, y2, cx, cy, conf))

        # ===== PASS 2: ORANG DEKAT HP =====
        for label, x1, y1, x2, y2, cx, cy, conf in person_boxes:
            if phone_detected:
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(phone_center))
                if dist < 400:
                    label = 'main_hp'
            counts[label] += 1

        # ===== FORMAT HASIL =====
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        result_text = timestamp.join(
            [f"{k}:{v}" for k, v in counts.items() if v > 0]
        )

        # ===== PUBLISH =====
        msg.label = result_text
        self.pub.publish(msg)

        self.get_logger().info(f"📤 {result_text}")


def main():
    rclpy.init()
    node = ActivityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
