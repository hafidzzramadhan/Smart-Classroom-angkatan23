import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from smartclass_msgs.msg import BondingBoxCv
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from smartclass_cv.remote_client import RemoteError, RemoteModel, RttStats


class Expression(Node):
    def __init__(self):
        super().__init__('Expression')

        # =========================
        # PARAMETER
        # =========================
        # mode      : 'local'  -> model dijalankan di node ini (perilaku lama)
        #             'remote' -> model dipanggil di server lewat HTTP
        self.declare_parameter('mode', 'local')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000')
        self.declare_parameter('timeout_s', 1.0)
        self.declare_parameter('image_format', 'jpg')   # 'jpg' atau 'png'
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('model_path', '')        # kosong -> models/best.pt

        self.mode = self.get_parameter('mode').value
        if self.mode not in ('local', 'remote'):
            raise ValueError("parameter 'mode' harus 'local' atau 'remote'")

        # =========================
        # MODEL
        # =========================
        if self.mode == 'local':
            from ultralytics import YOLO  # hanya dibutuhkan di mode local

            model_path = self.get_parameter('model_path').value
            if not model_path:
                model_path = os.path.join(
                    get_package_share_directory('smartclass_cv'),
                    'models', 'best.pt')
            self.get_logger().info(f"📦 Model path: {model_path}")
            self.model = YOLO(model_path)
        else:
            url = self.get_parameter('server_url').value.rstrip('/') + '/expression'
            self.remote = RemoteModel(
                url,
                timeout=self.get_parameter('timeout_s').value,
                image_format=self.get_parameter('image_format').value,
                jpeg_quality=self.get_parameter('jpeg_quality').value,
            )
            self.stats = RttStats(self.get_logger(), 'expression')
            self.get_logger().info(f"🌐 Remote model: {url}")

        # =========================
        # ROS2 COMMUNICATION
        # =========================
        self.sub = self.create_subscription(
            BondingBoxCv,
            '/cv/bondingbox/face',
            self.listener_callback,
            10
        )

        self.pub = self.create_publisher(
            BondingBoxCv,
            '/cv/bondingbox/face_emotion',
            10
        )

        self.br = CvBridge()

        self.get_logger().info(f"✅ NODE READY - Emotion Detection (mode={self.mode})")

    # ------------------------------------------------------------------
    # Inferensi: hasilnya selalu list[(label, confidence)]
    # ------------------------------------------------------------------
    def _infer_local(self, frame):
        results = self.model(frame, conf=0.5, verbose=False)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                out.append((self.model.names[int(box.cls[0])], float(box.conf[0])))
        return out

    def _infer_remote(self, frame):
        try:
            res = self.remote.infer(frame)
        except RemoteError as e:
            self.stats.add_fail(e)
            return []
        self.stats.add(res)
        return [(d['label'], float(d['conf'])) for d in res['detections']]

    def listener_callback(self, msg: BondingBoxCv):
        try:
            # 1. ROS Image → OpenCV
            frame = self.br.imgmsg_to_cv2(msg.cropped_image)

            # 2. Inferensi (lokal atau di server)
            if self.mode == 'local':
                emotions = self._infer_local(frame)
            else:
                emotions = self._infer_remote(frame)

            print("Terima")
            for emotion, conf in emotions:
                waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"emosi - {emotion} frame {msg.frame_id} time {waktu} ")
                msg.label = emotion
                msg.confident_level = round(conf, 2)
                msg.timestamp = waktu
                self.pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"❌ Error: {e}")


def main():
    rclpy.init()
    node = Expression()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
