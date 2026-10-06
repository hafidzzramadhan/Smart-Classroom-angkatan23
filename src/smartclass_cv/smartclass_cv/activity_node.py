import rclpy
from rclpy.node import Node
from smartclass_msgs.msg import BondingBoxCv
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import datetime
import numpy as np

class ActivityNode(Node):
    def __init__(self):
        super().__init__('ActivityNode')

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

        # ===== YOLO =====
        self.model = YOLO("/home/hafidz/smartclass_ws/src/smartclass_cv/models/best.onnx")

        self.class_names = ['belajar', 'tidur', 'main_hp', 'berdiri', 'berjalan']
        self.alert_classes = ['tidur', 'main_hp']

        self.get_logger().info("🟢 SMART CLASSROOM ROS NODE AKTIF")

        self.last_log_time = time.time()

    def image_callback(self, msg: BondingBoxCv):
        frame = self.bridge.imgmsg_to_cv2(msg.cropped_image, desired_encoding='bgr8')

        results = self.model(frame, conf=0.4, verbose=False)[0]

        counts = {c: 0 for c in self.class_names}
        phone_detected = False
        phone_center = None
        person_boxes = []

        # ===== PASS 1: HP & PERSON =====
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
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

