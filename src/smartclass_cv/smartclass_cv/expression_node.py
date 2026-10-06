import rclpy
from rclpy.node import Node
from smartclass_msgs.msg import BondingBoxCv
from smartclass_msgs.msg import ResultCv
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import sqlite3
import os
import json
from datetime import datetime

import torch
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class Expression(Node):
    def __init__(self):
        super().__init__('Expression')

        # =========================
        # PATH HANDLING (ROS2 WAY)
        # =========================
        pkg_path = get_package_share_directory('smartclass_cv')

        # Model path (read-only, from install/share)
        self.model_path = os.path.join(pkg_path, 'models', 'best.pt') 

        self.get_logger().info(f"📦 Model path: {self.model_path}") 

        # =========================
        # LOAD YOLO MODEL
        # =========================
        
        # Use the context manager to load the model safely
        #with torch.serialization.safe_globals([YOLO]):
          #  self.model = torch.load(self.model_path, map_location=torch.device('cpu')) 
            #self.model = torch.load(self.model_path)  # Memuat model .pt menggunakan PyTorch
            #self.model.eval()  # Set model ke mode evaluasi

        self.model = YOLO(self.model_path)
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

        self.get_logger().info("✅ NODE READY - Emotion Detection")

       # Verifikasi dan load model ONNX menggunakan onnxruntime
        self.verify_onnx_model()

    def verify_onnx_model(self):
        """Verifikasi model ONNX untuk memastikan itu valid dan siap digunakan."""
        try:
            # Cek jika file model ONNX valid
            onnx_model = onnx.load(self.model_path)
            onnx.checker.check_model(onnx_model)  # Verifikasi model

            # Buat session inference dengan ONNX Runtime
            ort_session = ort.InferenceSession(self.model_path)

            # Log jika model ONNX valid
            self.get_logger().info("✅ ONNX model is valid and ready for inference.")

        except Exception as e:
            self.get_logger().error(f"❌ Error verifying ONNX model: {e}")
            
    def listener_callback(self, msg: BondingBoxCv):
        try:
            # 1. ROS Image → OpenCV
            frame = self.br.imgmsg_to_cv2(msg.cropped_image)

            # 2. YOLO Emotion Inference
            results = self.model(frame, conf=0.5, verbose=False)

            emotions_detected = []
            print("Terima")
            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    emotion = self.model.names[cls_id]
                    conf = float(box.conf[0])

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
