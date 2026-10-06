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
import time  # <-- untuk pengukuran presisi (monotonic, bukan wall-clock)
import csv   # <-- untuk simpan hasil pengukuran ke file, dianalisis nanti

from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class DetectionNodeInstrumented(Node):
    def __init__(self):
        super().__init__('DetectionNodeInstrumented')
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            FrameCv, '/cv/frame', self.callback, 10
        )

        self.pub_counting = self.create_publisher(String, '/cv/counting', 10)
        self.pub_delete_frame = self.create_publisher(String, '/cv/frame/delete', 20)
        self.pub_person = self.create_publisher(BondingBoxCv, '/cv/bondingbox/person', 10)
        self.pub_nonperson = self.create_publisher(BondingBoxCv, '/cv/bondingbox/nonperson', 10)

        pkg_path = get_package_share_directory('smartclass_cv')
        model_path = os.path.join(pkg_path, 'models', 'best.onnx')
        self.get_logger().info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path, task='detect')
        self.get_logger().info("Detection Node (INSTRUMENTED) Started")

        # === SETUP LOGGING TIMING ===
        self.csv_path = os.path.expanduser('~/smartclass_ws/detection_timing_log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame_id', 'frame_sequence',
            't_recv_to_cv2_ms', 't_inference_ms', 't_postprocess_ms',
            't_publish_ms', 't_total_ms'
        ])

        # Rolling counter, biar kita juga bisa lihat rata-rata tiap N frame di terminal
        self.frame_counter = 0
        self.last_recv_time = None  # buat ukur interval ANTAR frame yang masuk callback

    def callback(self, msg: FrameCv):
        t_start = time.perf_counter()

        # --- Ukur interval antar frame yang BENERAN sampai ke callback ini ---
        # (kalau ini jauh lebih lambat dari interval publish frame_node, artinya ada frame di-drop di queue)
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

            # === TAHAP 2: YOLO INFERENCE (dugaan bottleneck utama) ===
            results = self.model(frame, verbose=False)
            t2 = time.perf_counter()

            # === TAHAP 3: POSTPROCESSING (loop bikin bbox, crop image) ===
            detections = []
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.model.names[int(cls)]
                    detections.append({
                        "label": label, "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })

            counts = {}
            for det in detections:
                label = det["label"]
                counts[label] = counts.get(label, 0) + 1

            out = String()
            out.data = json.dumps({"frame_id": frame_id, "counts": counts})
            t3 = time.perf_counter()

            # === TAHAP 4: PUBLISH ===
            self.pub_counting.publish(out)

            bondingid = 1
            for box in detections:
                msg_pub = BondingBoxCv()
                msg_pub.frame_id = frame_id
                msg_pub.frame_sampling = msg.frame_sampling
                msg_pub.frame_sequence.data = msg.frame_sequence.data
                msg_pub.bondingbox_id.data = bondingid
                msg_pub.bondingbox = self.bondingbox2d(*box["bbox"])
                croppedimage = frame[box["bbox"][1]:box["bbox"][3], box["bbox"][0]:box["bbox"][2]]
                msg_pub.cropped_image = self.bridge.cv2_to_imgmsg(croppedimage, encoding="bgr8")
                msg_pub.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                if box["label"] == 'person':
                    msg_pub.label = 'Person'
                    self.pub_person.publish(msg_pub)
                elif box["label"] == 'chair':
                    msg_pub.label = 'Chair'
                    self.pub_nonperson.publish(msg_pub)

                bondingid += 1

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
            self.csv_file.flush()  # langsung tulis ke disk, biar gak hilang kalau node di-Ctrl+C

            self.frame_counter += 1
            if self.frame_counter % 10 == 0:  # print ringkasan tiap 10 frame, biar gak spam terminal
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
