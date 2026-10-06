import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from collections import deque
import time

import mediapipe as mp

# ===== MediaPipe TASKS API =====
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===============================
# UTIL
# ===============================
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1)))

# ===============================
# GESTURE INFERENCE NODE
# ===============================
class GestureInferenceNode(Node):
    def __init__(self):
        super().__init__('gesture_inference')

        # ===============================
        # ROS
        # ===============================
        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(
            Image,
            '/camera/frame_pub',
            self.image_callback,
            10
        )

        self.pub_gesture = self.create_publisher(
            String,
            '/cv/gesture_result',
            10
        )

        self.sub_gesture_feedback = self.create_subscription(
            String,
            '/cv/gesture_feedback',
            self.gesture_feedback_callback,
            10
        )

        self.get_logger().info("Gesture Inference Node READY (HAND + FACE + SUBSCRIBER)")

        # ===============================
        # MEDIAPIPE HAND
        # ===============================
        hand_base = python.BaseOptions(
            model_asset_path="/usr/share/mediapipe/models/hand_landmarker.task"
        )

        hand_opts = vision.HandLandmarkerOptions(
            base_options=hand_base,
            num_hands=2
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

        # ===============================
        # MEDIAPIPE FACE
        # ===============================
        face_base = python.BaseOptions(
            model_asset_path="/usr/share/mediapipe/models/face_landmarker.task"
        )

        face_opts = vision.FaceLandmarkerOptions(
            base_options=face_base,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=3
        )

        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

        # ===============================
        # BUFFERS
        # ===============================
        self.gesture_buffers = {}
        self.x_history = deque(maxlen=7)

        self.face_y_history = {}
        self.FACE_HISTORY_LEN = 10

        self.last_publish_time = 0.0
        self.publish_interval = 0.5

    # ===============================
    # SUBSCRIBER CALLBACK
    # ===============================
    def gesture_feedback_callback(self, msg: String):
        self.get_logger().info(f"📡 Feedback diterima: {msg.data}")

    # ===============================
    # HAND GESTURE
    # ===============================
    def detect_hand_gesture(self, coords, frame_w):
        wrist = coords[0]
        mid = coords[12]
        hand_size = np.linalg.norm(np.array(wrist) - np.array(mid))
        if hand_size < 20:
            return "Unknown"

        if np.linalg.norm(np.array(coords[4]) - np.array(coords[8])) / hand_size < 0.35:
            return "OK Gesture"

        index_up  = angle(coords[8], coords[6], coords[5]) > 160
        middle_up = angle(coords[12], coords[10], coords[9]) > 160
        ring_up   = angle(coords[16], coords[14], coords[13]) > 160
        pinky_up  = angle(coords[20], coords[18], coords[17]) > 160

        if index_up and middle_up and ring_up and pinky_up:
            return "Raise Hand"

        if index_up and not middle_up and not ring_up and not pinky_up:
            return "Pointing"

        self.x_history.append(coords[8][0])
        if len(self.x_history) == self.x_history.maxlen:
            delta = self.x_history[-1] - self.x_history[0]
            self.x_history.clear()
            if delta > 0.25 * frame_w:
                return "Writing Strong"
            if delta > 0.15 * frame_w:
                return "Writing L → R"

        return "Unknown"

    # ===============================
    # IMAGE CALLBACK
    # ===============================
    def image_callback(self, msg: Image):
        self.get_logger().info("📥 Frame diterima")

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        if frame is None:
            return

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ FIX UTAMA DI SINI
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # ===============================
        # FACE → MEMAHAMI
        # ===============================
        face_res = self.face_landmarker.detect(mp_image)
        if face_res.face_landmarks:
            for i, face in enumerate(face_res.face_landmarks):
                self.face_y_history.setdefault(i, deque(maxlen=self.FACE_HISTORY_LEN))
                nose_y = int(face[1].y * h)
                self.face_y_history[i].append(nose_y)

                if len(self.face_y_history[i]) == self.FACE_HISTORY_LEN:
                    if max(self.face_y_history[i]) - min(self.face_y_history[i]) > 0.08 * h:
                        self.publish_gesture("Memahami")

        # ===============================
        # HAND
        # ===============================
        hand_res = self.hand_landmarker.detect(mp_image)
        if not hand_res.hand_landmarks:
            return

        for i, landmarks in enumerate(hand_res.hand_landmarks):
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            label = self.detect_hand_gesture(coords, w)

            if label == "Unknown":
                continue

            self.gesture_buffers.setdefault(i, deque(maxlen=10))
            self.gesture_buffers[i].append(label)
            final = max(set(self.gesture_buffers[i]), key=self.gesture_buffers[i].count)

            self.publish_gesture(final)

    # ===============================
    # PUBLISH
    # ===============================
    def publish_gesture(self, label: str):
        now = time.time()
        if now - self.last_publish_time < self.publish_interval:
            return

        msg = String()
        msg.data = label
        self.pub_gesture.publish(msg)

        self.last_publish_time = now
        self.get_logger().info(f"Publish: {label}")

# ===============================
# MAIN
# ===============================
def main():
    rclpy.init()

    node = GestureInferenceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

