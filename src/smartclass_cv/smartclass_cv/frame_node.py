import rclpy
from rclpy.node import Node

import datetime
import cv2
import numpy as np

from std_msgs.msg import Int64
from cv_bridge import CvBridge
from smartclass_msgs.msg import FrameCv


class FramePublisher(Node):

    def __init__(self):
        super().__init__("frame_node")

        # =====================================================
        # PUBLISHER
        # =====================================================

        self.pub_frame = self.create_publisher(
            FrameCv,
            "/cv/frame",
            10
        )

        self.pub_stream = self.create_publisher(
            Int64,
            "/cv/stream",
            10
        )

        # =====================================================
        # CAMERA
        # =====================================================

        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error(
                "❌ Kamera tidak bisa dibuka"
            )
            raise RuntimeError(
                "Kamera /dev/video0 tidak dapat dibuka"
            )

        # =====================================================
        # FPS
        # =====================================================

        self.target_fps = 30.0

        self.timer = self.create_timer(
            1.0 / self.target_fps,
            self.publish_frame
        )

        # =====================================================
        # FRAME COUNTER
        # =====================================================

        self.frame_sequence = Int64()
        self.frame_sequence.data = 1

        self.frame_sampling = Int64()
        self.frame_sampling.data = 1

        # =====================================================
        # SIMILARITY
        # =====================================================

        self.preframe = None

        # Untuk sementara similarity filter DIMATIKAN.
        #
        # Tujuannya:
        # kita ingin memastikan /cv/frame mampu
        # mengirim frame secara stabil terlebih dahulu.
        #
        # Setelah FPS stabil, similarity filter bisa
        # kita aktifkan kembali jika memang diperlukan.
        self.use_similarity_filter = False

        self.similarity_threshold = 30.0

        # =====================================================
        # LOG
        # =====================================================

        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            "✅ Frame Publisher started"
        )

        self.get_logger().info(
            f"Camera index : 0"
        )

        self.get_logger().info(
            f"Target FPS   : {self.target_fps}"
        )

        self.get_logger().info(
            f"Similarity   : "
            f"{self.use_similarity_filter}"
        )

        self.get_logger().info(
            "Topic frame  : /cv/frame"
        )

        self.get_logger().info(
            "Topic stream : /cv/stream"
        )

        self.get_logger().info(
            "========================================"
        )

    # =========================================================
    # SIMILARITY
    # =========================================================

    def isSimilar(self, image1, image2):

        image1 = cv2.cvtColor(
            image1,
            cv2.COLOR_BGR2GRAY
        )

        image2 = cv2.cvtColor(
            image2,
            cv2.COLOR_BGR2GRAY
        )

        if image1.shape != image2.shape:
            return float("inf")

        # Penting:
        # ubah uint8 -> float32 sebelum pengurangan.
        #
        # Kalau tidak, misalnya:
        # 10 - 20 pada uint8 menjadi 246.
        #
        # Dengan float32:
        # 10 - 20 = -10
        # (-10)^2 = 100

        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)

        err = np.mean(
            (image1 - image2) ** 2
        )

        return float(err)

    # =========================================================
    # CREATE FRAME MESSAGE
    # =========================================================

    def create_frame_message(self, frame):

        msg = FrameCv()

        # Image OpenCV -> ROS Image
        msg.frame_image = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8"
        )

        msg.camera_id = "1"

        msg.frame_sequence.data = (
            self.frame_sequence.data
        )

        msg.frame_sampling.data = (
            self.frame_sampling.data
        )

        # Timestamp frame
        ct = datetime.datetime.now()

        msg.frame_id = (
            ct.strftime("%Y-%m-%d %H:%M:%S")
            + f".{ct.microsecond // 1000:03d}"
        )

        return msg

    # =========================================================
    # PUBLISH FRAME
    # =========================================================

    def publish_frame(self):

        # -----------------------------------------------------
        # 1. Ambil frame dari kamera
        # -----------------------------------------------------

        ret, frame = self.cap.read()

        if not ret:

            self.get_logger().warning(
                "⚠️ Gagal mengambil frame kamera"
            )

            return

        # -----------------------------------------------------
        # 2. Similarity filter
        # -----------------------------------------------------
        #
        # Untuk sekarang DIMATIKAN.
        #
        # Jadi setiap frame kamera akan diteruskan
        # ke /cv/frame.
        #
        # Nanti setelah FPS stabil, filter ini bisa
        # kita uji kembali.
        # -----------------------------------------------------

        should_publish = True

        if self.use_similarity_filter:

            if self.preframe is None:

                should_publish = True

            else:

                error = self.isSimilar(
                    self.preframe,
                    frame
                )

                print(
                    f"error - {error}"
                )

                should_publish = (
                    error >
                    self.similarity_threshold
                )

            self.preframe = frame.copy()

        # -----------------------------------------------------
        # 3. Publish frame
        # -----------------------------------------------------

        if should_publish:

            msg = self.create_frame_message(
                frame
            )

            self.pub_frame.publish(
                msg
            )

            print(
                f"sequence take - "
                f"{self.frame_sequence.data}"
            )

            self.frame_sampling.data += 1

        # -----------------------------------------------------
        # 4. Publish stream sequence
        # -----------------------------------------------------

        self.pub_stream.publish(
            self.frame_sequence
        )

        # -----------------------------------------------------
        # 5. Increment sequence
        # -----------------------------------------------------

        self.frame_sequence.data += 1

    # =========================================================
    # DESTROY NODE
    # =========================================================

    def destroy_node(self):

        if (
            hasattr(self, "cap")
            and self.cap.isOpened()
        ):
            self.cap.release()

        super().destroy_node()


# =============================================================
# MAIN
# =============================================================

def main(args=None):

    rclpy.init(args=args)

    node = FramePublisher()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
