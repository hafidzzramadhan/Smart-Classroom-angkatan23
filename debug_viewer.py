#!/usr/bin/env python3
"""
debug_viewer.py — DEBUG ONLY, bukan node production.
Tidak didaftarkan di setup.py, tidak dijalankan di Raspberry Pi production.
Jalankan manual saja saat butuh cek visual: python3 debug_viewer.py

Subscribe:
  /cv/frame              (smartclass_msgs/msg/FrameCv)
  /cv/bondingbox/person  (smartclass_msgs/msg/BondingBoxCv)

Menampilkan frame + bounding box person + label + confidence.
Tekan 'q' di jendela viewer untuk keluar.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import cv2
from cv_bridge import CvBridge

from smartclass_msgs.msg import FrameCv, BondingBoxCv


class DebugViewer(Node):
    def __init__(self):
        super().__init__('debug_viewer')
        self.bridge = CvBridge()

        # depth=1 -> selalu ambil data TERBARU, jangan numpuk backlog
        qos = QoSProfile(depth=1)

        self.latest_frame = None       # cv2 image terakhir
        self.latest_frame_id = None    # frame_id terakhir yang diterima
        self.boxes_by_frame = {}       # {frame_id: [ (x1,y1,x2,y2,label,conf), ... ]}

        self.sub_frame = self.create_subscription(
            FrameCv, '/cv/frame', self.frame_callback, qos
        )
        self.sub_bbox = self.create_subscription(
            BondingBoxCv, '/cv/bondingbox/person', self.bbox_callback, qos
        )

        self.get_logger().info('Debug Viewer started. Tekan q di jendela untuk keluar.')

    def frame_callback(self, msg: FrameCv):
        # HANYA simpan data, JANGAN panggil cv2.imshow() di sini.
        # Rendering dipisah ke loop utama biar gak nge-block callback ROS2.
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg.frame_image, desired_encoding='bgr8')
            self.latest_frame_id = msg.frame_id
        except Exception as e:
            self.get_logger().warn(f'Gagal convert frame: {e}')

        # housekeeping ringan: buang cache box yang sudah lama, biar dict gak membengkak
        if len(self.boxes_by_frame) > 30:
            oldest_keys = list(self.boxes_by_frame.keys())[:-30]
            for k in oldest_keys:
                del self.boxes_by_frame[k]

    def bbox_callback(self, msg: BondingBoxCv):
        cx = msg.bondingbox.center.position.x
        cy = msg.bondingbox.center.position.y
        w = msg.bondingbox.size_x
        h = msg.bondingbox.size_y

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        entry = (x1, y1, x2, y2, msg.label, msg.confident_level)

        self.boxes_by_frame.setdefault(msg.frame_id, []).append(entry)

    def render(self):
        """Dipanggil dari loop utama (bukan dari callback ROS2)."""
        if self.latest_frame is None:
            return

        display = self.latest_frame.copy()
        boxes = self.boxes_by_frame.get(self.latest_frame_id, [])

        for (x1, y1, x2, y2, label, conf) in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {conf:.2f}'
            cv2.putText(
                display, text, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.putText(
            display, f'frame_id: {self.latest_frame_id}', (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
        )

        cv2.imshow('Debug Viewer - smartclass_cv', display)


def main(args=None):
    rclpy.init(args=args)
    node = DebugViewer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.render()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
