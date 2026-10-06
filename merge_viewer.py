#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class MergeViewer(Node):
    def __init__(self):
        super().__init__('merge_viewer')

        self.sub = self.create_subscription(
            CompressedImage,
            '/cv/frame/image',
            self.image_callback,
            10
        )

        self.get_logger().info('Viewing /cv/frame/image')

    def image_callback(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        print(f"RECEIVED bytes={len(msg.data)} decoded={None if frame is None else frame.shape}")

        if frame is not None:
            cv2.imshow('MERGE OUTPUT - /cv/frame/image', frame)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = MergeViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
