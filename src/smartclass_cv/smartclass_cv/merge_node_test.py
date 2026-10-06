import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from smartclass_msgs.msg import BondingBoxCv
from smartclass_msgs.msg import BondingBox
from smartclass_msgs.msg import FrameCv
from smartclass_msgs.msg import ResultCv
import datetime
import cv2
import numpy as np
import base64
import os
from std_msgs.msg import String
from std_msgs.msg import Int64
from cv_bridge import CvBridge
from queue import Queue
from collections import defaultdict, OrderedDict

def maintain_box(box,initial):
        if initial is not None:
            if initial.identity == "":
                initial.identity = box.identity
            if initial.expression == "":
                initial.expression = box.expression
                initial.confident_level = box.confident_level
            if initial.gesture == "":
                initial.gesture = box.gesture
            if initial.activity == "":
                initial.activity = box.activity
        else:
            initial = box

        return initial


def draw_box_label(cv_image,bondingboxs):

    if bondingboxs != None:
        colors = {"recognition":(0, 255, 0), "expression":(255, 0, 0),"gesture":(0, 0, 255),"activity":(255,255,0),"Unknown":(0,255,200)}
        # Colors for the boxes (Green, Red, Blue)
        for box in bondingboxs:
            w = int(box.bondingbox.size_x)
            h = int(box.bondingbox.size_y)
            x = int(box.bondingbox.center.position.x - w/2)
            y = int(box.bondingbox.center.position.y - h/2)

            color = colors[box.name]
            print(f"DRAW_BOX name={box.name} label={box.label} x={x} y={y} w={w} h={h} image={cv_image.shape}")
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 10)
            print(f"AFTER_DRAW pixel={cv_image[y, x].tolist()} expected_color={list(color)}")
            # Add label (text)
            if box.label == None:
                box.label = "Unknown"

            cv2.putText(cv_image, box.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        save_image(cv_image,"None")


    _, buffer = cv2.imencode('.jpg', cv_image)
    jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')

    return cv_image

def save_image(frame,filename):
        # Generate a filename with timestamp
        if filename == "None":
            ct = datetime.datetime.now()
            filename = ct.strftime("%Y-%m-%d %H:%M:%S")+ f".{ct.microsecond // 1000:03d}"
        filename=f"/home/hafidz/smartclass_ws/saved_detected_person/face/frame_{filename}.jpg"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the frame as an image (you can change the file format if necessary)
        cv2.imwrite(filename, frame)


class BondingBoxMerger(Node):
    def __init__(self):
        super().__init__("BondingBoxMerger")

        self.bridge = CvBridge()


        # subscriber frame
        self.sub_stream = self.create_subscription(Int64, "/cv/stream",self.stream_callback,10)

        # subscriber frame
        self.sub_frame = self.create_subscription(FrameCv, "/cv/frame",self.frame_callback,10)

        # subscriber recognition
        self.sub_recognition = self.create_subscription(BondingBoxCv, "/cv/bondingbox/face_identity", self.recognition_callback,10)

        # subscriber emotion
        self.sub_emotion = self.create_subscription(BondingBoxCv, "/cv/bondingbox/face_emotion", self.emotion_callback,10)
        # subscriber activity
        self.sub_activity = self.create_subscription(BondingBoxCv, "/cv/bondingbox/activity", self.activity_callback,10)

        # subscriber gesture
        self.sub_gesture = self.create_subscription(BondingBoxCv, "/cv/bondingbox/gesture", self.gesture_callback,10)

        # publish all
        self.pub_foxglove = self.create_publisher(CompressedImage, "/cv/frame/image",10)
        self.pub_result = self.create_publisher(ResultCv, "/cv/frame/result",10)
        self.pub_web = self.create_publisher(String, "/cv/frame/web",100)
        #self.timer = self.create_timer(0.03333, self.pub_to_web)  # ~30 FPS

        self.lastPublish = Int64()
        self.lastPublish.data = 0
        self.pub_status = "0"
        self.nframesample = 5
        self.QBox = defaultdict(list)
        # Engineering synchronization cache (NOT the TA dynamic playout buffer).
        # Temporary development limits; not research-derived values.
        self.frame_cache = OrderedDict()
        self.frame_cache_max = 1500
        self.qbox_max_pending_keys = 600
        self.QFrame = Queue()
        self.Qweb = Queue()
        self.QStream = Queue()
        self.CurrentFrame = FrameCv()
        self.CurrentFrame = None
        self.CurrentResult = ResultCv()
        self.counting = 1
        self.base64 = String()
        self.get_logger().info("Siap memproses")

    def pub_to_web(self):
        if self.Qweb.qsize() > 0:
            self.pub_web.publish(self.Qweb.get())

    def _cap_qbox_size(self):
        if len(self.QBox) > self.qbox_max_pending_keys:
            oldest_first = sorted(self.QBox.keys(), key=int)
            excess = len(self.QBox) - self.qbox_max_pending_keys
            for k in oldest_first[:excess]:
                del self.QBox[k]

    def _cache_frame(self, msg: FrameCv):
        """Cache exact frames as compressed JPEG while slower AI branches finish."""
        seq = int(msg.frame_sequence.data)
        cv_image = self.bridge.imgmsg_to_cv2(msg.frame_image, desired_encoding='bgr8')
        ok, encoded = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        self.frame_cache[seq] = {
            "jpeg": encoded.tobytes(),
            "camera_id": msg.camera_id,
            "frame_id": msg.frame_id,
            "frame_sampling": int(msg.frame_sampling.data),
        }
        self.frame_cache.move_to_end(seq)
        while len(self.frame_cache) > self.frame_cache_max:
            old_seq, _ = self.frame_cache.popitem(last=False)
            self.QBox.pop(str(old_seq), None)

    def _publish_exact_fusion(self, seq):
        """Fuse AI result N only with image N; never with a newer live frame."""
        cached = self.frame_cache.get(seq)
        bondings = self.QBox.get(str(seq))
        if cached is None or not bondings:
            return False

        arr = np.frombuffer(cached["jpeg"], dtype=np.uint8)
        cv_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return False
        cv_image = draw_box_label(cv_image, bondings)

        msg = ResultCv()
        msg.camera_id = cached["camera_id"]
        msg.frame_id = cached["frame_id"]
        msg.frame_sequence = seq
        msg.frame_sampling = cached["frame_sampling"]
        msg.timestamp_pub = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for box in bondings:
            boxs = BondingBox()
            boxs.label = box.label
            boxs.confident_level = box.confident_level
            boxs.name = box.name
            boxs.bondingbox = box.bondingbox
            msg.bondingboxs.append(boxs)

        ok, encoded = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ok:
            return False
        msg.image_base64 = base64.b64encode(encoded).decode('utf-8')
        self.base64.data = msg.image_base64
        image_msg = CompressedImage()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.format = 'jpeg'
        image_msg.data = encoded.tobytes()
        self.pub_foxglove.publish(image_msg)
        self.pub_result.publish(msg)
        self.CurrentResult = msg
        self.QBox.pop(str(seq), None)
        return True

    def stream_frame(self, stream):
        # Latest-only progress signal: stale stream IDs are discarded.
        self.QStream.put(stream, block=False)
        while self.QStream.qsize() > 1:
            self.QStream.get()
        latest_stream = self.QStream.get()
        self.lastPublish.data = latest_stream.data
        print(f'stream latest = {latest_stream.data}')

    def publish_live(self, frames):
        """Publish current live image without borrowing annotations from another frame."""
        cv_image = self.bridge.imgmsg_to_cv2(frames.frame_image, desired_encoding='bgr8')
        ok, encoded = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ok:
            image_msg = CompressedImage()
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.format = 'jpeg'
            image_msg.data = encoded.tobytes()
            self.pub_foxglove.publish(image_msg)

    def stream_callback(self, msg: Int64):
        self.stream_frame(msg)

    def frame_callback(self, msg: FrameCv):
        # No unbounded frame FIFO.
        self._cache_frame(msg)
        # self.publish_live(msg)  # TEST: jangan timpa annotated frame dengan raw live frame

    def _store_ai_result(self, msg: BondingBoxCv, name: str):
        msg.name = name
        seq = int(msg.frame_sequence.data)
        print(f'box frame {name}- {seq}')
        self.QBox.setdefault(str(seq), []).append(msg)
        self.pub_status = "1"
        self._cap_qbox_size()
        if not self._publish_exact_fusion(seq):
            print(f'waiting exact frame - {seq}')

    def recognition_callback(self, msg: BondingBoxCv):
        self._store_ai_result(msg, "recognition")

    def emotion_callback(self, msg: BondingBoxCv):
        self._store_ai_result(msg, "expression")

    def activity_callback(self, msg: BondingBoxCv):
        self._store_ai_result(msg, "activity")

    def gesture_callback(self, msg: BondingBoxCv):
        self._store_ai_result(msg, "gesture")



def main():
    rclpy.init()
    node = BondingBoxMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()