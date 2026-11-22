import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from smartclass_msgs.msg import BondingBoxCv
from smartclass_msgs.msg import BondingBox
from smartclass_msgs.msg import FrameCv
from smartclass_msgs.msg import ResultCv
import datetime
import copy
import time
import cv2
import numpy as np
import base64
import os
from std_msgs.msg import String
from std_msgs.msg import Int64
from cv_bridge import CvBridge
from queue import Queue
from collections import OrderedDict


FACE_ANNOTATIONS = {'recognition', 'expression'}

# Ketebalan garis bounding box (seragam untuk face, person, activity, gesture)
BOX_THICKNESS = 2

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
        # [PERSON PATCH] tambah warna "person" (magenta)
        colors = {"recognition":(0, 255, 0), "expression":(255, 0, 0),"gesture":(0, 0, 255),"activity":(255,255,0),"Unknown":(0,255,200),"person":(255, 0, 255)}
        for box in bondingboxs:
            w = int(box.bondingbox.size_x)
            h = int(box.bondingbox.size_y)
            x = int(box.bondingbox.center.position.x - w/2)
            y = int(box.bondingbox.center.position.y - h/2)

            # pencarian warna aman (tidak KeyError); ketebalan garis seragam untuk semua kotak
            color = colors.get(box.name, (255, 255, 255))
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, BOX_THICKNESS)

            if box.label == None:
                box.label = "Unknown"

            # activity/gesture: teks di dalam kotak agar tidak menimpa label "Person"
            if box.name in ("activity", "gesture"):
                text_pos = (x + 6, y + 30)
            else:
                text_pos = (x, y - 10)
            cv2.putText(cv_image, box.label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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

        # Koordinat wajah terbaru langsung dari YOLO
        self.sub_face_detection = self.create_subscription(
            BondingBoxCv,
            "/cv/bondingbox/face",
            self.face_detection_callback,
            10
        )

        # subscriber recognition
        self.sub_recognition = self.create_subscription(BondingBoxCv, "/cv/bondingbox/face_identity", self.recognition_callback,10)

        self.sub_face_analysis = self.create_subscription(
            BondingBoxCv, "/cv/bondingbox/face_analysis", self.recognition_callback, 10)

        # subscriber emotion
        self.sub_emotion = self.create_subscription(BondingBoxCv, "/cv/bondingbox/face_emotion", self.emotion_callback,10)
        # subscriber activity
        self.sub_activity = self.create_subscription(BondingBoxCv, "/cv/bondingbox/activity", self.activity_callback,10)

        # subscriber gesture
        self.sub_gesture = self.create_subscription(BondingBoxCv, "/cv/bondingbox/gesture", self.gesture_callback,10)

        # [PERSON PATCH] subscriber person (dari DetectionNode)
        self.sub_person = self.create_subscription(
            BondingBoxCv, "/cv/bondingbox/person", self.person_callback, 10)

        # publish all
        self.pub_foxglove = self.create_publisher(CompressedImage, "/cv/frame/image",10)
        self.pub_result = self.create_publisher(ResultCv, "/cv/frame/result",10)
        self.pub_web = self.create_publisher(String, "/cv/frame/web",100)
        #self.timer = self.create_timer(0.03333, self.pub_to_web)  # ~30 FPS

        self.lastPublish = Int64()
        self.lastPublish.data = 0
        self.pub_status = "0"
        self.nframesample = 5
        self.QBox = {}
        self.latest_overlay_seq = -1
        self.latest_overlay_time = 0.0
        self.latest_overlay = []
        self.latest_detection_seq = -1
        self.latest_detections = []
        self.latest_detection_time = 0.0

        # [PERSON PATCH] state kotak person
        self.latest_person_seq = -1
        self.latest_persons = []
        self.latest_person_time = 0.0
        self.person_ttl = 1.5  # detik; kotak person hilang jika tidak ada deteksi baru

        # [ACTIVITY PATCH] kotak activity & gesture untuk video live
        self.latest_extra = {
            "activity": {"seq": -1, "time": 0.0, "boxes": []},
            "gesture": {"seq": -1, "time": 0.0, "boxes": []},
        }
        self.extra_ttl = 1.5

        # BYTETRACK_ID_OVERLAY_V1
        self.track_labels = {}
        self.track_last_seen = {}
        self.track_label_ttl = 2.0
        self.track_detection_ttl = 2.0
        self.track_reuse_gap = 1.0

        self.overlay_max_age = 12
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

        # Publish fusi digabung per 50 ms supaya tidak berulang tiap hasil AI
        self.pending_fusion = set()
        self.create_timer(0.05, self._flush_fusion)

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

    def _cache_frame(self, msg: FrameCv, cv_image):
        """Cache exact frames as compressed JPEG while slower AI branches finish."""
        seq = int(msg.frame_sequence.data)
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
        annotations = self.QBox.get(str(seq))
        if cached is None or not annotations:
            return False
        bondings = list(annotations.values())

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
        msg.bondingboxs = bondings

        ok, encoded = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ok:
            return False
        msg.image_base64 = base64.b64encode(encoded).decode('utf-8')
        self.base64.data = msg.image_base64
        image_msg = CompressedImage()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.format = 'jpeg'
        image_msg.data = encoded.tobytes()
        self.pub_result.publish(msg)
        self.CurrentResult = msg
        # Keep earlier face boxes so the next crop from this frame can add its box.
        face_annotations = {
            key: box for key, box in annotations.items() if box.name in FACE_ANNOTATIONS
        }
        if face_annotations:
            self.QBox[str(seq)] = face_annotations
        else:
            self.QBox.pop(str(seq), None)
        return True

    def stream_frame(self, stream):
        # Latest-only progress signal: stale stream IDs are discarded.
        self.QStream.put(stream, block=False)
        while self.QStream.qsize() > 1:
            self.QStream.get()
        latest_stream = self.QStream.get()
        self.lastPublish.data = latest_stream.data

    def publish_live(self, cv_image):
        """Publish live video independently of AI completion."""

        # [PERSON PATCH] person digambar dulu, wajah di atasnya
        overlay = self._merged_person_extra()  # person + activity/gesture digabung
        overlay += self._tracked_overlay()
        if overlay:
            cv_image = draw_box_label(cv_image, overlay)

        ok, encoded = cv2.imencode(
            '.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not ok:
            return

        image_msg = CompressedImage()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.format = 'jpeg'
        image_msg.data = encoded.tobytes()
        self.pub_foxglove.publish(image_msg)

    def stream_callback(self, msg: Int64):
        self.stream_frame(msg)

    def frame_callback(self, msg: FrameCv):
        # No unbounded frame FIFO. Konversi image hanya sekali per frame.
        cv_image = self.bridge.imgmsg_to_cv2(msg.frame_image, desired_encoding='bgr8')
        seq = int(msg.frame_sequence.data)
        # Cache di-encode SEBELUM publish_live menggambar kotak di atas cv_image
        self._cache_frame(msg, cv_image)
        if str(seq) in self.QBox:
            self._publish_exact_fusion(seq)
        self.publish_live(cv_image)

    def _store_ai_result(self, msg: BondingBoxCv, name: str):
        seq = int(msg.frame_sequence.data)
        box = BondingBox()
        box.label = msg.label
        box.confident_level = msg.confident_level
        box.name = name
        box.bondingbox = msg.bondingbox
        # Face results have a crop ID. Other branches may reuse IDs in one frame.
        key = (name, msg.bondingbox_id.data) if name in FACE_ANNOTATIONS else object()
        self.QBox.setdefault(str(seq), {})[key] = box
        self.pub_status = "1"
        self._cap_qbox_size()

        # [ACTIVITY PATCH] simpan juga untuk overlay live (/cv/frame/image)
        if name in self.latest_extra:
            slot = self.latest_extra[name]
            if seq >= slot["seq"]:
                if seq > slot["seq"]:
                    slot["seq"] = seq
                    slot["boxes"] = []
                live_box = BondingBox()
                live_box.label = msg.label or name
                live_box.confident_level = msg.confident_level
                live_box.name = name
                live_box.bondingbox = copy.deepcopy(msg.bondingbox)
                slot["boxes"].append(live_box)
                slot["time"] = time.monotonic()

        if name in FACE_ANNOTATIONS:
            tid = int(msg.bondingbox_id.data)
            now = time.monotonic()

            # Jangan terima respons lama untuk track yang
            # sudah lama hilang atau sudah digunakan ulang.
            last_seen = self.track_last_seen.get(tid)

            if (
                last_seen is not None
                and seq <= last_seen["seq"]
                and now - last_seen["time"] <= self.track_reuse_gap
            ):
                previous = self.track_labels.get(tid)

                if previous is None or seq >= previous["seq"]:
                    self.track_labels[tid] = {
                        "label": str(msg.label or "Unknown"),
                        "seq": seq,
                        "time": now,
                    }

        if seq >= self.latest_overlay_seq:
            self.latest_overlay_seq = seq
            self.latest_overlay = list(self.QBox[str(seq)].values())
            self.latest_overlay_time = time.monotonic()
        # Gabungkan publish fusi: dikirim berkala oleh timer, bukan tiap hasil AI
        self.pending_fusion.add(seq)

    def _flush_fusion(self):
        pending = sorted(self.pending_fusion)
        self.pending_fusion.clear()
        for seq in pending:
            self._publish_exact_fusion(seq)

    def face_detection_callback(self, msg: BondingBoxCv):
        """Simpan koordinat YOLO berdasarkan ByteTrack ID."""
        seq = int(msg.frame_sequence.data)
        tid = int(msg.bondingbox_id.data)
        now = time.monotonic()

        if seq < self.latest_detection_seq:
            return

        if seq > self.latest_detection_seq:
            self.latest_detection_seq = seq
            self.latest_detections = {}

        previous = self.track_last_seen.get(tid)

        # Track lama yang muncul kembali setelah jeda
        # tidak boleh langsung mewarisi label lama.
        if (
            previous is not None
            and now - previous["time"] > self.track_reuse_gap
        ):
            self.track_labels.pop(tid, None)

        self.latest_detections[tid] = copy.deepcopy(msg.bondingbox)
        self.track_last_seen[tid] = {
            "seq": seq,
            "time": now,
        }
        self.latest_detection_time = now

    def _tracked_overlay(self):
        """Gambar posisi terbaru dan label milik track ID."""
        now = time.monotonic()

        if now - self.latest_detection_time > self.track_detection_ttl:
            return []

        overlays = []

        for tid, bbox in self.latest_detections.items():
            seen = self.track_last_seen.get(tid)

            if seen is None:
                continue

            if now - seen["time"] > self.track_detection_ttl:
                continue

            item = self.track_labels.get(tid)
            label = f"Tracking ID {tid}"

            if item is not None:
                if now - item["time"] <= self.track_label_ttl:
                    label = item["label"]

            box = BondingBox()
            box.name = "recognition"
            box.label = label
            box.bondingbox = copy.deepcopy(bbox)
            overlays.append(box)

        return overlays

    # [PERSON PATCH] callback + overlay untuk kotak person
    def person_callback(self, msg: BondingBoxCv):
        """Simpan kotak person terbaru (satu pesan per kotak, dikelompokkan per frame)."""
        seq = int(msg.frame_sequence.data)
        if seq < self.latest_person_seq:
            return
        if seq > self.latest_person_seq:
            self.latest_person_seq = seq
            self.latest_persons = []

        box = BondingBox()
        box.name = "person"
        box.label = msg.label or "Person"
        box.confident_level = msg.confident_level
        box.bondingbox = copy.deepcopy(msg.bondingbox)
        self.latest_persons.append(box)
        self.latest_person_time = time.monotonic()

    def _person_overlay(self):
        """Ambil kotak person terbaru, buang duplikat (tumpang tindih / bersarang)."""
        if time.monotonic() - self.latest_person_time > self.person_ttl:
            return []
        boxes = sorted(self.latest_persons,
                       key=lambda b: b.confident_level, reverse=True)
        kept = []
        for b in boxes:
            if all(not self._is_duplicate(b, k) for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def _is_duplicate(a, b, iou_thr=0.5, contain_thr=0.8):
        def rect(o):
            bb = o.bondingbox
            return (bb.center.position.x - bb.size_x / 2,
                    bb.center.position.y - bb.size_y / 2,
                    bb.center.position.x + bb.size_x / 2,
                    bb.center.position.y + bb.size_y / 2)
        ax1, ay1, ax2, ay2 = rect(a)
        bx1, by1, bx2, by2 = rect(b)
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return False
        iou = inter / union
        smaller = min(area_a, area_b)
        contain = inter / smaller if smaller > 0 else 0.0
        # duplikat jika tumpang tindih besar ATAU satu kotak hampir seluruhnya di dalam yang lain
        return iou >= iou_thr or contain >= contain_thr

    def _merged_person_extra(self):
        """Gabungkan activity/gesture ke kotak person kalau posisinya sama.

        Kotak activity/gesture yang menumpuk person tidak digambar sendiri;
        namanya ditambahkan ke label person. Yang tidak menumpuk person
        tetap digambar, tetapi duplikat antar-extra dibuang.
        """
        persons = [copy.deepcopy(p) for p in self._person_overlay()]
        extras = sorted(self._extra_overlay(),
                        key=lambda b: b.confident_level, reverse=True)
        standalone = []
        for ex in extras:
            host = next((p for p in persons if self._is_duplicate(ex, p)), None)
            if host is not None:
                tag = f"{ex.name}: {ex.label}"
                if tag not in host.label:
                    host.label = f"{host.label} | {tag}"
            elif ex.name == "activity":
                # kotak activity yang tidak menempel ke person (cyan) tidak digambar
                continue
            elif all(not self._is_duplicate(ex, s) for s in standalone):
                standalone.append(ex)
        return persons + standalone

    def _extra_overlay(self):
        """Kotak activity & gesture terbaru untuk video live."""
        now = time.monotonic()
        boxes = []
        for slot in self.latest_extra.values():
            if now - slot["time"] <= self.extra_ttl:
                boxes.extend(slot["boxes"])
        return boxes

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
