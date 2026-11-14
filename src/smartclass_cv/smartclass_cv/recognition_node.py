import os
import base64
import requests
import pickle
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from time import monotonic

import cv2
import rclpy
from rclpy.node import Node
from smartclass_msgs.msg import BondingBoxCv
from cv_bridge import CvBridge

import numpy as np
from smartclass_cv.recognition_match import compare_faces_np, pick_name
from smartclass_cv.remote_client import RemoteError, RemoteModel, RttStats


class FaceRecognition(Node):
    def __init__(self):
        super().__init__('recognition_node')

        # =========================
        # PARAMETER
        # =========================
        # mode 'local'  -> face_recognition (dlib) dijalankan di node ini (perilaku lama)
        # mode 'remote' -> hanya penghitungan ENCODING wajah (bagian berat) dikirim ke server.
        #                  Pencocokan dengan encodings.pickle dan penulisan absensi
        #                  (SQLite) tetap dilakukan di node ini.
        # mode 'server_match' -> server mencocokkan wajah dan menilai ekspresi.
        self.declare_parameter('mode', 'local')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000')
        self.declare_parameter('timeout_s', 2.0)
        self.declare_parameter('image_format', 'jpg')   # 'jpg' atau 'png'
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('tolerance', 0.45)
        # Kosong -> ~/smartclass_ws/src/smartclass_cv/face_registered
        # (sama dengan path lama untuk user hafidz)
        self.declare_parameter('work_dir', '')

        self.mode = self.get_parameter('mode').value
        if self.mode not in ('local', 'remote', 'server_match', 'fikri_api'):
            raise ValueError("mode harus local, remote, atau server_match")
        self.tolerance = float(self.get_parameter('tolerance').value)

        if self.mode not in ('server_match', 'fikri_api'):
            # --- KONFIGURASI PATH ---
            work_dir = self.get_parameter('work_dir').value or os.path.expanduser(
                "~/smartclass_ws/src/smartclass_cv/face_registered")
            self.work_dir = work_dir
            self.pickle_path = os.path.join(self.work_dir, "encodings.pickle")
            self.db_path = os.path.join(self.work_dir, "absensi.db")

            with open(self.pickle_path, "rb") as source:
                self.data = pickle.load(source)

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.absen_hari_ini = set()

        # Model / klien
        if self.mode == 'local':
            import face_recognition  # hanya dibutuhkan di mode local (butuh dlib)
            self.fr = face_recognition
        elif self.mode == 'fikri_api':
            self.fikri_url = (
                self.get_parameter('server_url').value.rstrip('/')
                + '/api/recognize-test/'
            )
            self.get_logger().info(f"Fikri API: {self.fikri_url}")
        else:
            endpoint = '/face_analysis' if self.mode == 'server_match' else '/face_encoding'
            url = self.get_parameter('server_url').value.rstrip('/') + endpoint
            self.remote = RemoteModel(
                url,
                timeout=self.get_parameter('timeout_s').value,
                image_format=self.get_parameter('image_format').value,
                jpeg_quality=self.get_parameter('jpeg_quality').value,
            )
            self.stats = RttStats(self.get_logger(), 'recognition')
            self.get_logger().info(f"🌐 Remote encoding: {url}")

        # ROS Communications
        self.sub = self.create_subscription(BondingBoxCv, '/cv/bondingbox/face', self.listener_callback, 10)
        topic = (
            '/cv/bondingbox/face_analysis' if self.mode in ('server_match', 'fikri_api')
            else '/cv/bondingbox/face_identity'
        )
        self.pub = self.create_publisher(BondingBoxCv, topic, 10)
        self.br = CvBridge()
        # Fikri API: 2 HTTP workers; at most 6 crops outstanding (running + queued).
        # ROS callback never waits for the network.
        if self.mode == 'fikri_api':
            self._fikri_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='fikri')
            self._fikri_slots = BoundedSemaphore(6)
            self._fikri_lock = Lock()
            self._fikri_latest_seq = -1
            self._fikri_dropped = 0
        self.get_logger().info(
            f"SERVER READY (mode={self.mode}): Menunggu gambar dari face detection...")

    # ------------------------------------------------------------------
    # Encoding wajah: hasilnya selalu list encoding (satu per wajah), atau None
    # kalau server gagal.
    # ------------------------------------------------------------------
    def _encode_local(self, frame):
        face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.fr.face_encodings(face_rgb)

    def _encode_remote(self, frame):
        try:
            res = self.remote.infer(frame)
        except RemoteError as e:
            self.stats.add_fail(e)
            return None
        self.stats.add(res)
        return [np.asarray(e, dtype=np.float64) for e in res['detections']]

    def _compare(self, encoding):
        if self.mode == 'local':
            return self.fr.compare_faces(
                self.data["encodings"], encoding, tolerance=self.tolerance)
        return compare_faces_np(self.data["encodings"], encoding, self.tolerance)

    def _publish_server_analysis(self, msg, frame):
        try:
            result = self.remote.infer(frame)
        except RemoteError as error:
            self.stats.add_fail(error)
            return
        self.stats.add(result)
        if result["status"] in ("no_face", "multiple_faces"):
            return

        identity = result["identity"]
        msg.label = identity["name"] if identity else "Unknown"
        expression = result["expression"]
        msg.confident_level = 0.0
        if expression is not None:
            msg.label += f" | {expression['label']}"
            msg.confident_level = round(float(expression["conf"]), 2)
        msg.timestamp = datetime.now().strftime("%H:%M:%S")
        self.pub.publish(msg)


    def _submit_fikri(self, msg, frame):
        """Bounded async HTTP: never stall the ROS subscription callback."""
        seq = int(msg.frame_sequence.data)
        with self._fikri_lock:
            self._fikri_latest_seq = max(self._fikri_latest_seq, seq)
        if not self._fikri_slots.acquire(blocking=False):
            self._fikri_dropped += 1
            if self._fikri_dropped % 50 == 1:
                self.get_logger().warning(
                    f'Fikri overload: {self._fikri_dropped} crops skipped (bounded queue)')
            return
        try:
            self._fikri_pool.submit(self._fikri_worker, msg, frame, seq)
        except RuntimeError:
            self._fikri_slots.release()

    def _fikri_worker(self, msg, frame, seq):
        try:
            # Skip queued jobs once newer frames have arrived.
            # Keep a 12-frame window so multiple faces in a frame can finish.
            with self._fikri_lock:
                latest = self._fikri_latest_seq
            if seq < latest - 12:
                return
            self._publish_fikri_analysis(msg, frame)
        finally:
            self._fikri_slots.release()

    def _publish_fikri_analysis(self, msg, frame):
        try:
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                self.get_logger().error("Gagal encode crop wajah")
                return

            image_base64 = base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")

            response = requests.post(
                self.fikri_url,
                json={"image": image_base64},
                timeout=float(self.get_parameter('timeout_s').value)
            )
            response.raise_for_status()
            result = response.json()

            name = (
                result.get("student_name")
                if result.get("success")
                else None
            )
            msg.label = name or "Unknown"

            expression = result.get("expression")
            if isinstance(expression, dict):
                expression = expression.get("label")
            if expression:
                msg.label += f" | {expression}"

            msg.confident_level = 0.0
            msg.timestamp = datetime.now().strftime("%H:%M:%S")
            self.pub.publish(msg)
            self.get_logger().info(f"Fikri API: {msg.label}")

        except (requests.RequestException, ValueError) as error:
            self.get_logger().error(f"Fikri API gagal: {error}")

    def destroy_node(self):
        if self.mode == 'fikri_api':
            self._fikri_pool.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()

    def listener_callback(self, msg: BondingBoxCv):
        try:
            # 1. Convert Image
            frame = self.br.imgmsg_to_cv2(msg.cropped_image)

            if self.mode == 'fikri_api':
                self._submit_fikri(msg, frame)
                return

            if self.mode == 'server_match':
                self._publish_server_analysis(msg, frame)
                return

            # 2. Face Recognition
            if self.mode == 'local':
                face_encodings = self._encode_local(frame)
            else:
                face_encodings = self._encode_remote(frame)
                if face_encodings is None:
                    return  # server gagal: lewati crop ini

            detected_names = []
            name = "Unknown"
            for encoding in face_encodings:
                matches = self._compare(encoding)
                name = pick_name(matches, self.data["names"])
                print(name)
                waktu = datetime.now().strftime("%H:%M:%S")
                msg.timestamp = waktu

                if name != "Unknown":
                    # Parsing Nama_NIM
                    if "_" in name:
                        nama_asli, nim = name.split("_", 1)
                    else:
                        nama_asli, nim = name, "0000"

                    # Cek Absen Harian & Insert DB
                    tgl = datetime.now().strftime("%Y-%m-%d")

                    key = (nim, tgl)

                    if key not in self.absen_hari_ini:
                        self.cursor.execute(
                            "INSERT INTO absensi (nim, nama, tanggal, waktu) VALUES (?, ?, ?, ?)",
                            (nim, nama_asli, tgl, waktu))
                        self.conn.commit()
                        self.absen_hari_ini.add(key)
                        log = f"✅ ABSEN: {nama_asli}"
                        detected_names.append(log)
                        self.get_logger().info(log)
                    else:
                        detected_names.append(f"👌 SUDAH ABSEN: {nama_asli}")
                else:
                    detected_names.append("❓ WAJAH TIDAK DIKENAL")

            # 3. Kirim Balik ke Raspi
            print(name)
            msg.label = name
            self.pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main():
    rclpy.init()
    node = FaceRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()