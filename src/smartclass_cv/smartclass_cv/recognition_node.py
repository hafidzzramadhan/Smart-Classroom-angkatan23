import os
import pickle
import sqlite3
from datetime import datetime

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
        #                  (SQLite) TETAP di node ini, jadi data biometrik tidak
        #                  meninggalkan perangkat.
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
        if self.mode not in ('local', 'remote'):
            raise ValueError("parameter 'mode' harus 'local' atau 'remote'")
        self.tolerance = float(self.get_parameter('tolerance').value)

        # --- KONFIGURASI PATH ---
        work_dir = self.get_parameter('work_dir').value or os.path.expanduser(
            "~/smartclass_ws/src/smartclass_cv/face_registered")
        self.work_dir = work_dir
        self.pickle_path = os.path.join(self.work_dir, "encodings.pickle")
        self.db_path = os.path.join(self.work_dir, "absensi.db")

        # Load data wajah terdaftar
        self.data = pickle.loads(open(self.pickle_path, "rb").read())

        # Model / klien
        if self.mode == 'local':
            import face_recognition  # hanya dibutuhkan di mode local (butuh dlib)
            self.fr = face_recognition
        else:
            url = self.get_parameter('server_url').value.rstrip('/') + '/face_encoding'
            self.remote = RemoteModel(
                url,
                timeout=self.get_parameter('timeout_s').value,
                image_format=self.get_parameter('image_format').value,
                jpeg_quality=self.get_parameter('jpeg_quality').value,
            )
            self.stats = RttStats(self.get_logger(), 'recognition')
            self.get_logger().info(f"🌐 Remote encoding: {url}")

        # Setup DB
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.absen_hari_ini = set()

        # ROS Communications
        self.sub = self.create_subscription(BondingBoxCv, '/cv/bondingbox/face', self.listener_callback, 10)
        self.pub = self.create_publisher(BondingBoxCv, '/cv/bondingbox/face_identity', 10)
        self.br = CvBridge()
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

    def listener_callback(self, msg: BondingBoxCv):
        try:
            # 1. Convert Image
            frame = self.br.imgmsg_to_cv2(msg.cropped_image)

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
