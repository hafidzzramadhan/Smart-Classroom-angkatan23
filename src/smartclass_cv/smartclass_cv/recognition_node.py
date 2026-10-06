import rclpy
from rclpy.node import Node
from smartclass_msgs.msg import BondingBoxCv

from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, pickle, sqlite3, face_recognition, os
from datetime import datetime

class FaceRecognition(Node):
    def __init__(self):
        super().__init__('recognition_node')
        
        # --- KONFIGURASI PATH (UBAH SESUAI USERNAME VM KAMU) ---
        self.work_dir = "/home/hafidz/smartclass_ws/src/smartclass_cv/face_registered"
        self.pickle_path = os.path.join(self.work_dir, "encodings.pickle")
        self.db_path = os.path.join(self.work_dir, "absensi.db")
        
        # Load Model
        self.data = pickle.loads(open(self.pickle_path, "rb").read())
        
        # Setup DB
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.absen_hari_ini = set()
        
        # ROS Communications
        self.sub = self.create_subscription(BondingBoxCv, '/cv/bondingbox/face', self.listener_callback, 10)
        self.pub = self.create_publisher(BondingBoxCv, '/cv/bondingbox/face_identity', 10) 
        self.br = CvBridge()
        self.get_logger().info("SERVER READY: Menunggu gambar dari face detection...")

    def listener_callback(self, msg: BondingBoxCv):
        try:
            # 1. Convert Image
             
            frame = self.br.imgmsg_to_cv2(msg.cropped_image)
             
            # Simpan gambar ke folder /img/subscribe
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            
            # 2. Face Recognition (Logic sesuai camera_recognition.py)
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get the face encodings (a vector representing the face)
            face_encodings = face_recognition.face_encodings(face_rgb)
            
            detected_names = []
            name = "Unknown"
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=0.45)
                name = "Unknown"
                
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        n = self.data["names"][i]
                        counts[n] = counts.get(n, 0) + 1
                    name = max(counts, key=counts.get)
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
                        self.cursor.execute("INSERT INTO absensi (nim, nama, tanggal, waktu) VALUES (?, ?, ?, ?)", 
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
