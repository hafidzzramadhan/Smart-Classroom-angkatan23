"""Pilot recognition: one CSV row per face crop, monotonic callback timings.

Uses the supplied pilot database, preserving recognition matching and ROS output.
Timing includes original prints/logs; CSV writing is outside callback timing.
face_encodings includes the library's internal face detection. Publish is not delivery.
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

EXPECTED_SOURCE = 'c397b228e297634ddc40e8255d088d3b7a173794746c3870e6b82d9afa1202e1'
FIELDS = ['callback_index', 'input_metadata_json', 'frame_id', 'camera_id', 'frame_sequence',
          'bondingbox_id', 'input_timestamp', 'output_timestamp',
          'callback_start_monotonic_ns', 'callback_end_monotonic_ns',
          'input_width', 'input_height', 'n_encodings', 'n_known', 'n_unknown',
          'n_db_inserts', 'n_published', 'output_label', 'encoding_names_json',
          't_cv2_ms', 't_rgb_ms', 't_face_encodings_ms', 't_matching_ms',
          't_db_ms', 't_publish_ms', 't_total_ms', 'status', 'error']


def make_node_class(original):
    class RecognitionProbe(original.FaceRecognition):
        def __init__(self, db_path, writer, stream):
            # Reproduce original initialization with only a separate DB path.
            original.Node.__init__(self, 'recognition_node')
            self.work_dir = '/home/hafidz/smartclass_ws/src/smartclass_cv/face_registered'
            self.pickle_path = os.path.join(self.work_dir, 'encodings.pickle')
            self.db_path = str(db_path)
            self.data = pickle.loads(Path(self.pickle_path).read_bytes())
            self.conn = sqlite3.connect(db_path.as_uri() + '?mode=rw', uri=True,
                                        check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.absen_hari_ini = set()
            self.sub = self.create_subscription(original.BondingBoxCv,
                '/cv/bondingbox/face', self.listener_callback, 10)
            self.pub = self.create_publisher(original.BondingBoxCv,
                '/cv/bondingbox/face_identity', 10)
            self.br = original.CvBridge()
            self.writer, self.stream = writer, stream
            self.count = 0
            self.get_logger().info('SERVER READY: Menunggu gambar dari face detection...')

        def listener_callback(self, msg):
            # Capture incoming metadata before original code mutates the message.
            row = {field: '' for field in FIELDS}
            metadata = {}
            if hasattr(msg, 'get_fields_and_field_types'):
                for field in msg.get_fields_and_field_types():
                    value = getattr(msg, field)
                    if isinstance(value, (str, int, float, bool)):
                        metadata[field] = value
            row['input_metadata_json'] = json.dumps(metadata, ensure_ascii=False)
            row.update(callback_index=self.count + 1,
                frame_id=getattr(msg, 'frame_id', ''),
                camera_id=getattr(msg, 'camera_id', ''),
                frame_sequence=getattr(msg, 'frame_sequence', ''),
                bondingbox_id=getattr(msg, 'bondingbox_id', ''),
                input_timestamp=getattr(msg, 'timestamp', ''),
                n_encodings=0, n_known=0, n_unknown=0, n_db_inserts=0,
                n_published=0, status='ok')
            for field in ('t_cv2_ms', 't_rgb_ms', 't_face_encodings_ms',
                          't_matching_ms', 't_db_ms', 't_publish_ms'):
                row[field] = 0.0
            names = []

            def measured(field, function):
                start = time.monotonic_ns()
                try:
                    return function()
                finally:
                    row[field] += (time.monotonic_ns() - start) / 1e6

            start = time.monotonic_ns()
            try:
                frame = measured('t_cv2_ms', lambda: self.br.imgmsg_to_cv2(msg.cropped_image))
                row['input_height'], row['input_width'] = frame.shape[:2]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                face_rgb = measured('t_rgb_ms', lambda: original.cv2.cvtColor(frame, original.cv2.COLOR_BGR2RGB))
                encodings = measured('t_face_encodings_ms', lambda: original.face_recognition.face_encodings(face_rgb))
                row['n_encodings'] = len(encodings)
                detected_names = []
                name = 'Unknown'
                for encoding in encodings:
                    def match_name():
                        matches = original.face_recognition.compare_faces(
                            self.data['encodings'], encoding, tolerance=0.45)
                        result = 'Unknown'
                        if True in matches:
                            matchedIdxs = [i for i, b in enumerate(matches) if b]
                            counts = {}
                            for i in matchedIdxs:
                                n = self.data['names'][i]
                                counts[n] = counts.get(n, 0) + 1
                            result = max(counts, key=counts.get)
                        return result
                    name = measured('t_matching_ms', match_name)
                    names.append(name)
                    row['n_unknown' if name == 'Unknown' else 'n_known'] += 1
                    print(name)
                    waktu = datetime.now().strftime('%H:%M:%S')
                    msg.timestamp = waktu
                    if name != 'Unknown':
                        if '_' in name:
                            nama_asli, nim = name.split('_', 1)
                        else:
                            nama_asli, nim = name, '0000'
                        tgl = datetime.now().strftime('%Y-%m-%d')
                        key = (nim, tgl)
                        if key not in self.absen_hari_ini:
                            def insert():
                                self.cursor.execute('INSERT INTO absensi (nim, nama, tanggal, waktu) VALUES (?, ?, ?, ?)',
                                                    (nim, nama_asli, tgl, waktu))
                                self.conn.commit()
                            measured('t_db_ms', insert)
                            row['n_db_inserts'] += 1
                            self.absen_hari_ini.add(key)
                            log = f'✅ ABSEN: {nama_asli}'
                            detected_names.append(log)
                            self.get_logger().info(log)
                        else:
                            detected_names.append(f'👌 SUDAH ABSEN: {nama_asli}')
                    else:
                        detected_names.append('❓ WAJAH TIDAK DIKENAL')
                print(name)
                msg.label = name
                measured('t_publish_ms', lambda: self.pub.publish(msg))
                row['n_published'] = 1
            except Exception as exc:
                self.get_logger().error(f'Error: {exc}')
                row['status'] = 'error'
                row['error'] = f'{type(exc).__name__}: {exc}'
            end = time.monotonic_ns()
            row.update(callback_start_monotonic_ns=start,
                callback_end_monotonic_ns=end, t_total_ms=(end-start)/1e6,
                output_timestamp=getattr(msg, 'timestamp', ''),
                output_label=getattr(msg, 'label', ''),
                encoding_names_json=json.dumps(names, ensure_ascii=False))
            self.count += 1
            self.writer.writerow(row)
            self.stream.flush()
    return RecognitionProbe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output', required=True)
    parser.add_argument('--db-path', required=True, help='Existing pilot DB; never use original absensi.db')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    db = Path(args.db_path).resolve()
    real_db = Path('/home/hafidz/smartclass_ws/src/smartclass_cv/face_registered/absensi.db').resolve()
    if (not db.is_file() or db == real_db or
            (real_db.exists() and os.path.samefile(db, real_db))):
        parser.error('Provide an existing separate pilot database')
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as check:
        columns = {r[1] for r in check.execute('PRAGMA table_info(absensi)')}
        if not {'nim', 'nama', 'tanggal', 'waktu'} <= columns:
            parser.error('Pilot DB lacks required absensi columns')
        initial_rows = check.execute('SELECT COUNT(*) FROM absensi').fetchone()[0]
    from smartclass_cv import recognition_node as original
    source = Path(inspect.getfile(original)).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE:
        parser.error('Original recognition source differs from reviewed SHA256')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Python:', sys.executable)
    print('Original source:', source, '\nSHA256:', digest)
    print('Probe SHA256:', hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print('Pilot DB:', db, '\nInitial absensi rows:', initial_rows)
    enc_path = real_db.parent / 'encodings.pickle'
    print('Encodings SHA256:', hashlib.sha256(enc_path.read_bytes()).hexdigest())
    print('Tolerance: 0.45; QoS depth: 10; one row per face crop')
    node = None
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        original.rclpy.init()
        try:
            init_start = time.monotonic_ns()
            node = make_node_class(original)(db, writer, stream)
            print(f'Node initialization: {(time.monotonic_ns()-init_start)/1e6:.2f} ms')
            print('Recording about', args.seconds, 'seconds; each row is one face crop.', flush=True)
            started = time.monotonic_ns()
            try:
                while original.rclpy.ok() and (time.monotonic_ns()-started)/1e9 < args.seconds:
                    original.rclpy.spin_once(node, timeout_sec=0.1)
            except KeyboardInterrupt:
                print('Interrupted by user; partial run retained.')
            print(f'Recorded {node.count} callbacks over {(time.monotonic_ns()-started)/1e9:.3f} monotonic seconds.')
            print('Final absensi rows:', node.cursor.execute('SELECT COUNT(*) FROM absensi').fetchone()[0])
        finally:
            if node is not None:
                node.conn.close()
                node.destroy_node()
            if original.rclpy.ok():
                original.rclpy.shutdown()
    print('CSV saved:', output)


if __name__ == '__main__':
    main()
