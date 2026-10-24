"""One row per frame callback. Original gesture logic, models, QoS and executor.
Monotonic processing timings; original wall-clock publish limiter is preserved.
CSV writing is outside total timing. Publish timing is not delivery latency.
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import threading
import time

EXPECTED_SOURCE = '2ca5995ad80d16a9b4ba0d26c3859443f0989b53fd0c33acf9efd535575bb39d'
FIELDS = ['callback_index','frame_id','camera_id','frame_sequence',
 'callback_start_monotonic_ns','callback_end_monotonic_ns',
 'input_width','input_height','n_faces','n_hands','n_publish_attempts',
 'n_published','n_rate_limited','attempt_labels_json','published_labels_json',
 't_cv_bridge_ms','t_face_detect_ms','t_hand_detect_ms',
 't_publish_method_ms','t_publish_call_ms','t_total_ms','status','error']


def make_node_class(original):
    class Proxy:
        def __init__(self, target, node, method, field, count_field=None):
            self.target,self.node,self.method,self.field,self.count_field = target,node,method,field,count_field
        def __getattr__(self, name):
            value = getattr(self.target, name)
            if name != self.method:
                return value
            def measured(*args, **kwargs):
                row = getattr(self.node.local, 'row', None)
                if row is None:
                    return value(*args, **kwargs)
                start = time.monotonic_ns()
                try:
                    result = value(*args, **kwargs)
                    if self.count_field:
                        landmarks = getattr(result, 'face_landmarks' if self.count_field == 'n_faces' else 'hand_landmarks', None)
                        row[self.count_field] = len(landmarks) if landmarks else 0
                    if self.method == 'publish':
                        row['n_published'] += 1
                        row['_published_labels'].append(args[0].label)
                    if self.method == 'imgmsg_to_cv2':
                        if result is None:
                            row['status'] = 'empty_frame'
                        else:
                            row['input_height'],row['input_width'] = result.shape[:2]
                    return result
                except Exception as exc:
                    row['status'] = 'error'
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    row[self.field] += (time.monotonic_ns()-start)/1e6
            return measured

    class GestureProbe(original.GestureInferenceNode):
        def __init__(self, writer, stream):
            self.local = threading.local()
            self.write_lock = threading.Lock()
            self.count_lock = threading.Lock()
            self.count = 0
            self.writer,self.stream = writer,stream
            super().__init__()
            self.bridge = Proxy(self.bridge,self,'imgmsg_to_cv2','t_cv_bridge_ms')
            self.face_landmarker = Proxy(self.face_landmarker,self,'detect','t_face_detect_ms','n_faces')
            self.hand_landmarker = Proxy(self.hand_landmarker,self,'detect','t_hand_detect_ms','n_hands')
            self.pub_gesture = Proxy(self.pub_gesture,self,'publish','t_publish_call_ms')

        def publish_gesture(self, label, msg):
            row = getattr(self.local, 'row', None)
            if row is None:
                return super().publish_gesture(label,msg)
            row['n_publish_attempts'] += 1
            row['_attempt_labels'].append(label)
            before = row['n_published']
            start = time.monotonic_ns()
            try:
                result = super().publish_gesture(label,msg)
                if row['n_published'] == before:
                    row['n_rate_limited'] += 1
                return result
            finally:
                row['t_publish_method_ms'] += (time.monotonic_ns()-start)/1e6

        def image_callback(self,msg):
            with self.count_lock:
                self.count += 1
                index = self.count
            sequence = getattr(msg,'frame_sequence','')
            sequence = getattr(sequence,'data',sequence)
            row = {field:'' for field in FIELDS}
            row.update(callback_index=index,frame_id=getattr(msg,'frame_id',''),
                camera_id=getattr(msg,'camera_id',''),frame_sequence=sequence,
                n_publish_attempts=0,n_published=0,n_rate_limited=0,status='ok',
                _attempt_labels=[],_published_labels=[])
            for key in ('t_cv_bridge_ms','t_face_detect_ms','t_hand_detect_ms',
                        't_publish_method_ms','t_publish_call_ms'):
                row[key]=0.0
            self.local.row=row
            start=time.monotonic_ns()
            try:
                return super().image_callback(msg)
            except Exception as exc:
                row['status']='error'
                row['error']=f'{type(exc).__name__}: {exc}'
                raise
            finally:
                end=time.monotonic_ns()
                row.update(callback_start_monotonic_ns=start,callback_end_monotonic_ns=end,
                           t_total_ms=(end-start)/1e6,
                           attempt_labels_json=json.dumps(row.pop('_attempt_labels'),ensure_ascii=False),
                           published_labels_json=json.dumps(row.pop('_published_labels'),ensure_ascii=False))
                self.local.row=None
                with self.write_lock:
                    self.writer.writerow(row)
                    self.stream.flush()
    return GestureProbe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds<=0:
        parser.error('--seconds must be positive and finite')
    from smartclass_cv import gesture_node as original
    source=Path(inspect.getfile(original)).resolve()
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=EXPECTED_SOURCE:
        parser.error('Gesture source differs from reviewed SHA256')
    print('Python:',sys.executable)
    print('Original source:',source,'\nSHA256:',digest)
    print('Probe SHA256:',hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print('MediaPipe version:',getattr(original.mp,'__version__','unavailable'))
    for filename in ('face_landmarker.task','hand_landmarker.task'):
        path=Path(original.get_package_share_directory('smartclass_cv'))/'models'/filename
        print('Model:',path,'\nSHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
    print('Original MultiThreadedExecutor; depth 10; wall-clock publish interval 0.5 seconds')
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    node=None
    executor=None
    with output.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        original.rclpy.init()
        try:
            init_start=time.monotonic_ns()
            node=make_node_class(original)(writer,stream)
            executor=original.MultiThreadedExecutor()
            executor.add_node(node)
            print(f'Node initialization: {(time.monotonic_ns()-init_start)/1e6:.2f} ms')
            print('Recording about',args.seconds,'seconds; each row is one frame callback.',flush=True)
            started=time.monotonic_ns()
            try:
                while original.rclpy.ok() and (time.monotonic_ns()-started)/1e9<args.seconds:
                    executor.spin_once(timeout_sec=0.1)
            except KeyboardInterrupt:
                print('Interrupted by user; partial run retained.')
            finally:
                executor.remove_node(node)
                executor.shutdown()
                executor=None
            print(f'Recorded {node.count} callbacks over {(time.monotonic_ns()-started)/1e9:.3f} monotonic seconds.')
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.hand_landmarker.close()
                node.face_landmarker.close()
                node.destroy_node()
            if original.rclpy.ok():
                original.rclpy.shutdown()
    print('CSV saved:',output)


if __name__=='__main__':
    main()
