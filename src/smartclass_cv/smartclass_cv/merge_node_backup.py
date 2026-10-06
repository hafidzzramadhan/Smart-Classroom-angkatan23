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
from collections import defaultdict

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
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 2)
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
            
    def stream_frame(self,stream):
        #print(f"frame ke - {stream.data}")
        #print(f'publish status {self.pub_status}')
        #print(f'Current Frame - {self.CurrentFrame.frame_sequence.data}')
        self.QStream.put(stream, block=False)
        if self.pub_status == "1" and self.QStream.qsize()>0:
            stream = self.QStream.get()
            print(f'pub stream = {stream.data}')
            if self.CurrentFrame == None:
                if self.QFrame.qsize() > 0:
                    self.CurrentFrame = self.QFrame.get() 
            if self.CurrentFrame != None:
                if stream.data == self.CurrentFrame.frame_sequence.data:
                    #print(f'publish frame ke - {self.CurrentFrame.frame_sequence.data}')
                    self.publish(self.CurrentFrame,"1")  
                    if self.QFrame.qsize() > 0:
                        self.CurrentFrame = self.QFrame.get() 
                else: 
                    if stream.data > self.CurrentFrame.frame_sequence.data and self.QFrame.qsize() > 0:
                        self.CurrentFrame = self.QFrame.get() 
                    self.publish(self.CurrentFrame,"") 
                    #print(f'same frame {self.CurrentFrame.frame_sequence.data}')  
           
            self.lastPublish.data = stream.data
            print(f'publish - {stream.data}')
        else:
            print(f'unpublish - {stream.data}')
        
    def publish(self,frames,mode):  
        msg = ResultCv()
        cv_image = self.bridge.imgmsg_to_cv2(frames.frame_image, desired_encoding='bgr8') 
        if mode == "1":
            bondings = BondingBoxCv() 
            msg.camera_id = frames.camera_id 
            msg.frame_id = frames.frame_id 
            msg.frame_sequence = frames.frame_sequence.data
            msg.frame_sampling = frames.frame_sampling.data
            msg.timestamp_pub = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #convert image to base64 
            #cv_image = self.bridge.imgmsg_to_cv2(frames.frame_image, desired_encoding='bgr8')  
            bondings =  None
            bondingstatus = ""
            if str(frames.frame_sequence.data) in self.QBox: 
                bondings = self.QBox.pop(str(frames.frame_sequence.data),None) 
                if bondings is not None: 
                    for box in bondings:
                        #print(f":label - {box.label}")
                        boxs = BondingBox()
                        boxs.label = box.label
                        boxs.confident_level = box.confident_level
                        boxs.name = box.name 
                        boxs.bondingbox = box.bondingbox
                        msg.bondingboxs.append(boxs) 
                    self.CurrentResult = msg
                    #print('Bonding')
            cv_image = draw_box_label(cv_image,bondings) 
            _, buffer = cv2.imencode('.jpg', cv_image)
            msg.image_base64 = base64.b64encode(buffer).decode('utf-8')
            self.base64.data = msg.image_base64 
            
            
        if self.CurrentResult != None:
            msg = self.CurrentResult 
          
        success, compressed_img = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not success:
            raise ValueError("Image compression failed!")

        # compressed_img is a NumPy array of bytes
        compressed_bytes = compressed_img.tobytes()
        self.pub_foxglove.publish(compressed_bytes)
        #self.pub_web.publish(self.base64)
        #self.Qweb.put(self.base64, block=False) 
        self.pub_result.publish(msg)  
        #print(f'frame ke-{msg.frame_sequence.data}');
    
    def stream_callback(self,msg: Int64):
        print(f'stream ke- {msg.data}')
        self.stream_frame(msg) 
         
    def frame_callback(self,msg: FrameCv): 
        self.QFrame.put(msg, block=False) 
        
        
         
    def recognition_callback(self,msg: BondingBoxCv): 
        msg.name = "recognition" 
        print(f'box frame {msg.name}- {msg.frame_sequence.data}')
        if self.lastPublish.data  < msg.frame_sequence.data:
            self.QBox.setdefault(str(msg.frame_sequence.data), []).append(msg)
            self.pub_status = "1"
        else:
            self.pub_status = "0"
            
    def emotion_callback(self,msg: BondingBoxCv):
        msg.name = "expression" 
        print(f'box frame {msg.name}- {msg.frame_sequence.data}')
        if self.lastPublish.data < msg.frame_sequence.data:
            self.QBox.setdefault(str(msg.frame_sequence.data), []).append(msg)
            self.pub_status = "1"
        else:
            self.pub_status = "0"
            
    def activity_callback(self,msg: BondingBoxCv):
        msg.name = "activity" 
        print(f'box frame {msg.name}- {msg.frame_sequence.data}')
        if self.lastPublish.data < msg.frame_sequence.data: 
            self.QBox.setdefault(str(msg.frame_sequence.data), []).append(msg)
            self.pub_status = "1"
        else:
            self.pub_status = "0"
        
        
    def gesture_callback(self,msg: BondingBoxCv):
        msg.name = "gesture" 
        print(f'box frame {msg.name}- {msg.frame_sequence.data}')
        if self.lastPublish.data < msg.frame_sequence.data: 
            self.QBox.setdefault(str(msg.frame_sequence.data), []).append(msg)
            self.pub_status = "1"
        else:
            self.pub_status = "0"
         
        
        
def main():
    rclpy.init()
    node = BondingBoxMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
