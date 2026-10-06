import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Point
from vision_msgs.msg import Pose2D 
from vision_msgs.msg import BoundingBox2D
from std_msgs.msg import String
from smartclass_msgs.msg import FrameCv
from smartclass_msgs.msg import BondingBoxCv
import datetime
import json
import cv2
import numpy as np
import os

from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class DetectionNode(Node):
    def __init__(self):
        super().__init__('DetectionNode')
        self.bridge = CvBridge()

        # === SUBSCRIBE IMAGE DARI LAPTOP TEMAN ===
        self.sub = self.create_subscription(
            FrameCv,
            '/cv/frame',
            self.callback,
            10
        )

        # === PUBLISH BOUNDING BOX ===
        self.pub_counting = self.create_publisher(
            String,
            '/cv/counting',
            10
        )
        
        self.pub_delete_frame = self.create_publisher(
            String,
            '/cv/frame/delete',
            20
        )
        
        self.pub_person = self.create_publisher(
            BondingBoxCv,
            '/cv/bondingbox/person',
            10
        )
        
        self.pub_nonperson = self.create_publisher(
            BondingBoxCv,
            '/cv/bondingbox/nonperson',
            10
        )

        self.bridge = CvBridge() 

        # === LOAD YOLO MODEL ===
        pkg_path = get_package_share_directory('smartclass_cv')
        model_path = os.path.join(pkg_path, 'models', 'best.onnx')

        self.get_logger().info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path, task='detect')

        self.get_logger().info("Detection Node Started")

    def callback(self, msg: FrameCv):
        # === IMAGE → OPENCV ===
        if msg.frame_sampling.data > 0: 
            
            frame = self.bridge.imgmsg_to_cv2(msg.frame_image, desired_encoding='bgr8')
            frame_id = msg.frame_id
            
            # === YOLO INFERENCE ===
            results = self.model(frame, verbose=False)

            detections = []

            for r in results:
                for box, cls, conf in zip(
                    r.boxes.xyxy,
                    r.boxes.cls,
                    r.boxes.conf
                ):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.model.names[int(cls)]

                    detections.append({
                        "label": label,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            counts = {}

            for det in detections:
                label = det["label"]
                counts[label] = counts.get(label, 0) + 1

            out = String()
            out.data = json.dumps({
                "frame_id": frame_id,
                "counts": counts
            })

            self.pub_counting.publish(out)
            
            # === PUBLISH JSON ===
            bondingid = 1
            for box in detections:
                msg_pub = BondingBoxCv()
                msg_pub.frame_id = frame_id
                msg_pub.frame_sampling = msg.frame_sampling
                msg_pub.frame_sequence.data = msg.frame_sequence.data
                msg_pub.bondingbox_id.data = bondingid
                msg_pub.bondingbox = self.bondingbox2d(box["bbox"][0],box["bbox"][1],box["bbox"][2],box["bbox"][3])  
                croppedimage = frame[box["bbox"][1]:box["bbox"][3], box["bbox"][0]:box["bbox"][2]]  
                msg_pub.cropped_image = self.bridge.cv2_to_imgmsg(croppedimage, encoding="bgr8")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                msg_pub.timestamp = timestamp
                 
                if (box["label"] == 'person'): 
                    filename = f"/home/agung/ros2_ws/saved_detected_person/frame_{timestamp}_{bondingid}.png"
                    #self.save_image(croppedimage,filename)
                    msg_pub.label = 'Person'
                    self.pub_person.publish(msg_pub)
                else:
                    if (box["label"] == 'chair'): 
                        filename = f"/home/agung/ros2_ws/saved_detected_nonperson/frame_{timestamp}_{bondingid}.png"
                        #self.save_image(croppedimage,filename)
                        msg_pub.label = 'Chair'
                        self.pub_nonperson.publish(msg_pub)
                    #else:
                       # self.pub_delete_frame.publish(msg_pub.frame_id)

                bondingid += 1 

    def bondingbox2d(self,x1,y1,x2,y2):
        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1, x2)
        max_y = max(y1, y2)
        
        bbox = BoundingBox2D()
 
        # Create the center Pose2D message
        center = Pose2D()

        # Calculate the center based on the bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Set the position (x, y) of the center
        center.position.x = float(center_x)
        center.position.y = float(center_y)
         
        # Set the orientation (theta) of the bounding box
        center.theta = 0.0  # Assuming no rotation (can be adjusted based on the bounding box's rotation)

        # Set the center in the BoundingBox2D message
        bbox.center = center

        # Optionally, set the size of the bounding box
        bbox.size_x = float(max_x - min_x)
        bbox.size_y = float(max_y - min_y)
        
        return bbox
        
    def save_image(self, frame,filename):
        # Generate a filename with timestamp
        

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the frame as an image (you can change the file format if necessary)
        cv2.imwrite(filename, frame)
        self.get_logger().info(f"Saved frame to {filename}")
    
def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
