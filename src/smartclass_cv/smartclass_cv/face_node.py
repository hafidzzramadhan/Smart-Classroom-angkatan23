import rclpy
from rclpy.node import Node 

from vision_msgs.msg import Pose2D 
from vision_msgs.msg import BoundingBox2D
from smartclass_msgs.msg import BondingBoxCv 
from smartclass_msgs.msg import FrameCv
from cv_bridge import CvBridge
from datetime import datetime
from ultralytics import YOLO
import os
import cv2
from ament_index_python.packages import get_package_share_directory


class FaceNode(Node):
    def __init__(self):
        super().__init__('face_node')

        # =========================
        # PATH HANDLING (ROS2 WAY)
        # =========================
        pkg_path = get_package_share_directory('smartclass_cv')

        # Model path (read-only, from install/share)
        self.model_path = os.path.join(pkg_path, 'models', 'face.pt')

        # =========================
        # LOAD YOLO MODEL
        # =========================
        self.model = YOLO(self.model_path)

        # =========================
        # ROS2 COMMUNICATION
        # =========================
        self.sub = self.create_subscription(
            FrameCv,
            '/cv/frame',
            self.listener_callback,
            10
        )

        # Publisher untuk mengirimkan gambar dan bounding box dalam satu pesan
        
        self.pub = self.create_publisher(
            BondingBoxCv,  # Menggunakan pesan kustom
            '/cv/bondingbox/face',
            10
        )

        self.br = CvBridge()

        self.get_logger().info("✅ NODE READY - Face Detection")

    def listener_callback(self, msg: FrameCv):
        try:
            if msg.frame_sampling.data > 0:
                # 1. ROS Image → OpenCV
                frame = self.br.imgmsg_to_cv2(msg.frame_image, desired_encoding='bgr8')
                 
                # 2. YOLO face Inference
                results = self.model(frame)
                for result in results:
                    # Get detections as a pandas DataFrame
                    detections = result.boxes.xyxy  # Bounding boxes in [x1, y1, x2, y2] format
                    confidences = result.boxes.conf  # Confidence scores
                    class_ids = result.boxes.cls     # Class IDs
                    bondingboxid = 1
                    # Convert to numpy for OpenCV
                    bboxes = detections.cpu().numpy() if hasattr(detections, "cpu") else detections
                    for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = map(int, bbox) 
                        # Create timestamp
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # 3. Crop the image using bounding box coordinates
                        cropped_image = frame[y1:y2, x1:x2]  # Crop the image
                        filename = f"/home/agung/ros2_ws/saved_detected_person/face/frame_{timestamp}_{bondingboxid}.png"
                        #self.save_image(cropped_image,filename)
                        
                        # Convert cropped image to ROS Image
                        cropped_ros_image = self.br.cv2_to_imgmsg(cropped_image, encoding="bgr8")

                        # 4. Publish cropped image and bounding box info as a single message
                        cropped_msg = BondingBoxCv()
                        cropped_msg.camera_id = msg.camera_id
                        cropped_msg.frame_id =  msg.frame_id
                        cropped_msg.frame_sampling = msg.frame_sampling
                        cropped_msg.frame_sequence.data = msg.frame_sequence.data 
                        cropped_msg.bondingbox_id.data = bondingboxid
                        cropped_msg.cropped_image = cropped_ros_image  # Assigning cropped image to message
                        cropped_msg.bondingbox = self.bondingbox2d(x1,y1,x2,y2)
                        cropped_msg.timestamp = timestamp  # Adding timestamp to the message

                        # Publish the message
                        self.pub.publish(cropped_msg)
                        bondingboxid += 1

        except Exception as e:
            self.get_logger().error(f"❌ Error: {e}")

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
    
def main():
    rclpy.init()
    node = FaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
