import rclpy
from rclpy.node import Node 
import subprocess
from std_msgs.msg import String
 

class WekeonNode(Node):

    def __init__(self):
        super().__init__('WekeonNode')
        self.get_logger().info('Standby has been started')
        
        self.sub = self.create_subscription(
            String,
            '/activate',
            self.callback,
            10
        )
        self.pub = self.create_publisher(
            String,
            '/activate/status',
            10
        )
        
        
    def callback(self,msg: String):
        try:
            
            if msg.data == 'gesture': 
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/gesture_node.py'])
                msg.data = 'Gesture On'
                self.pub.publish(msg)
            if msg.data == 'activity':
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/activity_node.py'])
                msg.data = 'Activity On'
                self.pub.publish(msg)
            if msg.data == 'merge':
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/merge_node.py'])
                msg.data = 'Merge On'
                self.pub.publish(msg)
            if msg.data == 'recognition':
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/recognition_node.py'])
                msg.data = 'Recognition On'
                self.pub.publish(msg)
            if msg.data == 'expression': 
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/src/smartclass_cv/smartclass_cv/expression_node.py'])
                msg.data = 'Emotion On'
                self.pub.publish(msg)
            if msg.data == 'face':
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/face_node.py'])
                msg.data = 'Face On'
                self.pub.publish(msg)
            if msg.data == 'detection':
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/detection_node.py'])
                msg.data = 'Detection On'
                self.pub.publish(msg)
            if msg.data == 'frame':
                subprocess.Popen(['python3', 'src/smartclass_cv/smartclass_cv/frame_node.py'])  
                msg.data = 'Frame On'
                self.pub.publish(msg)
            
            self.get_logger().error(f"All nodes already run")
                
        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WekeonNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
