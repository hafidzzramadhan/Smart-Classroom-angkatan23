import rclpy
from rclpy.node import Node 
from std_msgs.msg import String 

class FrameDel(Node):
    def __init__(self):
        super().__init__("frame_delete_node")

        # publisher image
        self.sub = self.create_subcriber(String, "/cv/frame/delete", self.listener_callback,100) 

    def listener_callback(self,msg):
        # Generate a filename with timestamp 
        filename = f"/home/agung/ros2_ws/saved_frames/frame_{msg.data}.png"

        # Ensure the directory exists
        os.remove(filename)

        # Save the frame as an image (you can change the file format if necessary)
         
        self.get_logger().info(f"Remove File {filename}")
 
def main():
    rclpy.init()
    node = FrameDel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
