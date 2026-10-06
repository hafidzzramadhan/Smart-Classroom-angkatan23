import rclpy
from rclpy.node import Node
import requests  # Importing the requests library to make the REST API call

class RestApiNode(Node):

    def __init__(self):
        super().__init__('rest_api_node')
        self.get_logger().info('RestApiNode has been started')
        
        # Define the API endpoint
        self.api_url = "https://jsonplaceholder.typicode.com/posts/1"
        
        # Call the REST API when the node starts
        self.call_api()

    def call_api(self):
        try:
            # Make a GET request to the API
            response = requests.get(self.api_url)
            
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                self.get_logger().info(f"API Response: {data}")
            else:
                self.get_logger().error(f"Failed to call API. Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"An error occurred: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RestApiNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
