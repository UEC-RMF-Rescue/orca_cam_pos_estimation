import rclpy
from rclpy.node import Node
import time

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage

from orca_msg.srv import JpegImage

DEBUG

class Estimation(Node):
    def __init__(self):
        super().__init__("estimation")
        # topic pub
        self.orca_00_pos_abs_publisher = self.create_publisher(Twist, "/orca_00/pos_abs", 10)
        self.orca_01_pos_abs_publisher = self.create_publisher(Twist, "/orca_01/pos_abs", 10)
        self.orca_02_pos_abs_publisher = self.create_publisher(Twist, "/orca_02/pos_abs", 10)
        
        # service
        self.cli = self.create_client(JpegImage, '/capture')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = JpegImage.Request()

        # timer
        timer_period = 0.8
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.latest_image = CompressedImage()

    def timer_callback(self):
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.img_callback)
    
    def img_callback(self, future):
        try:
            self.latest_image = future.result().image.data
        except Exception as e:
            self.get_logger().error(f"Service call failed : {e}")
        else:



def main(args=None):
    rclpy.init(args=args)

    estimation = Estimation()

    rclpy.spin(estimation)
    estimation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()