import rclpy
from rclpy.node import Node
import time

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

from orca_msg.srv import JpegImage

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import math

from orca_cam_pos_estimation import ar_marker_binary

CAM = True
SAVE = True

class Estimation(Node):
    def __init__(self):
        super().__init__("estimation")
        # topic pub
        self.orca_00_pos_abs_publisher_ = self.create_publisher(Twist, "/orca_00/pos_abs", 10)
        self.orca_01_pos_abs_publisher_ = self.create_publisher(Twist, "/orca_01/pos_abs", 10)
        self.orca_02_pos_abs_publisher_ = self.create_publisher(Twist, "/orca_02/pos_abs", 10)

        self.cam_est_flag_publisher_ = self.create_publisher(Int32, "/cam_est_flag", 10)
        
        # service
        self.cli = self.create_client(JpegImage, '/capture')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = JpegImage.Request()

        # timer
        timer_period = 0.8
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.latest_image = CompressedImage()
        self.orca_pos_abs = []

        ############################################
        self.tf_buffer = Buffer()
        self.tf_listner = TransformListener(self.tf_buffer, self)

        self.whale_odom_subscriber_ = self.create_subscription(Odometry, "/whale/odom",
                                                                self.whale_callback, 10)
        self.orca_00_odom_subscriber_ = self.create_subscription(Odometry, "/orca_00/odom",
                                                                self.orca_00_callback, 10)
        self.orca_01_odom_subscriber_ = self.create_subscription(Odometry, "/orca_01/odom",
                                                                self.orca_01_callback, 10)
        self.orca_02_odom_subscriber_ = self.create_subscription(Odometry, "/orca_02/odom",
                                                                self.orca_02_callback, 10)
        self.whale_pos = [0,0]
        self.orca_00_pos = [0,0]
        self.orca_01_pos = [0,0]
        self.orca_02_pos = [0,0]
        ############################################


    def timer_callback(self):
        self.orca_pos_abs = []
        
        msg = Int32()
        msg.data = 0
        self.cam_est_flag_publisher_.publish(msg)
        
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.img_callback)

    def img_callback(self, future):
        try:
            self.latest_image = future.result().image.data
            if SAVE:
                with open('immg_cb.jpg', 'wb') as f:
                    f.write(self.latest_image)
        except Exception as e:
            self.get_logger().error(f"Service call failed : {e}")
        else:
            if CAM:
                self.orca_pos_abs = ar_marker_binary.detect_aruco_and_get_real_positions(self.latest_image)
                print(self.orca_pos_abs)
            ############################################
            else:
                self.orca_pos_abs.append( [ (self.orca_00_pos[0]-self.whale_pos[0]), (self.orca_00_pos[1]-self.whale_pos[1]) ] )
                self.orca_pos_abs.append( [ (self.orca_01_pos[0]-self.whale_pos[0]), (self.orca_01_pos[1]-self.whale_pos[1]) ] )
                self.orca_pos_abs.append( [ (self.orca_02_pos[0]-self.whale_pos[0]), (self.orca_02_pos[1]-self.whale_pos[1]) ] )
            #############################################

            msg = Twist()
            msg.linear.x = float(self.orca_pos_abs[0][0])
            msg.linear.y = float(self.orca_pos_abs[0][1])
            self.orca_00_pos_abs_publisher_.publish(msg)

            msg.linear.x = float(self.orca_pos_abs[1][0])
            msg.linear.y = float(self.orca_pos_abs[1][1])
            self.orca_01_pos_abs_publisher_.publish(msg)    

            msg.linear.x = float(self.orca_pos_abs[2][0])
            msg.linear.y = float(self.orca_pos_abs[2][1])
            self.orca_02_pos_abs_publisher_.publish(msg)
    
    #############################################
    def whale_callback(self, msg):
        if msg.child_frame_id == "whale_base_link":
            self.whale_pos[0] = msg.pose.pose.position.x
            self.whale_pos[1] = msg.pose.pose.position.y
    
    def orca_00_callback(self, msg):
        if msg.child_frame_id == "orca_00_base_link":
            self.orca_00_pos[0] = msg.pose.pose.position.x
            self.orca_00_pos[1] = msg.pose.pose.position.y

    def orca_01_callback(self, msg):
        if msg.child_frame_id == "orca_01_base_link":
            self.orca_01_pos[0] = msg.pose.pose.position.x
            self.orca_01_pos[1] = msg.pose.pose.position.y

    def orca_02_callback(self, msg):
        if msg.child_frame_id == "orca_02_base_link":
            self.orca_02_pos[0] = msg.pose.pose.position.x
            self.orca_02_pos[1] = msg.pose.pose.position.y
    #############################################


def main(args=None):
    rclpy.init(args=args)

    estimation = Estimation()

    rclpy.spin(estimation)
    estimation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
