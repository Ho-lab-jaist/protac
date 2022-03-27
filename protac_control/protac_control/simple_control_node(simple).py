import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

from .tacsense import TactilePerception

class ProTacVelocityController(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        
        self.tacitle_perception = TactilePerception()
        self.publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.velocity_jnt1_ = 0.2
        self.velocity_jnt2_ = 0.2
        self.count = 0.01

    def timer_callback(self):
        self.velocity_jnt1_ += self.count
        if self.velocity_jnt1_ > 0.4:
            self.count = -0.01
        elif self.velocity_jnt1_ < -0.4:
            self.count = 0.01

        # condition fedback from tactile sensor
        while self.tacitle_perception.detect_contact():
            print('Contact Detected')
            msg = Float64MultiArray()
            msg.data = [0.0, 0.0]
            self.get_logger().info('Publishing: "%s"' % msg.data)
            self.publisher_.publish(msg)

        msg = Float64MultiArray()
        msg.data = [self.velocity_jnt1_, self.velocity_jnt2_]
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    velocity_controller = ProTacVelocityController()

    rclpy.spin(velocity_controller)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    velocity_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
