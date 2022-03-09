import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool


class ProTacVelocityController(Node):
    def __init__(self):
        super().__init__('protac_velocity_controller')

        self.publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)

        self.subscription_ = self.create_subscription(
            Bool,
            '/protac_perception/contact',
            self.contact_callback,
            10)
        self.subscription_  # prevent unused variable warning

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.velocity_jnt1_ = 0.0
        self.velocity_jnt2_ = 0.0
        self.velocity_trajectory1 = self.velocity_trajectory_generation(0., 0.4, 0.01)
        self.velocity_trajectory2 = self.velocity_trajectory_generation(0., 0.4, 0.01)

        self.contact = False

    def velocity_trajectory_generation(self, start_velocity, bound_velocity, step):
        current_velocity = start_velocity
        while True:
            yield current_velocity
            if current_velocity > bound_velocity or current_velocity < -bound_velocity:
                step = -step
            current_velocity += step

    def timer_callback(self):   
        msg = Float64MultiArray()
        if self.contact:
            self.velocity_jnt1_ = 0.
            self.velocity_jnt2_ = 0.
        else:
            self.velocity_jnt1_ = next(self.velocity_trajectory1)
            # self.velocity_jnt1_ = 0.
            self.velocity_jnt2_ = next(self.velocity_trajectory2)
            # self.velocity_jnt2_ = 0.

        msg.data = [self.velocity_jnt1_, self.velocity_jnt2_]
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.publisher_.publish(msg)

    def contact_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        if msg.data:
            self.contact = True
        else:
            self.contact = False


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
