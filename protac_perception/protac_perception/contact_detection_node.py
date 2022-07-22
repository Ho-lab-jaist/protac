import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from .tacsense import TactilePerception

class ContactDetectionPublisher(Node):

    def __init__(self):
        super().__init__('contact_detection_publisher')
        self.publisher_ = self.create_publisher(Bool, '/protac_perception/contact', 10)
        timer_period = 0.03  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.tacitle_perception = TactilePerception()
    def timer_callback(self):
        msg = Bool()
        if self.tacitle_perception.detect_contact():
            # Contact detection
            msg.data = True
        else:
            msg.data = False
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    contact_detection_publisher = ContactDetectionPublisher()

    rclpy.spin(contact_detection_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    contact_detection_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()