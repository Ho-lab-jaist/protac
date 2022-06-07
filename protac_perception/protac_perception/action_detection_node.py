import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from .tacsense import TactilePerception

import numpy as np

class ActionDetectionPublisher(Node):
    def __init__(self):
        super().__init__('action_detection_publisher')
        self.publisher_ = self.create_publisher(String, '/protac_perception/action', 10)
        timer_period = 0.03  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.tacitle_perception = TactilePerception()
        self.prev_position = self.tacitle_perception.extract_contact_positions()[0]
        self.contact_count = 0
        self.action = "No"

    def timer_callback(self):
        contact_positions = self.tacitle_perception.extract_contact_positions()
        if self.tacitle_perception.detect_contact() and len(contact_positions)==1:
            self.contact_count += 1
            current_position = contact_positions[0]
            if self.contact_count >= 4:
                directional_vector = current_position - self.prev_position
                distance = np.linalg.norm(directional_vector)
                if distance > 2: # unit: mm
                    if directional_vector[2] > 0:
                        sign = "+"
                    else:
                        sign = "-"
                    self.action = "Stroke({0})".format(sign)
                else:
                    # print(distance)
                    self.action = "Press"
            self.prev_position = current_position
        elif self.tacitle_perception.detect_contact() and len(contact_positions)>=2:
            # print('Multi-point contact: {0}'.format(len(contact_positions)))
            pass
        else:
            self.action = "No"
            self.contact_count = 0
        
        msg = String()
        msg.data = self.action
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)

    contact_detection_publisher = ActionDetectionPublisher()

    rclpy.spin(contact_detection_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    contact_detection_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()