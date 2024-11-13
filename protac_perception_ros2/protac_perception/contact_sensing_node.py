import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from protac_interfaces.msg import ContactState

from .tacsense import TactilePerception, DeformationSensing
import numpy as np

class ContactDetectionPublisher(Node):

    def __init__(self):
        super().__init__('contact_detection_publisher')
        self.publisher_ = self.create_publisher(ContactState, '/protac_perception/contact_state', 10)
        timer_period = 0.03  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.tacitle_perception = DeformationSensing()
    
    def timer_callback(self):
        contact_depths, contact_positions, contact_radial_vectors = self.tacitle_perception.extract_contact_information()
        msg = ContactState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.contact_depth = contact_depths
        if len(msg.contact_depth)==0:
            msg.contact_depth = [np.float64(0), np.float64(0)]
        elif len(msg.contact_depth)==1:
            msg.contact_depth.append(np.float64(0))
        for contact_position in contact_positions:
            msg_contact_pos = Vector3()
            msg_contact_pos.x = 0.001*contact_position[0]
            msg_contact_pos.y = 0.001*contact_position[1]
            msg_contact_pos.z = 0.001*contact_position[2]
            msg.contact_location.append(msg_contact_pos)
        for contact_radial_vector in contact_radial_vectors:
            msg_contact_dir = Vector3()
            msg_contact_dir.x = contact_radial_vector[0]
            msg_contact_dir.y = contact_radial_vector[1]
            msg_contact_dir.z = contact_radial_vector[2]
            msg.contact_direction.append(msg_contact_dir)
        msg.num_of_contacts = len(contact_positions)
        self.get_logger().info('Contact State Publishing...!')
        self.publisher_.publish(msg)

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