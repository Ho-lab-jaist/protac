import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Vector3

from .tacsense import TactilePerception

import numpy as np

class ActionDetectionPublisher(Node):
    def __init__(self):
        super().__init__('action_detection_publisher')
        self.publisher_ = self.create_publisher(String, '/protac_perception/action', 10)
        self.press_direction_publisher_ = self.create_publisher(Vector3, '/protac_perception/press_direction', 10)
        timer_period = 0.03  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.tacitle_perception = TactilePerception()
        self.init_positions = self.tacitle_perception.extract_contact_positions()[0]
        self.prev_position = self.tacitle_perception.extract_contact_positions()[0]
        self.contact_count = 0
        self.action = "No"

        # message for velocity command
        self.press_direction = Vector3()

        self.distance_intervals = list()

    def timer_callback(self):
        contact_positions, contact_radial_vectors, contact_depths = self.tacitle_perception.extract_contact_positions()
        if self.tacitle_perception.detect_contact() and len(contact_positions)==1:
            self.contact_count += 1
            current_position = contact_positions[0]           
            directional_vector = current_position - self.prev_position
            distance = np.linalg.norm(directional_vector)
            self.distance_intervals.append(distance)
            if self.contact_count == 1:
                self.init_positions = current_position
            if self.contact_count >= 8:
                # directional_vector = current_position - self.prev_position
                # distance = np.linalg.norm(directional_vector)

                distance_intervals_np = np.array(self.distance_intervals)
                reliance_count = np.count_nonzero(distance_intervals_np > 0.)

                if reliance_count > 0.3*len(distance_intervals_np): # unit: mm
                    stroke_direction = current_position - self.init_positions
                    if stroke_direction[2] > 0:
                        sign = "+"
                    else:
                        sign = "-"
                    self.action = "Stroke({0})".format(sign)
                else:

                    # print(distance)
                    self.action = "Press"
                    # TODO: get press direction
                    self.press_direction.x = contact_depths[0]*contact_radial_vectors[0][0]
                    self.press_direction.y = contact_depths[0]*contact_radial_vectors[0][1]
                    self.press_direction.z = contact_depths[0]*contact_radial_vectors[0][2]
                    self.press_direction_publisher_.publish(self.press_direction)

            self.prev_position = current_position
        elif self.tacitle_perception.detect_contact() and len(contact_positions) >= 2:
            press_direction_1 = contact_radial_vectors[0]
            press_direction_2 = contact_radial_vectors[1]
            z1 = contact_positions[0][2]
            z2 = contact_positions[1][2]
            x_axis = np.array([1, 0, 0])
            if np.dot(press_direction_1, press_direction_2) < -0.8:
                if z1 > z2 and np.dot(press_direction_1, x_axis) >= 0.7071:
                    self.action = "My(+)"
                elif z1 < z2 and np.dot(press_direction_1, x_axis) >= 0.7071:
                    self.action = "My(-)"
                elif z1 > z2 and np.dot(press_direction_1, x_axis) < 0.7071:
                    self.action = "Mx(+)"
                elif z1 < z2 and np.dot(press_direction_1, x_axis) < 0.7071:
                    self.action = "Mx(-)"
        else:
            self.action = "No"
            self.contact_count = 0
            self.distance_intervals = list()
        
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