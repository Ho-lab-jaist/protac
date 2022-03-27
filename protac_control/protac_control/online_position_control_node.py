import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import numpy as np


from protac_kinematics.kinematics import Kinematics

class OnlinePositionController(Node):
    def __init__(self):
        super().__init__('online_position_controller')

        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)

        self.subscription_ = self.create_subscription(
            Bool,
            '/protac_perception/contact',
            self.contact_callback,
            10)
        self.subscription_  # prevent unused variable warning

        # load kinematics module for protac (protac_kinematics package)
        self.kinematics = Kinematics()

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ts = timer_period # sampling time

        ps = np.array([0.3, -0.3, 0.4])
        pe = np.array([0.3, 0.3, 0.4])
        self.time_scale = 1
        Tf = 5
        self.trajectory_generation = self.jonit_trajectory_generation(ps, pe, Tf, 3)

        self.contact = False

    def jonit_trajectory_generation(self, pointstart, pointend, Tf, method):
        paths = [{"start": pointstart, "end": pointend, "duration": Tf}, {"start": pointend, "end": pointstart, "duration": Tf}]
        while True:
            for path in paths:
                t = 0
                while t <= path["duration"]:
                    # update trajectory time
                    t_plan = t*self.time_scale
                    # plan for trajectory
                    if method == 3:
                        s, _ = self.kinematics.CubicTimeScaling(path["duration"]*self.time_scale, t_plan)
                    else:
                        s, _ = self.kinematics.QuinticTimeScaling(path["duration"]*self.time_scale, t_plan)
                    pos = s * np.array(path["end"]) + (1 - s) * np.array(path["start"])
                    # calculate jont position (by inverse kinematics)
                    joint = self.kinematics.IKinPos(pos)[0]
                    yield joint
                    # update dynamic time stamp
                    t = t + self.ts/self.time_scale

    def timer_callback(self):   
        msg = Float64MultiArray()
        if self.contact:
            self.time_scale = 2
        else:
            self.time_scale = 1
        joint_cmd = next(self.trajectory_generation)
        # print('Yeilding time scale: {0}'.format(joint_cmd))
        msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]
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
    position_controller = OnlinePositionController()
    rclpy.spin(position_controller)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    position_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
