import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import numpy as np


from protac_kinematics.kinematics import Kinematics

class ProTacPositionController(Node):
    def __init__(self):
        super().__init__('protac_position_controller')

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
        Tf = 10
        self.trajectory_generation = self.joint_trajectory_generation(ps, pe, Tf)

        self.contact = False

    def joint_trajectory_generation(self, pointstart, pointend, Tf):
        N = Tf/self.ts + 1.0
        traj, trajdot = self.kinematics.LineTrajectoryGeneration(pointstart, pointend, Tf, N, 3)
        traj = np.concatenate((traj, np.flip(traj, axis=0)))
        trajdot = np.concatenate((trajdot, -trajdot))
        while True:
            for pos in traj:
                joint = self.kinematics.IKinPos(pos)[0]
                yield joint

    def timer_callback(self):   
        msg = Float64MultiArray()
        if self.contact:
            pass
        else:
            joint_cmd = next(self.trajectory_generation)

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
    position_controller = ProTacPositionController()
    rclpy.spin(position_controller)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    position_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
