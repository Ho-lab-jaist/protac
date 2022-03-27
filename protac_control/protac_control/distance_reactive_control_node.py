import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
import numpy as np


from protac_kinematics.kinematics import Kinematics

class DistanceReactiveController(Node):
    def __init__(self):
        super().__init__('distance_reactivate_controller')

        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)

        self.subscription_cam3 = self.create_subscription(
            Float64,
            '/cam3/protac_perception/object_area',
            self.cam3_distance_callback,
            10)
        self.subscription_cam3  # prevent unused variable warning

        self.subscription_cam4 = self.create_subscription(
            Float64,
            '/cam4/protac_perception/object_area',
            self.cam4_distance_callback,
            10)
        self.subscription_cam4  # prevent unused variable warning

        # load kinematics module for protac (protac_kinematics package)
        self.kinematics = Kinematics()

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ts = timer_period # sampling time

        ps = np.array([0.3, -0.3, 0.4])
        pe = np.array([0.3, 0.3, 0.4])
        self.time_scale = 1
        Tf = 8
        self.trajectory_generation = self.jonit_trajectory_generation(ps, pe, Tf, 3)

        self.proximity = False
        self.distance_threshold = 100000


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
        if self.proximity:
            self.time_scale = 3
            self.get_logger().info('Slow Speed')
        else:
            self.time_scale = 1
            self.get_logger().info('Normal Speed')
        joint_cmd = next(self.trajectory_generation)
        # print('Yeilding time scale: {0}'.format(joint_cmd))
        msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]
        self.publisher_.publish(msg)

    def cam3_distance_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        area = msg.data
        if area > self.distance_threshold:
            self.proximity = True
        else:
            self.proximity = False

    def cam4_distance_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        area = msg.data
        if area > self.distance_threshold :
            self.proximity = True
        else:
            self.proximity= False


def main(args=None):
    rclpy.init(args=args)
    position_controller = DistanceReactiveController()
    rclpy.spin(position_controller)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    position_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
