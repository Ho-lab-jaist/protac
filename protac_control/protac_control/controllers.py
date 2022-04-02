from rclpy.node import Node

from sensor_msgs.msg import JointState

import numpy as np

from protac_kinematics.kinematics import Kinematics

class JointControllerBase(Node):
  """
  Base class for the Joint Position Controllers. It subscribes to the C{joint_states} topic by default.
  """
  def __init__(self, namespace, timeout):
    """
    JointControllerBase constructor. It subscribes to the C{joint_states} topic and informs after 
    successfully reading a message from the topic.
    @type namespace: string
    @param namespace: Override ROS namespace manually. Useful when controlling several robots 
    @type  timeout: float
    @param timeout: Time in seconds that will wait for the controller
    from the same node.
    """
    super().__init__('joint_controller_base')
    self.ns = namespace
    # Set-up publishers/subscribers
    self._js_sub = self.create_subscription(
            JointState, 
            '%sjoint_states' % self.ns, 
            self.joint_states_cb, 
            1)
    # self.get_logger().info('Waiting for [%sjoint_states] topic' % self.ns)
    # start_time = self.get_clock().now().nanoseconds
    # while not hasattr(self, '_joint_names'):
    #   if (self.get_clock().now().nanoseconds - start_time)/1e9 > timeout:
    #     self.get_logger().error('Timed out waiting for joint_states topic')
    #     return
    #   time.sleep(0.01)
    # self._num_joints = len(self._joint_names)
    self.get_logger().info('Topic [%sjoint_states] found' % self.ns)
    
  def get_joint_efforts(self):
    """
    Returns the current joint efforts of the Protac robot.
    @rtype: numpy.ndarray
    @return: Current joint efforts of the Protac robot.
    """
    return np.array(self._current_jnt_efforts)
  
  def get_joint_positions(self):
    """
    Returns the current joint positions of the Protac robot.
    @rtype: numpy.ndarray
    @return: Current joint positions of the Protac robot.
    """
    return np.array(self._current_jnt_positions)


  def get_joint_velocities(self):
    """
    Returns the current joint velocities of the Protac robot.
    @rtype: numpy.ndarray
    @return: Current joint velocities of the Protac robot.
    """
    return np.array(self._current_jnt_velocities)  

  def get_eef_positions(self):
    """
    Returns the current end-effector positions of the Protac robot.
    @rtype: numpy.ndarray
    @return: Current 3D eef positions of the Protac robot.
    """
    return self.kinematics.FKinSpace(self.get_joint_positions())[:3, 3]
  
  def joint_states_cb(self, msg):
    """
    Callback executed every time a message is publish in the C{joint_states} topic.
    @type  msg: sensor_msgs/JointState
    @param msg: The JointState message published by the RT hardware interface.
    """
    valid_joint_names = ['joint1','joint2','joint3','joint4']
    position = []
    velocity = []
    effort = []
    name = []
    for joint_name in valid_joint_names:
      if joint_name in msg.name:
        idx = msg.name.index(joint_name)
        name.append(msg.name[idx])
        position.append(msg.position[idx])
        velocity.append(msg.velocity[idx])
        effort.append(msg.effort[idx])
    if set(name) == set(valid_joint_names):
      self._current_jnt_positions = np.array(position)
      self._current_jnt_velocities = np.array(velocity)
      self._current_jnt_efforts = np.array(effort)
      self._joint_names = list(name)