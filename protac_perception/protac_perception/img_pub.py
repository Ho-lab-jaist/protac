import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
import threading
from collections import deque
import time

class ImagePublisher(Node):
  """
  Create an ImagePublisher class, which is a subclass of the Node class.
  """
  def __init__(self, args):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_publisher')

    # Define ROS parameters
    self.init_parameters()

    # Create the publisher. This publisher will publish an Image
    # to the video_frames topic. The queue size is 10 messages.
    self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
      
    # # We will publish a message every 0.1 seconds
    # timer_period = 0.01 # seconds
      
    # # Create the timer
    # self.timer = self.create_timer(timer_period, self.timer_callback)
        
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
   
    # Number of frames to average for computing FPS.
    self.num_fps_frames = 30
    self.previous_fps = deque(maxlen=self.num_fps_frames)

    cam_id = self.get_parameter('cam_id').get_parameter_value().integer_value
    self.cap = cv2.VideoCapture(cam_id)
    # fps = self.cap.get(cv2.CAP_PROP_FPS)
    # print('fps: {}'.format(fps))
    self.k_new = self.get_parameter('k_new').get_parameter_value().double_value
    self.K = np.array(self.get_parameter('K').get_parameter_value().double_array_value).reshape(3,3)
    self.K_new = np.array(self.K)
    self.K_new[(0, 1), (0, 1)] = self.k_new * self.K_new[(0, 1), (0, 1)]
    self.D = np.array(self.get_parameter('d').get_parameter_value().double_array_value)
    self.undistored = np.array(self.get_parameter('image_undistored').get_parameter_value().bool_value)

    self.get_logger().info('{}'.format(self.K.shape))

    self.grabbed, self.frame = self.cap.read()
    self.stopped = False
         
  def init_parameters(self):
    """
    Define ROS parameters
    """
    cam_id_descriptor = ParameterDescriptor(description='Camera id [Default 0]')
    width_descriptor = ParameterDescriptor(description='Width of the image')
    height_descriptor = ParameterDescriptor(description='Height of the image')
    k_new_descriptor = ParameterDescriptor(description='refined scaler parameter [Default 0.9]')
    K_descriptor = ParameterDescriptor(description='Intrinsic calibration matrix')
    d_descriptor = ParameterDescriptor(description='Distortion parameters')
    undistored_descriptor = ParameterDescriptor(description='Undistored or Original image')

    self.declare_parameter('cam_id', 0, cam_id_descriptor)
    self.declare_parameter('width', 640, width_descriptor)
    self.declare_parameter('height', 480, height_descriptor)
    self.declare_parameter('k_new', 0.9, k_new_descriptor)
    self.declare_parameter('K', [239.35312993382436, 0.00000000000000, 308.71493813687908,
                                 0.00000000000000, 239.59440270542146, 226.24771387864561,
                                 0.00000000000000, 0.00000000000000, 1.00000000000000], K_descriptor)
    self.declare_parameter('d', [-0.04211185968680, 0.00803630431552, -0.01334505838778, 0.00370625371074], d_descriptor)
    self.declare_parameter('image_undistored', False, undistored_descriptor)


  def start(self):
      threading.Thread(target=self.get, args=()).start()
      return self

  def get(self):
      """
      Method called in a thread to continually read frames from `self.cap`.
      This way, a frame is always ready to be read. Frames are not queued;
      if a frame is not read before `get()` reads a new frame, previous
      frame is overwritten.
      """
      while not self.stopped:
          if not self.grabbed:
              self.stop()
          else:
              self.grabbed, self.original_frame = self.cap.read()
              if self.undistored:
                self.frame = cv2.fisheye.undistortImage(
                              self.original_frame,
                              self.K,
                              D=self.D ,
                              Knew=self.K_new,
                )
              else:
                self.frame = self.original_frame

  def stop(self):
      self.stopped = True

def main(args=None):
  # Number of frames to average for computing FPS.
  num_fps_frames = 30
  previous_fps = deque(maxlen=num_fps_frames)

  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  image_publisher = ImagePublisher(args).start()
  
  try:
    while True:
      loop_start_time = time.time()

      if image_publisher.stopped:
        image_publisher.stop()
        break

      frame = image_publisher.frame
      image_publisher.publisher_.publish(image_publisher.br.cv2_to_imgmsg(frame))
      image_publisher.get_logger().info('Publishing video frame: {0}(fps)'.format(int(sum(previous_fps) / num_fps_frames)))

      previous_fps.append(int(1 / (time.time() - loop_start_time)))
  except KeyboardInterrupt:
      image_publisher.stop()
      image_publisher.get_logger().info('Stop Publishing')
  finally:
      # Destroy the node explicitly
      image_publisher.destroy_node()

  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()