# Import the necessary libraries
from .midas.depth_inference import DepthProcessing
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import Float64

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from collections import deque
import time
import threading

def write_mp4(frames, fps, filepath):
    """
    Write provided frames to an .mp4 video.

    Args:
        frames (list): List of frames (np.ndarray).
        fps (int): Framerate (frames per second) of the output video.
        filepath (str): Path to output video file.
    """
    if not filepath.endswith(".mp4"):
        filepath += ".mp4"

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        filepath, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h)
    )

    for frame in frames:
        writer.write(frame)
    writer.release()


class VideoShower():
    def __init__(self, frame=None, win_name="Video"):
        """
        Class to show frames in a dedicated thread.

        Args:
            frame (np.ndarray): (Initial) frame to display.
            win_name (str): Name of `cv2.imshow()` window.
        """
        self.frame = frame
        self.win_name = win_name
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        Method called within thread to show new frames.
        """
        while not self.stopped:
            # We can actually see an ~8% increase in FPS by only calling
            # cv2.imshow when a new frame is set with an if statement. Thus,
            # set `self.frame` to None after each call to `cv2.imshow()`.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        cv2.destroyWindow(self.win_name)
        self.stopped = True


class DepthSensing(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('depth_sensing')

    # Define ROS parameters
    self.init_parameters()

    self.model_weights = self.get_parameter('model_weights').get_parameter_value().string_value
    self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
    self.optimize = self.get_parameter('optimize').get_parameter_value().bool_value
    self.show_fps = self.get_parameter('show_fps').get_parameter_value().bool_value
    self.raw_append = self.get_parameter('raw_append').get_parameter_value().bool_value
    self.frames = list() if self.get_parameter('output').get_parameter_value().string_value else None

    # Instantiate depth processing class
    self.depth_processing = DepthProcessing(self.model_weights, self.model_type, self.optimize)

    # Initialize Viewer
    self.video_shower = VideoShower(None, "Depth Map Stream").start()

    # Number of frames to average for computing FPS.
    self.num_fps_frames = 30
    self.previous_fps = deque(maxlen=self.num_fps_frames)

    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning

    self.start_time = time.time()

  def init_parameters(self):
    """
    Define ROS parameters
    """
    weights_descriptor = ParameterDescriptor(description='path to the trained weights of model.')
    type_descriptor = ParameterDescriptor(description='model type: dpt_large, dpt_hybrid, midas_v21 or midas_v21_small. [Default: dpt_large]')
    optimize_descriptor = ParameterDescriptor(description='Optimize CUDA performance')
    fps_descriptor = ParameterDescriptor(description="Display frames processed per second.")
    raw_append_descriptor = ParameterDescriptor(description="Display raw RGB image alongside the encoded depth map")
    output_descriptor = ParameterDescriptor(description="Path for writing output video file.")

    self.declare_parameter('model_weights', "/home/protac/ros/protac_ws/src/protac_perception/resource/midas/weights", weights_descriptor)
    self.declare_parameter('model_type', "dpt_large", type_descriptor)
    self.declare_parameter('optimize', True, optimize_descriptor)
    self.declare_parameter('show_fps', True, fps_descriptor)
    self.declare_parameter('raw_append', True, raw_append_descriptor)
    self.declare_parameter('output', "/home/protac/ros/protac_ws/src/protac_perception/resource/midas/outputs/test-depth.mp4", output_descriptor)

  def listener_callback(self, data):
    """
    Image receive callback function and depth processing.
    """
    loop_start_time = time.time()

    # Convert ROS Image message to OpenCV image
    frame = self.br.imgmsg_to_cv2(data)

    # Infer raw depth map (relative distance)
    depth_map = self.depth_processing.run(frame)
    # TODO: process depth sensing for high-level perception

    fps = int(sum(self.previous_fps) / self.num_fps_frames) if self.show_fps else None
    
    display = self.depth_processing.display_depth(bits=1, fps=fps, raw_append=self.raw_append)

    self.video_shower.frame = display

    if self.frames is not None:
        self.frames.append(display)

    self.previous_fps.append(int(1 / (time.time() - loop_start_time)))

def main():
  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  depth_processing = DepthSensing()
  try:
    # Spin the node so the callback function is called.
    rclpy.spin(depth_processing)
  except KeyboardInterrupt:
    depth_processing.video_shower.stop()
    depth_processing.get_logger().info('Stop Depth Perception')
  except Exception as e:
      depth_processing.video_shower.stop()
      raise e
  finally:
      if depth_processing.get_parameter('output').get_parameter_value().string_value and depth_processing.frames:
          # Get average FPS and write output at that framerate.
          fps = 1 / ((time.time() - depth_processing.start_time) / len(depth_processing.frames))
          write_mp4(depth_processing.frames, fps, depth_processing.get_parameter('output').get_parameter_value().string_value)
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      depth_processing.destroy_node()
      # Shutdown the ROS client library for Python
      rclpy.shutdown()
  
if __name__ == '__main__':
  main()