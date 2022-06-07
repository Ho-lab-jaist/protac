# Import the necessary libraries
from .midas.depth_inference import DepthProcessing
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import Float64
import numpy as np

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


class ImageDisplay(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('iamge_display')

    # Initialize Viewer
    self.video_shower = VideoShower(None, "RGB Stream").start()

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

  def listener_callback(self, data):
    """
    Image receive callback function and depth processing.
    """

    # Convert ROS Image message to OpenCV image
    frame = self.br.imgmsg_to_cv2(data)
    self.video_shower.frame = frame

def main():
  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  image_display = ImageDisplay()
  try:
    # Spin the node so the callback function is called.
    rclpy.spin(image_display)
  except KeyboardInterrupt:
    image_display.video_shower.stop()
  except Exception as e:
      image_display.video_shower.stop()
      raise e
  finally:
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      image_display.destroy_node()
      # Shutdown the ROS client library for Python
      rclpy.shutdown()
  
if __name__ == '__main__':
  main()