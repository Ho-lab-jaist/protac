# Import the necessary libraries
from .tacsense import ForceSensing

import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes

from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

import numpy as np
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import PIL

import scipy.signal
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

class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        if np.isnan(x):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")

class LiveLFilter(LiveFilter):
    def __init__(self, b, a):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy.
            a (array-like): denominator coefficients obtained from scipy.
        """
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)

    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y

class ForcePerception(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('force_perception')

    # define lowpass filter with 2.5 Hz cutoff frequency
    fs = 120 # sampling rate, Hz
    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
    self.live_lfilter = LiveLFilter(b, a)

    # Instantiate depth processing class
    self.force_sensing = ForceSensing()

    # Initialize Viewer
    self.video_shower = VideoShower(None, "RGB Tactile RBG Stream").start()

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

    self.contact_position_publisher = self.create_publisher(Vector3, '/protac_perception/contact_position', 10)
    self.contact_force_publisher = self.create_publisher(Float64, '/protac_perception/contact_force', 10)
    self.filtered_contact_force_publisher = self.create_publisher(Float64, '/protac_perception/filtered_contact_force', 10)
    self.contact_force_data = Float64()
    self.filtered_contact_force_data = Float64()
    self.contact_position_data = Vector3()
    
    self.num_fps_frames = 30
    self.previous_fps = deque(maxlen=self.num_fps_frames)
    self.start_time = time.time()

  def listener_callback(self, data):
    """
    Image receive callback function and depth processing.
    """
    loop_start_time = time.time()

    # Convert ROS Image message to OpenCV image
    frame_numpy = cv2.cvtColor(self.br.imgmsg_to_cv2(data), cv2.COLOR_BGR2RGB)
    tac_image = PIL.Image.fromarray(frame_numpy)

    contact_force, contact_position = self.force_sensing.compute_contact_force(tac_image)
    filtered_contact_force = self.live_lfilter(contact_force)
    # if contact_position[1] > 0:
    #     contact_force = -contact_force
    #     filtered_contact_force = -filtered_contact_force
    if abs(filtered_contact_force) < 0.15:
        filtered_contact_force = 0.
    self.contact_force_data.data = np.float64(-contact_force)
    self.filtered_contact_force_data.data = np.float64(-filtered_contact_force)
    self.contact_position_data.x = contact_position[0]
    self.contact_position_data.y = contact_position[1]
    self.contact_position_data.z = contact_position[2]

    self.contact_force_publisher.publish(self.contact_force_data)
    self.filtered_contact_force_publisher.publish(self.filtered_contact_force_data)
    self.contact_position_publisher.publish(self.contact_position_data)

    fps = int(sum(self.previous_fps) / self.num_fps_frames)
    self.get_logger().info('Fps: {0}Hz'.format(fps))
    self.previous_fps.append(int(1 / (time.time() - loop_start_time)))
    
    self.video_shower.frame = frame_numpy


def main():
  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  force_processing = ForcePerception()
  try:
    # Spin the node so the callback function is called.
    rclpy.spin(force_processing)
  except KeyboardInterrupt:
    force_processing.video_shower.stop()
    force_processing.get_logger().info('Stop Force Sensing')
  except Exception as e:
      force_processing.video_shower.stop()
      raise e
  finally:
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      force_processing.destroy_node()
      # Shutdown the ROS client library for Python
      rclpy.shutdown()
  
if __name__ == '__main__':
  main()