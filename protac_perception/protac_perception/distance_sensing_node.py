# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import Float64

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import torch

from .yolov3.darknet import Darknet
from .yolov3.inference import inference, draw_boxes
import os
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


class DistanceSensing(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self, args):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('distance_sensing')

    # Define ROS parameters
    self.init_parameters()

    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    self.device = self.get_parameter('device').get_parameter_value().string_value
    if self.device.startswith("cuda") and not torch.cuda.is_available():
        self.get_logger().warn(
            "CUDA not available; falling back to CPU. Pass `-d cpu` or ensure "
            "compatible versions of CUDA and pytorch are installed.",
        )
        self.device = "cpu"

    self.net = Darknet(self.get_parameter('config').get_parameter_value().string_value, device=self.device)
    self.net.load_weights(self.get_parameter('weights').get_parameter_value().string_value)
    self.net.eval()

    if self.device.startswith("cuda"):
        self.net.cuda(device=self.device)

    if self.get_parameter('verbose').get_parameter_value().bool_value:
        if self.device == "cpu":
            device_name = "CPU"
        else:
            device_name = torch.cuda.get_device_name(self.net.device)
        self.get_logger().info(f"Running model on {device_name}")

    self.class_names = None
    class_names_path = self.get_parameter('class_names').get_parameter_value().string_value
    if class_names_path is not None \
           and os.path.isfile(class_names_path):
        with open(class_names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    self.prob_threshd = self.get_parameter('prob_thresh').get_parameter_value().double_value
    self.nms_iou_thresh = self.get_parameter('iou_thresh').get_parameter_value().double_value
    self.show_fps = self.get_parameter('show_fps').get_parameter_value().bool_value

    self.video_shower = VideoShower(None, "SensorView").start()
    # Number of frames to average for computing FPS.
    self.num_fps_frames = 30
    self.previous_fps = deque(maxlen=self.num_fps_frames)

    self.frames = list() if self.get_parameter('output').get_parameter_value().string_value else None

    # Create a publisher. This publisher will broadcast area of human bounding box
    # from which the protac control node received for purposeful reaction
    self.publisher_ = self.create_publisher(Float64, '/protac_perception/object_area', 10)

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
    iou_thresh_descriptor = ParameterDescriptor(description='Non-maximum suppression IOU threshold. [Default 0.3]')
    prob_thresh_descriptor = ParameterDescriptor(description='Detection probability threshold. [Default 0.5]')
    config_descriptor = ParameterDescriptor(description='[Required] Path to Darknet model config file.')
    weights_descriptor = ParameterDescriptor(description='[Required] Path to Darknet model weights file.')
    class_names_descriptor = ParameterDescriptor(description='Path to text file of class names. If omitted, class index is displayed instead of name.')
    device_descriptor = ParameterDescriptor(description="Device for inference ('cpu', 'cuda'). [Default 'cuda']")
    output_descriptor = ParameterDescriptor(description="Path for writing output video file.")
    fps_descriptor = ParameterDescriptor(description="Display frames processed per second (for --cam input).")
    verbose_descriptor = ParameterDescriptor(description="Verbose output")

    self.declare_parameter('iou_thresh', 0.3, iou_thresh_descriptor)
    self.declare_parameter('prob_thresh', 0.5, prob_thresh_descriptor)
    self.declare_parameter('config', "/home/protac/ros/protac_ws/src/protac_perception/resource/models/yolov3.cfg", config_descriptor)
    self.declare_parameter('weights', "/home/protac/ros/protac_ws/src/protac_perception/resource/models/yolov3.weights", weights_descriptor)
    self.declare_parameter('class_names', "/home/protac/ros/protac_ws/src/protac_perception/resource/models/coco.names", class_names_descriptor)
    self.declare_parameter('device', "cuda", device_descriptor)
    self.declare_parameter('output', "/home/protac/ros/protac_ws/src/protac_perception/resource/outputs/test-yolo.mp4", output_descriptor)
    self.declare_parameter('show_fps', True, fps_descriptor)
    self.declare_parameter('verbose', True, verbose_descriptor)

  def listener_callback(self, data):
    """
    Image receive callback function.
    """
    loop_start_time = time.time()

    # Convert ROS Image message to OpenCV image
    frame = self.br.imgmsg_to_cv2(data)

    bbox_tlbr, class_prob, class_idx = inference(
        self.net, frame, device="cuda", prob_thresh=self.prob_threshd,
        nms_iou_thresh=self.nms_iou_thresh
    )[0]

    if 0 in list(class_idx):
        msg = Float64()
        # publish the area human bounding box
        person_class_idx = list(class_idx).index(0)
        tl_x, tl_y, br_x, br_y = bbox_tlbr[person_class_idx]
        msg.data = float((br_x-tl_x)*(br_y-tl_y))
        self.publisher_.publish(msg)

    draw_boxes(
        frame, bbox_tlbr, class_prob=class_prob, class_idx=class_idx, class_names=self.class_names
    )

    if self.show_fps:
        cv2.putText(
            frame,  f"{int(sum(self.previous_fps) / self.num_fps_frames)} fps",
            (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
            (255, 255, 255)
        )

    self.video_shower.frame = frame
    if self.frames is not None:
        self.frames.append(frame)

    self.previous_fps.append(int(1 / (time.time() - loop_start_time)))

def main(args=None):
  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  image_subscriber = DistanceSensing(args)
  try:
    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)
  except KeyboardInterrupt:
    image_subscriber.video_shower.stop()
    image_subscriber.get_logger().info('Stop Sensing')
  except Exception as e:
      image_subscriber.video_shower.stop()
      raise e
  finally:
      if image_subscriber.get_parameter('output').get_parameter_value().string_value and image_subscriber.frames:
          # Get average FPS and write output at that framerate.
          fps = 1 / ((time.time() - image_subscriber.start_time) / len(image_subscriber.frames))
          write_mp4(image_subscriber.frames, fps, image_subscriber.get_parameter('output').get_parameter_value().string_value)
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      image_subscriber.destroy_node()
      # Shutdown the ROS client library for Python
      rclpy.shutdown()
  
if __name__ == '__main__':
  main()