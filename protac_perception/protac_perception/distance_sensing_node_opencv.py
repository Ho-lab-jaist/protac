# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

import numpy as np
import argparse
import os


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

    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning

    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
 
    # load passed arguments
    self.output = args["output"]
    self.confidence = args["confidence"]
    self.threshold = args["threshold"]
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    self.LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
      dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    self.ln = self.net.getLayerNames()
    self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    # pointer to output video file, dimension,
    # and recorded fps
    self.writer = None
    (self.W, self.H) = (None, None)
    self.fs = 10

  def listener_callback(self, data):
    """
    Callback function.
    """
    # Convert ROS Image message to OpenCV image
    frame = self.br.imgmsg_to_cv2(data)
    
    # if the frame dimensions are empty, grab them
    if self.W is None or self.H is None:
      (self.H, self.W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
      swapRB=True, crop=False)
    self.net.setInput(blob)
    layerOutputs = self.net.forward(self.ln)
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
      # loop over each of the detections
      for detection in output:
        # extract the class ID and confidence (i.e., probability)
        # of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > self.confidence:
          # scale the bounding box coordinates back relative to
          # the size of the image, keeping in mind that YOLO
          # actually returns the center (x, y)-coordinates of
          # the bounding box followed by the boxes' width and
          # height
          box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
          (centerX, centerY, width, height) = box.astype("int")
          # use the center (x, y)-coordinates to derive the top
          # and and left corner of the bounding box
          x = int(centerX - (width / 2))
          y = int(centerY - (height / 2))
          # update our list of bounding box coordinates,
          # confidences, and class IDs
          boxes.append([x, y, int(width), int(height)])
          confidences.append(float(confidence))
          classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
    # ensure at least one detection exists
    if len(idxs) > 0:
      # loop over the indexes we are keeping
      for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # draw a bounding box rectangle and label on the frame
        color = [int(c) for c in self.COLORS[classIDs[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
          confidences[i])
        cv2.putText(frame, text, (x, y - 5),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video stream", frame)
    cv2.waitKey(1)
    # check if the video writer is None
    if self.writer is None:
      # initialize our video writer
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      self.writer = cv2.VideoWriter(self.output, fourcc, self.fs,
        (frame.shape[1], frame.shape[0]), True)
    # write the output frame to disk
    self.writer.write(frame)
  
def main(args=None):
  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-o", "--output", default='/home/protac/ros/protac_ws/src/protac_perception/resource/output/test.avi',
    help="path to output video")
  ap.add_argument("-y", "--yolo", default='/home/protac/ros/protac_ws/src/protac_perception/resource/yolo-coco',
    help="base path to YOLO directory")
  ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
  ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
  args = vars(ap.parse_args())

  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  image_subscriber = DistanceSensing(args)
  
  try:
    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)
  except KeyboardInterrupt:
    image_subscriber.writer.release()
  finally:
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      image_subscriber.destroy_node()
      # Shutdown the ROS client library for Python
      rclpy.shutdown()
  
if __name__ == '__main__':
  main()