import cv2
import time
import torch
import os
import numpy as np
from collections import deque

from .yolov3.darknet import Darknet
from .yolov3.inference import inference
from .yolov3.inference import draw_boxes, map_area_to_distance
import rospkg

r = rospkg.RosPack()
path = r.get_path('protac_perception')
FULL_PATH = path + '/resource/models'

class ObjectDetection():
  """
  Create an ImageSubscriber class, 
  which is a subclass of the Node class.
  """
  def __init__(self, 
               output = False,
               show_fps = True,
               verbose = True):

    # Define ROS parameters
    self.init_parameters(output = output,
                         show_fps = show_fps,
                         verbose = verbose)

    if self.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        self.device = "cpu"

    self.net = Darknet(self.config, 
                       device="cuda")
    self.net.load_weights(self.weights)
    self.net.eval()

    if self.device.startswith("cuda"):
        self.net.to(self.device)

    if self.verbose:
        if self.device == "cpu":
            device_name = "CPU"
        else:
            device_name = torch.cuda.get_device_name(self.net.device)
        print("This is an INFO message. "
              "Running model on {}".format(device_name))

    self.class_names = None
    if self.class_names_path is not None \
           and os.path.isfile(self.class_names_path):
        with open(self.class_names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    # Number of frames to average for computing FPS.
    self.num_fps_frames = 30
    self.previous_fps = deque(maxlen=self.num_fps_frames)

  def init_parameters(self, output, show_fps, verbose):
    """
    Define parameters
    """
    self.nms_iou_thresh = 0.3
    self.prob_thresh = 0.5
    self.config = os.path.join(FULL_PATH, "yolov3.cfg")
    self.weights = os.path.join(FULL_PATH, "yolov3.weights")
    self.class_names_path = os.path.join(FULL_PATH, "coco.names")
    self.device = "cuda"
    self.output = output
    self.show_fps = show_fps
    self.verbose = verbose

  def processing(self, img, risk_score):
    """
    Image receive callback function.
    """
    loop_start_time = time.time()
    object_detected = False
    human_detected = False
    human_distance = 4.0
    frame = np.array(img)

    bbox_tlbr, class_prob, class_idx = inference(
        self.net, 
        frame, 
        device = self.device, 
        prob_thresh = self.prob_thresh,
        nms_iou_thresh = self.nms_iou_thresh
    )[0]

    if len(list(class_idx)) > 0:
        object_detected = True
        if 0 in list(class_idx):
            human_detected = True
            person_class_idx = list(class_idx).index(0)
            tl_x, tl_y, br_x, br_y = bbox_tlbr[person_class_idx]
            area = float( (br_x-tl_x)*(br_y-tl_y) )
            human_distance = map_area_to_distance(area)
        else:
            human_detected = False
            human_distance = 4.0
    else:
        object_detected = False
        human_detected = False
        human_distance = 4.0

    # msg = Float64()
    # if 0 in list(class_idx):
    #     # publish the area human bounding box
    #     person_class_idx = list(class_idx).index(0)
    #     tl_x, tl_y, br_x, br_y = bbox_tlbr[person_class_idx]
    #     msg.data = float((br_x-tl_x)*(br_y-tl_y))
    #     self.publisher_.publish(msg)
    # else:
    #     msg.data = 0.
    #     self.publisher_.publish(msg)

    draw_boxes(
        frame, 
        bbox_tlbr, 
        risk_score,
        class_prob = class_prob, 
        class_idx = class_idx, 
        class_names = self.class_names
    )

    if self.show_fps:
        cv2.putText(
            frame,  f"{int(sum(self.previous_fps) / self.num_fps_frames)} fps",
            (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
            (255, 255, 255)
        )

    self.previous_fps.append(int(1 / (time.time() - loop_start_time)))

    return frame, object_detected, human_detected, human_distance