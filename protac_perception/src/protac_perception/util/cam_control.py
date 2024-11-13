import threading
import cv2
import numpy as np
import subprocess
from typing import Literal


class CameraControl:
    def __init__(self, 
                 cam_id=0, 
                 width=640, 
                 height=480, 
                 fps=30, 
                 mjpg_enabled = False,
                 undistored = False,
                 os: Literal["ubuntu", "windows"] = "ubuntu",
                 exposure_mode: Literal["automatic", "manual"] = "manual",
                 exposure_value:int = -7):
        self.cam_id = cam_id
        self._cam = cv2.VideoCapture(self.cam_id)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._undistored = undistored
        self.k_new = 0.9
        self.K = np.array(
                    [[239.35312993382436, 0.00000000000000, 308.71493813687908],
                    [0.00000000000000, 239.59440270542146, 226.24771387864561],
                    [0.00000000000000, 0.00000000000000, 1.00000000000000]])
        self.D = np.array([-0.04211185968680, 
                    0.00803630431552, 
                    -0.01334505838778, 
                    0.00370625371074])
        self.K_new = np.array(self.K)
        self.K_new[(0, 1), (0, 1)] = self.k_new * self.K_new[(0, 1), (0, 1)]
            
        if fps==120:
            if mjpg_enabled:
                self._cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self._cam.set(cv2.CAP_PROP_FPS, fps)
            else:
                print("Please enable MJPG for fps 120Hz")
        else:
            self._cam.set(cv2.CAP_PROP_FPS, fps)

        self.exposure_mode = exposure_mode
        self.os = os
        if self.os == "windows":    
            if self.exposure_mode == "manual":
                self.set_exposure_mode("manual")
                self.set_exposure(exposure_value)
            elif self.exposure_mode == "automatic":
                self.set_exposure_mode("automatic")
        elif self.os == "ubuntu":
            if self.exposure_mode == "manual":
                # switch to manual mode
                command = f"v4l2-ctl -d {self.cam_id} -c auto_exposure=1"
                subprocess.call(command, shell=True)
                # set exposure mode
                command = f"v4l2-ctl -d {self.cam_id} -c exposure_time_absolute={exposure_value}"
                subprocess.call(command, shell=True)
            elif self.exposure_mode == "automatic":
                command = f"v4l2-ctl -d {self.cam_id} -c auto_exposure=3"
                subprocess.call(command, shell=True)

        # remove initial unstable frames
        if self.isOpened():
            for _ in range(50):
                _ = self.read()
        else:
            print("Failed to open camera")
    
    def isOpened(self):
        return self._cam.isOpened()

    def release(self):
        self._cam.release()

    def read(self):
        frame = self._cam.read()[1]
        if self._undistored:
            frame = cv2.fisheye.undistortImage(
                              frame,
                              self.K,
                              D = self.D ,
                              Knew = self.K_new)
        return frame

    def set_image_perspective(self, value: bool):
        self._undistored  = value

    def set_exposure(self, value: int):
        if self.os == "windows":    
            if self.exposure_mode == "automatic":
                self.set_exposure_mode("manual")
            self._cam.set(cv2.CAP_PROP_EXPOSURE, value)
        elif self.os == "ubuntu":
            if self.exposure_mode == "automatic":
                command = f"v4l2-ctl -d {self.cam_id} -c auto_exposure=1"
                subprocess.call(command, shell=True)
                self.exposure_mode = "manual"
            # set exposure mode
            command = f"v4l2-ctl -d {self.cam_id} -c exposure_time_absolute={value}"
            subprocess.call(command, shell=True)

    def set_exposure_mode(self, mode: Literal["automatic", "manual"]):
        if mode == "automatic":
            self._cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.exposure_mode = "automatic"
            print("Switched to Automatic exposure mode")
        elif mode =="manual":
            self._cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.exposure_mode = "manual"
            print("Switched to Manual exposure mode")
        else:
            print("Invalid exposure mode")

    def set_brightness(self, value: int):
        """
        brighness: [-64, 64]
        """
        if self.os == 'windows':
            self._cam.set(cv2.CAP_PROP_BRIGHTNESS, value)
        if self.os == 'ubuntu':
            command = f"v4l2-ctl -d {self.cam_id} -c brightness={value}"
            subprocess.call(command, shell=True)

    def set_contrast(self, value: int):
        """
        contrast: [0, 64]
        """
        if self.os == 'windows':
            self._cam.set(cv2.CAP_PROP_CONTRAST, value)
        if self.os == 'ubuntu':
            command = f"v4l2-ctl -d {self.cam_id} -c contrast={value}"
            subprocess.call(command, shell=True)


class VideoShower():
    def __init__(self, 
                 frame = None, 
                 win_name = "RGB"):
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
        display_thread = threading.Thread(target=self.show)
        # Daemonize the thread (will terminate with the main program)
        display_thread.daemon = True  
        display_thread.start()
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