import cv2
import subprocess
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CameraControl:
    def __init__(self, 
                 cam_id = 0, 
                 width = 640, 
                 height = 480, 
                 fps = 30, 
                 mjpg_enabled = False,
                 os: Literal["ubuntu", "windows"] = "ubuntu",
                 exposure_mode: Literal["automatic", "manual"] = "manual",
                 exposure_value: int = -6):
        
        self.cam_id = cam_id
        self._cam = cv2.VideoCapture(self.cam_id)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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

    def isOpened(self):
        return self._cam.isOpened()

    def release(self):
        self._cam.release()

    def read(self):
        frame = self._cam.read()[1]
        return frame

    def set_exposure(self, value: int):
        if self.exposure_mode == "automatic":
            self._cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            print("Switched to manual exposure mode")
        self._cam.set(cv2.CAP_PROP_EXPOSURE, value)

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

def create_nodal_radial_vectors(X0):
    """
    Create a list of fixed nodal inward radial vectors names (unit vectors)
    Parameters:
        - X0: undeformed nodal positions (N, 3)
    Returns:
        - N: nodal radial vectors (N, 3)
    """
    N = list()
    for x0 in X0:
        n = np.array([0, 0, x0[2]]) - x0
        n_unit = n/np.linalg.norm(n)  # normalize vector
        N.append(n_unit)
    return np.array(N)

def compute_directional_similarity(D, N):
    """
    Compute the directional similarity between two vectors
    which is measured by cos(\phi); \phi is angle between the two vectors
    cos(\phi) = D.N/||D||.||N||
    Parameters:
        - D: estimated nodal displacement vectors (N, 3)
    Returns:
        - N: nodal radial vectors (N, 3)
    """
    assert len(D)==len(N), 'The two vectors should be same size'
    Phi_ls = list()
    if len(D) > 1:
        for d, n in zip(D, N):
            Phi_ls.append(np.dot(d, n))    
    else:
        return np.dot(D, N)
    return np.array(Phi_ls)

def modify_displacement_direction(Dm, dir_signals):
    """
    Assign sign for for the displacement magnitude
    Parameters:
        - Dm: estimated displacement magnitudes (N, 1)
    Returns:
        - dir_signals: binary directonal signal (N, 1)
    """
    assert len(Dm)==len(dir_signals), 'The two vectors should be same size'
    signed_Dm = list()
    for dm, dir_sig in zip(Dm, dir_signals):
        if not dir_sig:
            signed_Dm.append(-dm)
        else:
            signed_Dm.append(dm)
    return np.array(signed_Dm)
    

def compute_signed_displacement(free_displacement_vectors, radial_vectors):
    """
    Compute nodal contact depths (displacments) with direction
    i.e., inward or outward deformation
    Parameters:
        - free_displacement_vectors: estimated nodal displacement vectors (N, 3)
        - radial_vectors: radial vectors
    Returns:
        - sigend nodal displacments (N, 3)
    """
    dir_sim = compute_directional_similarity(free_displacement_vectors, radial_vectors)
    dir_signals = dir_sim > 0.
    nodal_displacement_magnitude = np.linalg.norm(free_displacement_vectors, axis=1)
    return modify_displacement_direction(nodal_displacement_magnitude, dir_signals)

def setup_2dplot():
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Plot some data
    line, = ax.plot([], [])
    ax.set_ylim((0, 28))
    ax.set_xlabel("Time")
    ax.set_ylabel("Contact Depth")
    # Get the figure's canvas
    canvas = fig.canvas
    # Show the figure
    # canvas.show()
    plt.show(block=False)
    return canvas, ax, line