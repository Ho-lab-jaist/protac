#!/usr/bin/env python3
import cv2
import numpy as np
cam_bot = cv2.VideoCapture(4)
cam_top = cv2.VideoCapture(6)

while cam_bot.isOpened():
    frame_top = cam_top.read()[1]
    frame_bot = cam_bot.read()[1]
    # Display the resulting frame
    cv2.imshow("Top Camera", frame_top)
    # cv2.imshow("Shifted Top Camera", shifted_top)
    cv2.imshow("Bottom Camera", frame_bot)
    # cv2.imshow("Shifted Bottom Camera", shifted_bot)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam_top.release()
cam_bot.release()
cv2.destroyAllWindows()
