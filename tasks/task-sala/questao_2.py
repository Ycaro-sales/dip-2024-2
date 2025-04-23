import cv2 as cv
import numpy as np

# 2. Visualize Individual Color Channels


def visualize_color_channels(img: np.ndarray):
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]

    cv.imshow("red channel", red)
    cv.imshow("green channel", green)
    cv.imshow("blue channel", blue)

    cv.waitKey()
