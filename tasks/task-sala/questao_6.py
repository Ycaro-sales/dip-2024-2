import numpy as np
import cv2 as cv
from utils import BGR_BLUE_SLICE, BGR_GREEN_SLICE, BGR_RED_SLICE


# 6. High-pass and Low-pass FIltering in the Frequency Domain
# Tools: cv2.dft(), cv2.idft(), numpy.fft


def high_low_pass_filtering(img: np.ndarray):
    red = img[BGR_RED_SLICE]
    blue = img[BGR_BLUE_SLICE]
    green = img[BGR_GREEN_SLICE]
