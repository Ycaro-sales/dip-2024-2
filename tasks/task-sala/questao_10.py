from typing import Callable
import numpy as np
import cv2 as cv
from utils import BGR_BLUE_SLICE as blue_s
from utils import BGR_GREEN_SLICE as green_s
from utils import BGR_RED_SLICE as red_s


def create_equalizing_function(img: np.ndarray) -> Callable:
    hist, _ = np.histogram(img.ravel(), bins=256)
    total = img.size

    def equalizer(r: int):
        return int(hist.cumsum()[r] / total * 255)

    return equalizer


def histogram_equalization(img: np.ndarray, flag: int) -> np.ndarray:
    converted_img = cv.cvtColor(img, code=flag)

    eq_img = np.zeros(shape=img.shape, dtype=img.dtype)
    for channel in range(converted_img.shape[2]):
        curr_channel = converted_img[:, :, channel]
        channel_equalizer = create_equalizing_function(img)

        for i, v in np.ndenumerate(curr_channel):
            eq_img[i, channel] = channel_equalizer(v)

    return eq_img
