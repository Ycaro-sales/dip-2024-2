import numpy as np
import cv2 as cv
import utils

# 5. Apply Edge Detection Filters (Sobel, Laplacian) on Color Images


def edge_detection_filters(img: np.ndarray):
    ddepth = cv.CV_16S
    kernel_size = 3
    scale = 1
    delta = 0

    red = img[utils.BGR_RED_SLICE]
    green = img[utils.BGR_GREEN_SLICE]
    blue = img[utils.BGR_BLUE_SLICE]

    sob_red = cv.Sobel(
        red,
        ddepth,
        1,
        0,
        ksize=kernel_size,
        scale=scale,
        delta=delta,
        borderType=cv.BORDER_DEFAULT,
    )

    sob_green = cv.Sobel(
        green,
        ddepth,
        1,
        0,
        ksize=kernel_size,
        scale=scale,
        delta=delta,
        borderType=cv.BORDER_DEFAULT,
    )

    sob_blue = cv.Sobel(
        blue,
        ddepth,
        1,
        0,
        ksize=kernel_size,
        scale=scale,
        delta=delta,
        borderType=cv.BORDER_DEFAULT,
    )

    lap_red = cv.Laplacian(red, ddepth)
    lap_green = cv.Laplacian(green, ddepth)
    lap_blue = cv.Laplacian(blue, ddepth)
