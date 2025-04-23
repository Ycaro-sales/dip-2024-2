import cv2 as cv
import numpy as np

# 3. Convert Between Color Spaces (RGB <-> HSV<LAB<YCrB, CMYK)


def convert_img_color_space(img: np.ndarray, flag: int):
    new_img = cv.cvtColor(img, flag)

    cv.imshow("Converted image", new_img)
    cv.imshow("Original Image", img)

    cv.waitKey(0)


img = cv.imread("./source.jpg")

new_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
