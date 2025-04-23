import cv2 as cv
import numpy as np

# 4. Compare Effects of Blurring in RGB vs HSV


def compare_gaussian_blurring(img: np.ndarray):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    blurred_rgb = cv.GaussianBlur(img, (5, 5), 0)
    blurred_hsv = cv.GaussianBlur(img_hsv, (5, 5), 0)

    cv.imshow("HSV Gaussian Blur", blurred_hsv)
    cv.imshow("RGB Gaussian Blur", blurred_rgb)

    cv.waitKey(0)
