import numpy as np
import cv2 as cv


def linear_combination(img1, img2, scalar1, scalar2):
    h = scalar1 * img1 + scalar2 * img2
    return h


img = cv.imread("./images.jpeg")
cv.imshow("Image", img)
cv.waitKey(0)

