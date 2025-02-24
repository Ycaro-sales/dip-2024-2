import numpy as np
import cv2 as cv

alpha_slider_max = 100
title_window = 'H'

img1 = cv.imread("./images.jpeg")
img2 = cv.imread("./hades.jpeg")

cv.imshow("F", img1)
cv.imshow("G", img2)


def create_linear_combination_image_function(alpha=0.0, beta=0.0, img1=img1, img2=img2):
    scalarA = alpha
    scalarB = beta

    def linear_combination(new_alpha=None, new_beta=None):
        nonlocal scalarA
        nonlocal scalarB
        if new_alpha is not None:
            scalarA = new_alpha
        if new_beta is not None:
            scalarB = new_beta

        scaled_img1 = (img1[:, :, :] * scalarA).clip(0, 255).astype(np.uint8)
        scaled_img2 = (img2[:, :, :] * scalarB).clip(0, 255).astype(np.uint8)

        h = np.add(scaled_img1, scaled_img2)
        return h

    return linear_combination


new_linear_combination = create_linear_combination_image_function(img1=img1, img2=img2)


def on_trackbar(scalar_index):
    def change_scalar(value):
        new_scalar = value / alpha_slider_max
        if scalar_index == 1:
            new_image = new_linear_combination(new_alpha=new_scalar)
        elif scalar_index == 2:
            new_image = new_linear_combination(new_beta=new_scalar)
        cv.imshow(title_window, new_image)
    return change_scalar


cv.namedWindow("Trackbars", cv.WINDOW_AUTOSIZE)
cv.createTrackbar("A", "Trackbars", 0, alpha_slider_max, on_trackbar(scalar_index=1))
cv.createTrackbar("B", "Trackbars", 0, alpha_slider_max, on_trackbar(scalar_index=2))

cv.imshow(title_window, new_linear_combination())

cv.waitKey(0)
cv.destroyAllWindows()
