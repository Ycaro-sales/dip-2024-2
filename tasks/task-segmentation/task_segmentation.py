import cv2 as cv
import numpy as np
from typing import NewType

image = NewType("image", np.ndarray)


def first_order_derivative(f: image, x:int, y:int):
    first_order_mask = [[0,1,0], [1,-4,1], [0,1,0]]
    derivative_matrix = f[x-1:x+1,y-1:y+1] * first_order_mask
    dx = np.sum(derivative_matrix)
        
    return dx

def second_order_derivative(f: image, x:int, y:int):
    first_order_mask = [[1,1,1], [1,-8,1], [1,1,1]]
    derivative_matrix = f[x-1:x+1,y-1:y+1] * first_order_mask
    dx = np.sum(derivative_matrix)

    return dx

