import cv2
import urllib.request
import numpy as np

req = urllib.request.urlopen(
    'https://variety.com/wp-content/uploads/2021/12/doctor-strange.jpg?w=681&h=383&crop=1&resize=681%2C383')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)  # 'Load it as it is'

cv2.imshow('random_title', img)
if cv2.waitKey() & 0xff == 27:
    quit()
