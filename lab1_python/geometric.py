
import cv2
import numpy as np


# Resizing
img = cv2.imread('pizza2.jpg')
res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
cv2.imshow("Resizing 0.5", res)
# OR
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Resizing 2", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

