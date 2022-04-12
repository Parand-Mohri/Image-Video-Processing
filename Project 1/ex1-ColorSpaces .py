import math

import cv2
import numpy as np

BGRImage = cv2.imread("images project1/birds.jpg")
# Part 1 --> using build in function to get HSV
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
cv2.imshow('Original image', BGRImage)
# cv2.imshow('HSV image', RGBImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Part 2 --> calculating HSI by hand
bgr = np.float32(BGRImage)/255
R = bgr[:, :, 2]
G = bgr[:, :, 1]
B = bgr[:, :, 0]
nominator = 1 / 2 * ((R - G) + (R - B))
denominator = np.sqrt(((R - G) * (R - G)) + ((R - B) * (G - B)))
H = np.arccos(nominator / (denominator + 0.000001))
if B.all() > G.all():
    H = 360 - H

H = H / 360
S = 1 - (3 / ((R + G + B) + 0.001)) * np.minimum(R, G, B)
I = np.divide((R + G + B), 3)

HSI = cv2.merge((H, S, I))
cv2.imshow('HSI image', HSI)
cv2.waitKey(0)
cv2.destroyAllWindows()
