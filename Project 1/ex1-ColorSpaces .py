import math

import cv2
import numpy as np

BGRImage = cv2.imread("images project1/birds.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
# Part 1 --> using build in function to get HSV
HSVImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HSV)

# Part 2 --> calculating HSI by hand
rgb = np.float32(RGBImage)/255
R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

nominator = np.multiply(0.5, ((R - G) + (R - B)))
denominator = np.sqrt(((R - G) * (R - G)) + ((R - B) * (G - B)))
H = np.arccos(nominator / (denominator + 0.001))
if B.all() > G.all():
    H = 360 - H

H = H / 360
S = 1 - (3 / ((R + G + B) + 0.001)) * np.minimum(np.minimum(R, G), B)
I = np.divide((R + G + B), 3)

HSI = cv2.merge((H, S, I))


# Part 2 --> find V from HSV
V = np.maximum(np.maximum(R, G), B)

cv2.imshow('Original image', BGRImage)
cv2.imshow('HSV image', HSVImage)
cv2.imshow('HSI image', HSI)
cv2.imshow('V', V)
cv2.waitKey(0)
cv2.destroyAllWindows()
