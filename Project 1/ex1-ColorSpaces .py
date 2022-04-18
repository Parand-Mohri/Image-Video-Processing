import math

import cv2
import numpy as np

# BGRImage = cv2.imread("images project1/birds.jpg")
BGRImage = cv2.imread("images project1/rgb_image.jpeg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
# Part 1 --> using build in function to get HSV
HSVImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HSV)
cv2.imshow('Original image', BGRImage)
cv2.imshow('HSV image', HSVImage)
# endregion

# Part 2 --> calculating HSI by hand
# bgr = np.float32(BGRImage)/255
rgb = np.float32(RGBImage)/255
# rgb = np.float32(RGBImage)
# B = bgr[:, :, 0]
# G = bgr[:, :, 1]
# R = bgr[:, :, 2]
R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

nominator = 1 / 2 * ((R - G) + (R - B))
# nominator = np.multiply(0.5,((R - G) + (R - B)) )
denominator = np.sqrt(((R - G) * (R - G)) + ((R - B) * (G - B)))
# denominator = np.sqrt(((R - G) * (R - G)) + ((R - B) * (G - B)))
H = np.arccos(nominator / (denominator + 0.001))
if B.all() > G.all():
    H = 360 - H

H = H / 360
S = 1 - (3 / ((R + G + B) + 0.001)) * np.minimum(np.minimum(R, G), B)
I = np.divide((R + G + B), 3)

HSI = cv2.merge((H, S, I))
cv2.imshow('HSI image', HSI)

# Part 2 -->
Cmax = np.maximum(np.maximum(R, G), B)
Cmin = np.minimum(np.maximum(R, G), B)
delta = Cmax - Cmin
if delta.all() == 0:
    H1 = np.zeros_like(Cmax)
elif np.equal(Cmax, R):
    H1 = np.degrees(60) * (((G - B)/delta) + 0)
elif np.equal(Cmax, G):
    H1 = np.degrees(60) * (((B - R)/delta) + 2)
elif np.equal(Cmax, B):
    H1 = np.degrees(60) * (((R - G) / delta) + 4)

if Cmax.all() == 0:
    S1 = np.zeros_like(Cmax)
else:
    S1 = np.divide(delta ,Cmax)

V = Cmax
HSV = cv2.merge((H1, S1, V))
cv2.imshow('HSVVVV image', HSV)




cv2.waitKey(0)
cv2.destroyAllWindows()
