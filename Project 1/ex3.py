import cv2
import numpy as np
from matplotlib import pyplot as plt

BGRImage = cv2.imread("images project1/rgb_image.jpeg")
# RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
Image = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)

# def adjust(image):
s = [0] * 10
height, width, _ = BGRImage.shape
for i in range(0, height - 1):
    for j in range(0, width - 1):
        pixel = Image[i, j]
        rho = np.sqrt(i**2 + j**2)
        phi = np.arctan2(j, i)
        s[rho * np.cos(phi), rho * np.sin(phi)] = pixel
    # return s


# p = adjust(BGRImage)
cv2.imshow('this', s)
cv2.waitKey(0)
cv2.destroyAllWindows()