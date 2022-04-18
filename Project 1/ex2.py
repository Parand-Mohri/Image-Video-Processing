import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


BGRImage = cv2.imread("images project1/fog.jpg")
Image = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
histogram(Image)

# cv2.imshow('original', Image)

height, width, _ = BGRImage.shape
negImage = BGRImage
for i in range(0, height - 1):
    for j in range(0, width - 1):
        pixel = BGRImage[i, j]
        pixel[0] = 255 - pixel[0]
        pixel[1] = 255 - pixel[1]
        pixel[2] = 255 - pixel[2]
        negImage[i, j] = pixel
# plt.imshow(BGRImage)
# plt.show()
cv2.imshow('neg', negImage)

histogram(negImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
