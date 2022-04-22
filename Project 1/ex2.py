import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


BGRImage = cv2.imread("images project1/shadows.jpg")
# ------- original image -----------
# Imagegray = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)

#  ----------- negative image --------
# negImage = 255 - BGRImage

# ------- histograms of the two images -------
# histogram(Image)
# histogram(negImage)

# -------power law pointwise transform-------



# height, width, _ = BGRImage.shape
# for i in range(0, height - 1):
#     for j in range(0, width - 1):
#         pixel = BGRImage[i, j]
#         # pixel[0] = 255 - pixel[0]
#         # pixel[1] = 255 - pixel[1]
#         # pixel[2] = 255 - pixel[2]
#         # negImage[i, j] = pixel




# def adjust_gamma(image):
#     # print(image.shape)
#     # bgrimage = cv2.cvtColor(bgrimage, cv2.COLOR_BGR2GRAY)
#     s = image
#     height, width, _ = image.shape
#     for i in range(0, height - 1):
#         for j in range(0, width - 1):
#             pixel = image[i, j]
#             s[i, j] = np.power(pixel, 2)
#     return s


#
#
# x = adjust_gamma(BGRImage, 0.7)
# x = adjust_gamma(BGRImage)
# print(BGRImage.shape)
# Image = np.divide(Imagegray, 255)
S = np.power(BGRImage, 2)
# cv2.imshow('gamma', x)
# cv2.imshow('gamma', np.multiply(S, 255))
cv2.imshow('gamma', S)

# cv2.imshow('original', Imagegray)
# cv2.imshow('neg', negImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
