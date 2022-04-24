import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

# ------- original image -----------
BGRImage = cv2.imread("images project1/shadows.jpg")

#  ----------- negative image --------
negImage = 255 - BGRImage

# ------- histograms of the two images -------
histogram(BGRImage)
histogram(negImage)

# -------power law pointwise transform-------
S = np.power(BGRImage, 2)

cv2.imshow('gamma', S)
cv2.imshow('original', BGRImage)
cv2.imshow('neg', negImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
