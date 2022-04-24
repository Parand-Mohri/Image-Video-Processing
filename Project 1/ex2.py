import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(image, title):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(title)
    plt.show()


# ------- original image -----------
BGRImage_S = cv2.imread("images project1/shadows.jpg")
BGRImage_F = cv2.imread("images project1/fog.jpg")

#  ----------- negative image --------
negImage_S = 255 - BGRImage_S
negImage_F = 255 - BGRImage_F

# ------- histograms of the two images -------
histogram(BGRImage_S, "Original Shadow image")
histogram(negImage_S, "Negative shadow image")
histogram(BGRImage_F, "Original Fog image")
histogram(negImage_F, "Negative Fog image")

# -------power law pointwise transform-------
gamma_S1 = np.power(BGRImage_S, 2)
gamma_F1 = np.power(BGRImage_F, 2)
gamma_S2 = np.power(BGRImage_S, 0.7)
gamma_F2 = np.power(BGRImage_F, 0.7)


cv2.imshow('Original shadow image', BGRImage_S)
cv2.imshow('Negative shadow image', negImage_S)
cv2.imshow('power law pointwise transform shadow image', gamma_S1)
cv2.imshow('Original fog image', BGRImage_F)
cv2.imshow('Negative fog image', negImage_F)
cv2.imshow('power law pointwise transform fog image', gamma_F1)
cv2.waitKey(0)
cv2.destroyAllWindows()
