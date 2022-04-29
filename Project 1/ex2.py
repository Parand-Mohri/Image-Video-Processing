import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(image, title):
    plt.hist(image.ravel(), 256,[0,256])
    plt.title(title)
    plt.show()


# ------- original image -----------
BGRImage_S = cv2.imread("images project1/shadows.jpg")
BGRImage_F = cv2.imread("images project1/fog.jpg")
GImage_S = cv2.cvtColor(BGRImage_S, cv2.COLOR_BGR2GRAY)
GImage_F = cv2.cvtColor(BGRImage_F, cv2.COLOR_BGR2GRAY)


#  ----------- negative image --------
negImage_S = np.abs(255 - GImage_S)
negImage_F = np.abs(255 - GImage_F)

# ------- histograms of the two images -------
histogram(GImage_S, "Original Shadow image")
histogram(negImage_S, "Negative shadow image")
histogram(GImage_F, "Original Fog image")
histogram(negImage_F, "Negative Fog image")

# -------power law pointwise transform-------
gamma_S1 = np.power(GImage_S, 2)
gamma_F1 = np.power(GImage_F, 2)
gamma_S2 = np.power(GImage_S, 0.7)
gamma_F2 = np.power(GImage_F, 0.7)

#------ histogrram of transformed images -------
histogram(gamma_F1, "power law pointwise transform n = 2 fog image")
histogram(gamma_F2, "power law pointwise transform n = 0.8 fog image")
histogram(gamma_S1, "power law pointwise transform n = 2 shadow image")
histogram(gamma_S2, "power law pointwise transform n = 0.7 shadow image")


cv2.imshow('Original shadow image', BGRImage_S)
cv2.imshow('Negative shadow image', negImage_S)
cv2.imshow('power law pointwise transform n = 2 shadow image', gamma_S1)
cv2.imshow('power law pointwise transform n = 0.7 shadow image', gamma_S2)
cv2.imshow('Original fog image', BGRImage_F)
cv2.imshow('Negative fog image', negImage_F)
cv2.imshow('power law pointwise transform n = 2 fog image', gamma_F1)
cv2.imshow('power law pointwise transform n = 0.7 fog image', gamma_F2)
cv2.waitKey(0)
cv2.destroyAllWindows()
