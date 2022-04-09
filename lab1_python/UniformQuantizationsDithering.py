import cv2, math
import numpy as np
from skimage.util import random_noise

# np.digitize does not give expected output for q=2 thus a simple manual method is used for this case
def uniform_quantization(GrayImage, q):
    if q == 2:
        QImage = np.zeros(GrayImage.shape)
        for x in range(GrayImage.shape[0]):
            for y in range(GrayImage.shape[1]):
                QImage[x][y] = math.floor(float(GrayImage[x][y]) / (256.0 / q)) * (256 / q)
        return QImage
    else:
        bins = np.linspace(GrayImage.min(), GrayImage.max(), q)
        QImage = np.digitize(GrayImage, bins)
        QImage = (np.vectorize(bins.tolist().__getitem__)(QImage-1).astype(int))
        return QImage.astype(np.uint8)

# A quick MSE method
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

Image = cv2.imread('pizza2.jpg')

GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

# Gaussian noise added to image (Other distributions can be used)
GrayImage = random_noise(GrayImage, mode='gaussian', seed=10001)
GrayImage = (255*GrayImage).astype(np.uint8)

Q2Image = uniform_quantization(GrayImage, 2)
Q16Image = uniform_quantization(GrayImage, 16)
Q128Image = uniform_quantization(GrayImage, 128)

print("MSE K=2: ", mse(GrayImage, Q2Image))
print("MSE K=16: ", mse(GrayImage, Q16Image))
print("MSE K=128: ", mse(GrayImage, Q128Image))

cv2.imshow('Original Image', Image)
cv2.imshow('GrayScaled Image', GrayImage)
cv2.imshow('Quantized lvl 2', Q2Image)
cv2.imshow('Quantized lvl 16', Q16Image)
cv2.imshow('Quantized lvl 128', Q128Image)
cv2.waitKey(0)
cv2.destroyAllWindows()



