import cv2
import numpy as np
from matplotlib import pyplot as plt


def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype


BGRImage = cv2.imread("images project1/flower.jpeg")
ima_gray = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
row, col = ima_gray.shape
X = np.linspace(0, 1, row)
Y = np.linspace(0, 1, col)
iman = im2double(ima_gray) + np.multiply(np.cos(32*np.pi * X), 0.25)

ftimage = np.fft.fft2(iman)
ftimage = np.fft.fftshift(ftimage)
psd2D = np.abs( ftimage )**2
# cv2.imwrite("images project1/pnois1.jpg", g)
# f = np.fft.fft2(iman)
# f1 = np.fft.fftshift(f)
# power_spectrum = np.abs(f1) **2
# plt.figure(2)
# plt.clf()
# plt.imshow( np.log10( power_spectrum ))
# plt.show()
# plt.hist(power_spectrum.ravel(), 256, [0, 256])
# plt.show()



cv2.imshow('noise', iman)
cv2.waitKey(0)
cv2.destroyAllWindows()
