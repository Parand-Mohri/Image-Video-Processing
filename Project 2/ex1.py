import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# add blur to image given a and b and remove the blur using inverse filtering
def addAndRemoveBlur(img, a, b):
    n2, n1, n0 = img.shape
    [u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
    u = 2 * u / n2
    v = 2 * v / n1
    # calculate the forier transform of each chanel
    F1 = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
    F2 = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
    F3 = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))
    # H given for adding blur
    H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
    # add the blur function to FFT of image
    G1 = F1 * H
    G2 = F2 * H
    G3 = F3 * H
    # remove the blur funciton
    D1 = G1 / H
    D2 = G2 / H
    D3 = G3 / H
    g1 = np.abs(np.fft.ifft2(G1))
    g2 = np.abs(np.fft.ifft2(G2))
    g3 = np.abs(np.fft.ifft2(G3))
    d1 = np.abs(np.fft.ifft2(D1))
    d2 = np.abs(np.fft.ifft2(D2))
    d3 = np.abs(np.fft.ifft2(D3))
    g = cv2.merge((g1, g2, g3))
    d = cv2.merge((d1,d2,d3))
    # return blur and de_blur image
    return g, d


def addNoise(img, mean, var):
    # add guassian nois to image with given mean and var
    xn = random_noise(img.astype(np.uint8), 'gaussian', mean=mean, var=var).astype(np.double)
    return xn


BGRImage = cv2.imread("images project 2/bird.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
# numbers are chosen here to blur and noisy the image but enough for image to be visible
blur, de_blur = addAndRemoveBlur(BGRImage, 9, 3)  # only blur and de_blured
noise = addNoise(BGRImage, 0, 1)  # only noisy
noise_blur = addNoise(blur, 0, 0.09)  # blur and noisy
cv2.imshow("blur", blur/np.max(blur))
cv2.imshow("de_blur", de_blur/np.max(de_blur))
cv2.imshow("original", BGRImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(blur)
# plt.show()
# plt.imshow(noise_blur)
# plt.show()
# plt.imshow(denoise_only_blur)
# plt.show()

# [u, v] = np.mgrid[-n2 / 2:n2 / 2, -n1 / 2:n1 / 2]


# F = np.fft.fft2(RGBImage)

# F = np.fft.fft2(grey)
# F = np.fft.fftshift(F)
# a = 0.15
# b = 0


# plt.imshow(abs(g), cmap='gray')

# gaussian = np.random.normal(0, math.sqrt(0.002), RGBImage.shape)

# plt.imshow(xn)
# plt.show()
# GGGGG = np.fft.fft2(gaussian)
# GG = G + GGGGG
# gg = np.fft.ifft2(GG)
# plt.imshow(abs(gg)/255)
# plt.show()


# R1 = G
# dh = np.abs(H) ** 2
# Hw = np.conj(H) / dh
# Wiener filtered motion blurred image

# R1 = G
# R1[:, :, 0] = np.divide(G[:,:,0], H)
# R1[:, :, 1] = np.divide(G[:,:,1], H)
# R1[:, :, 2] = np.divide(G[:,:,2], H)
# rx1 = np.abs(np.fft.ifft2(R1))/255
# plt.imshow(rx1)
# plt.show()
#
# R2 = G
# R2[:, :, 0] =F[:, :, 0] + np.divide(xn[:,:,0], H)
# R2[:, :, 1] =F[:, :, 1] + np.divide(xn[:,:,1], H)
# R2[:, :, 2] =F[:, :, 2] + np.divide(xn[:,:,2], H)
# rx2 = np.abs(np.fft.ifft2(R2))/255
# plt.imshow(rx2)
# plt.show()


# Sn(u,v)/Sf(u,v) is zero because we are only doing for blur image
# dh = np.abs(H) ** 2
# Hw = np.conj(H) / dh
# R3 = G
# R3[:, :, 0] = np.multiply(Hw, G[:,:,0])
# R3[:, :, 1] = np.multiply(Hw, G[:,:,1])
# R3[:, :, 2] = np.multiply(Hw, G[:,:,2])
# rx3 = np.abs(np.fft.ifft2(R3)) / 255
# plt.imshow(rx3)
# plt.show()

# Fn = np.fft.fft2(xn)
# nn = RGBImage - xn
# snn = abs(np.fft.fft2(nn)) ** 2
# sxx = abs(np.fft.fft2(RGBImage)) ** 2
# K = np.mean(snn / sxx)
# print(np.mean(snn / sxx))
# print(np.average(snn / sxx))
# dh1 = np.abs(H) ** 2
# dh1[:,:,0] = (np.abs(H) ** 2) + (snn[:,:,0] / sxx[:,:,0])
# dh1[:,:,1] = (np.abs(H) ** 2 )+ snn[:,:,1] / sxx[:,:,1]
# dh1[:,:,2] = (np.abs(H) ** 2 )+ (snn[:,:,2] / sxx[:,:,2])
# Hw1 = np.conj(H) / dh1
# Hw1[:,:,0] = np.conj(H) / dh1[:,:,0]
# Hw1[:,:,1] = np.conj(H) / dh1[:,:,1]
# Hw1[:,:,2] = np.conj(H) / dh1[:,:,2]
# R4 = Fn
# R4[:, :, 0] = np.multiply(Hw1, Fn[:,:,0])
# R4[:, :, 1] = np.multiply(Hw1, Fn[:,:,1])
# R4[:, :, 2] = np.multiply(Hw1, Fn[:,:,2])
# rx4 = np.abs(np.fft.ifft2(R4)) / 255
# plt.imshow(rx4)
# plt.show()


# R2 = Hw *
# R2 = G
# R2[:, :, 0] = np.multiply(Fn[:, :, 0], Hw)
# R2[:, :, 1] = np.multiply(Fn[:, :, 1], Hw)
# R2[:, :, 2] = np.multiply(Fn[:, :, 2], Hw)
#
# rx2 = np.abs(np.fft.ifft2(R2))
# plt.imshow(rx2)
# plt.show()
