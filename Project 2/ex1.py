import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


def getH(img, a, b):
    n2, n1, n0 = img.shape
    [u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
    u = 2 * u / n2
    v = 2 * v / n1
    # H given for adding blur
    H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
    return H

# add blur to image given a and b and remove the blur using inverse filtering
def addAndRemoveBlur(img,H):
    # calculate the forier transform of each chanel
    F1 = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
    F2 = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
    F3 = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))
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
    d = cv2.merge((d1, d2, d3))
    # return blur and de_blur image
    return g, d


def addNoise(img, mean, var, H):
    # add guassian nois to image with given mean and var
    xn = random_noise(abs(img).astype(np.uint8), 'gaussian', mean=mean, var=var).astype(np.double)
    # Fn = np.fft.fft2(xn)
    F1 = np.fft.fftshift(np.fft.fft2(xn[:, :, 0]))
    F2 = np.fft.fftshift(np.fft.fft2(xn[:, :, 1]))
    F3 = np.fft.fftshift(np.fft.fft2(xn[:, :, 2]))
    D1 = F1 / H
    D2 = F2 / H
    D3 = F3 / H
    d1 = np.abs(np.fft.ifft2(D1))
    d2 = np.abs(np.fft.ifft2(D2))
    d3 = np.abs(np.fft.ifft2(D3))
    d = cv2.merge((d1,d2,d3))
    return xn, d

def MMSE(original_img, noisy_img):
    # F= np.fft.fftshift(np.fft.fft2(original_img))
    G1 = np.fft.fft2(noisy_img[:, :, 0])
    G2 = np.fft.fft2(noisy_img[:, :, 1])
    G3 = np.fft.fft2(noisy_img[:, :, 2])
    F1 = np.fft.fft2(original_img[:, :, 0])
    F2 = np.fft.fft2(original_img[:, :, 1])
    F3 = np.fft.fft2(original_img[:, :, 2])
    nn = original_img - noisy_img
    HH1 = np.divide(abs(np.fft.fft2(nn[:, :, 0])), abs(F1))
    HH2 = np.divide(abs(np.fft.fft2(nn[:, :, 1])), abs(F2))
    HH3 = np.divide(abs(np.fft.fft2(nn[:, :, 2])), abs(F3))
    # HH1 =np.divide(abs(np.fft.fft2(nn[:, :, 0])),  abs(np.fft.fft2(original_img[:, :, 0])))
    # HH2 = np.divide(abs(np.fft.fft2(nn[:, :, 1])),  abs(np.fft.fft2(original_img[:, :, 1])))
    # HH3 = np.divide(abs(np.fft.fft2(nn[:, :, 2])),  abs(np.fft.fft2(original_img[:, :, 2])))
    snn1 = abs(np.fft.fft2(nn[:, :, 0])) ** 2
    snn2 = abs(np.fft.fft2(nn[:, :, 1])) ** 2
    snn3 = abs(np.fft.fft2(nn[:, :, 2])) ** 2
    sxx1 = abs(np.fft.fft2(original_img[:, :, 0])) ** 2
    sxx2 = abs(np.fft.fft2(original_img[:, :, 1])) ** 2
    sxx3 = abs(np.fft.fft2(original_img[:, :, 2])) ** 2
    dh1 = np.abs(HH1) ** 2 + snn1 / sxx1
    dh2 = np.abs(HH2) ** 2 + snn2 / sxx2
    dh3 = np.abs(HH3) ** 2 + snn3 / sxx3
    # Hw1 = np.abs(HH1) ** 2 / dh1
    # Hw2 = np.abs(HH2) ** 2 / dh2
    # Hw3 = np.abs(HH3) ** 2 / dh3
    Hw1 = np.conj(HH1) / dh1
    Hw2 = np.conj(HH2) / dh2
    Hw3 = np.conj(HH3) / dh3
    R1 = Hw1 * G1
    R2 = Hw2 * G2
    R3 = Hw3 * G3
    d1 = np.abs(np.fft.ifft2(R1))
    d2 = np.abs(np.fft.ifft2(R2))
    d3 = np.abs(np.fft.ifft2(R3))
    d = cv2.merge((d1,d2,d3))
    return d


BGRImage = cv2.imread("images project 2/bird.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
# numbers are chosen here to blur and noisy the image but enough for image to be visible
H = getH(BGRImage, 0.9,0.8)
blur, de_blur = addAndRemoveBlur(BGRImage, H)  # only blur and de_blured
# de_blur = inverse_filtering(blur, H)
noise, x = addNoise(BGRImage, 0, 0.009, H)  # only noisy
# noise_blur, denoised = addNoise(blur, 0, 0.009, H)  # blur and noisy
denoised = MMSE(BGRImage, noise)
# cv2.imshow("blur", blur/np.max(blur))
# cv2.imshow("de_blur", de_blur/np.max(de_blur))
# cv2.imshow("denoised", denoised/np.max(denoised))
cv2.imshow("noised", noise)
cv2.imshow("denoised", denoised)
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
