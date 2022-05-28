import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# add blur to image given a and b
def addBlur(img, a, b):
    n2, n1, n0 = img.shape
    [u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
    u = 2 * u / n2
    v = 2 * v / n1
    # calculate the forier transform of each chanel
    F1 = np.fft.fftshift(np.fft.fft2(RGBImage[:,:,0]))
    F2 = np.fft.fftshift(np.fft.fft2(RGBImage[:,:,1]))
    F3 = np.fft.fftshift(np.fft.fft2(RGBImage[:,:,2]))
    # H given for adding blur
    H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
    # add the blur function to FFT of image
    G1 = F1 * H
    G2 = F2 * H
    G3 = F3 * H
    g1 = np.abs(np.fft.ifft2(G1))
    g2 = np.abs(np.fft.ifft2(G2))
    g3 = np.abs(np.fft.ifft2(G3))
    g = cv2.merge((g1,g2,g3))
    # return new image
    return g / np.max(g)



BGRImage = cv2.imread("images project 2/bird.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
grey = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY).astype(np.double)


# [u, v] = np.mgrid[-n2 / 2:n2 / 2, -n1 / 2:n1 / 2]


# F = np.fft.fft2(RGBImage)

# F = np.fft.fft2(grey)
# F = np.fft.fftshift(F)
# a = 0.15
# b = 0


plt.imshow(addBlur(RGBImage,10,10))
# plt.imshow(abs(g), cmap='gray')
plt.show()

# gaussian = np.random.normal(0, math.sqrt(0.002), RGBImage.shape)
# xn = random_noise(np.abs(g).astype(np.uint8), 'gaussian', mean=0, var=0.002).astype(np.double)
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


#Sn(u,v)/Sf(u,v) is zero because we are only doing for blur image
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
