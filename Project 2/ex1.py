import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

BGRImage = cv2.imread("images project 2/bird.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
grey = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY).astype(np.double)

n2, n1, n0 = RGBImage.shape
[u, v] = np.mgrid[-n2 / 2:n2 / 2, -n1 / 2:n1 / 2]
u = 2 * u / n2
v = 2 * v / n1
F = np.fft.fft2(RGBImage)
a = 0.1
b = 0.2
H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
G = F
G[:, :, 0] = np.multiply(F[:, :, 0], H)
G[:, :, 1] = np.multiply(F[:, :, 1], H)
G[:, :, 2] = np.multiply(F[:, :, 2], H)
g = np.fft.ifft2(G)
plt.imshow(abs(g) / 255)
plt.show()

# gaussian = np.random.normal(0, math.sqrt(0.002), RGBImage.shape)
xn = random_noise(np.abs(g).astype(np.uint8), 'gaussian', mean=0, var=0.002).astype(np.double)
plt.imshow(xn)
plt.show()
# GGGGG = np.fft.fft2(gaussian)
# GG = G + GGGGG
# gg = np.fft.ifft2(GG)
# plt.imshow(abs(gg)/255)
# plt.show()


# R1 = G
# R1 = F + (1/H)


# dh = np.abs(H) ** 2
# Hw = np.conj(H) / dh
# Wiener filtered motion blurred image

R1 = G
R1[:, :, 0] = np.divide(G[:,:,0], H)
R1[:, :, 1] = np.divide(G[:,:,1], H)
R1[:, :, 2] = np.divide(G[:,:,2], H)
rx1 = np.abs(np.fft.ifft2(R1))/255
plt.imshow(rx1)
plt.show()

R2 = G
R2[:, :, 0] =F[:, :, 0] + np.divide(xn[:,:,0], H)
R2[:, :, 1] =F[:, :, 1] + np.divide(xn[:,:,1], H)
R2[:, :, 2] =F[:, :, 2] + np.divide(xn[:,:,2], H)
rx2 = np.abs(np.fft.ifft2(R2))/255
plt.imshow(rx2)
plt.show()



# Fn = np.fft.fft2(xn)
# nn = RGBImage - xn
# snn = abs(np.fft.fft2(nn)) ** 2
# sxx = abs(np.fft.fft2(RGBImage)) ** 2
# dh = np.abs(H) ** 2 + snn / sxx
# Hw = np.conj(H) / dh
# # R2 = Hw * Fn
# R2 = G
# R2[:, :, 0] = np.multiply(Fn[:, :, 0], Hw)
# R2[:, :, 1] = np.multiply(Fn[:, :, 1], Hw)
# R2[:, :, 2] = np.multiply(Fn[:, :, 2], Hw)
#
# rx2 = np.abs(np.fft.ifft2(R2))
# plt.imshow(rx2)
# plt.show()
