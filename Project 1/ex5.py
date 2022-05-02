import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


# method for magnitude spectrum
def magnitude_spectrume(img_gray):
    ftimage = np.fft.fft2(img_gray)
    ftimage = np.fft.fftshift(ftimage)
    magnitude_spectrum = 20 * np.log(np.abs(ftimage))

    return magnitude_spectrum


# method for butter worth filtering
def butterworthLPF(d0, n1, n2, n):
    k1, k2 = np.meshgrid(np.arange(-round(n2 / 2) + 1, math.floor(n2 / 2) + 1),
                         np.arange(-round(n1 / 2) + 1, math.floor(n1 / 2) + 1))
    d = np.sqrt(k1 ** 2 + k2 ** 2)
    h = 1 / (1 + (d / d0) ** (2 * n))
    return h


BGRImage = cv2.imread("images project1/flower.jpeg")

ima_gray = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
magnitude_spectrume_gray = magnitude_spectrume(ima_gray)

# power spevtrum grey image
psd2D_gray = magnitude_spectrume_gray ** 2
wG, hG = ima_gray.shape
psd1D_grey = psd2D_gray[int(wG / 2), :]

# making the noisy image using sine
row, col = ima_gray.shape
x = np.linspace(30, 40, row)
y = np.linspace(30, 40, col)
[X, Y] = np.meshgrid(y, x)
iman = ima_gray.astype(float) / np.max(ima_gray) + np.multiply(np.cos(28 * np.pi * X), 0.1)


# denoising the image
d0 = 50
butterworthLPF = butterworthLPF(d0, iman.shape[0], iman.shape[1], 1)
f = np.fft.fftshift(np.fft.fft2(iman))
f1 = f * butterworthLPF
x1 = np.fft.ifft2(np.fft.ifftshift(f1))
dneoised = abs(x1) / 255
plt.imshow(dneoised, cmap="gray")
plt.title("deionised")
plt.show()

# power spevtrum denoised image
magnitude_spectrume_denoised = magnitude_spectrume(dneoised)
psd2D_denoised = magnitude_spectrume_denoised ** 2

# power spevtrum noisy image
magnitude_spectrume = magnitude_spectrume(iman)
psd2D = magnitude_spectrume ** 2
w, h = iman.shape
psd1D = psd2D[int(w / 2), :]
n3D, m3D = np.meshgrid(np.arange(w), np.arange(h))
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
ax.plot_surface(n3D, m3D, iman.T, cmap=plt.cm.coolwarm, linewidth=0)
plt.title("3D power spectrum noisy image")
plt.show()
n3D_grey, m3D_grey = np.meshgrid(np.arange(wG), np.arange(hG))

fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
ax.plot_surface(n3D_grey, m3D_grey, ima_gray.T, cmap=plt.cm.coolwarm, linewidth=0)
plt.title("3D power spectrum grey image")
plt.show()
plt.imshow(psd2D, cmap="gray")
plt.title('2D power spectrum noisy image')
plt.show()
plt.plot(psd1D)
plt.title('1D power spectrum noisy image')
plt.show()
plt.imshow(psd2D_denoised, cmap="gray")
plt.title('2D power spectrum deionised image')
plt.show()
plt.imshow(psd2D_gray, cmap="gray")
plt.title('2D power spectrum grey image')
plt.show()
plt.plot(psd1D_grey)
plt.title('1D power spectrum grey image')
plt.show()


cv2.imshow('noise', iman)
cv2.waitKey(0)
cv2.destroyAllWindows()
