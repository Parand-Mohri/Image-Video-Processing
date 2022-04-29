import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def butterworthLPF(d0, n1, n2, n):
    k1, k2 = np.meshgrid(np.arange(-round(n2 / 2) + 1, math.floor(n2 / 2) + 1),
                         np.arange(-round(n1 / 2) + 1, math.floor(n1 / 2) + 1))
    d = np.sqrt(k1 ** 2 + k2 ** 2)
    h = 1 / (1 + (d / d0) ** (2 * n))
    return h


BGRImage = cv2.imread("images project1/flower.jpeg")

ima_gray = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
row, col = ima_gray.shape
x = np.linspace(30, 40, row)
y = np.linspace(30, 40, col)
[X, Y] = np.meshgrid(y, x)
iman = ima_gray.astype(float) / np.max(ima_gray) + np.multiply(np.cos(28 * np.pi * X), 0.1)

d0 = 50
butterworthLPF = butterworthLPF(d0, iman.shape[0], iman.shape[1],1)
f = np.fft.fftshift(np.fft.fft2(iman))
f1 = f * butterworthLPF
x1 = np.fft.ifft2(np.fft.ifftshift(f1))
plt.imshow(abs(x1) / 255, cmap="gray")
plt.title("Transformed")
plt.show()
# print(iman.shape)
# print(x1.shape)

# ftimage = np.fft.fft2(ima_gray)
# ftimage = np.fft.fftshift(ftimage)
# psd2D = np.abs(ftimage)**2
# m = ftimage.shape[0]
# n = ftimage.shape[1]
# for u in range(m):
#     for v in range(n):
#         for d in range(len(points)):
#                 u0 = points[d][0]
#                 v0 = points[d][1]
#                 u0, v0 = v0, u0
#                 d1 = pow(pow(u - u0, 2) + pow(v - v0, 2), 0.5)
#                 d2 = pow(pow(u + u0, 2) + pow(v + v0, 2), 0.5)
#                 ftimage[u][v] *= (1.0 / (1 + pow((d0 * d0) / (d1 * d2), order)))

#
# f_ishift = np.fft.ifftshift(ftimage)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)


# ftimage = np.fft.fft2(ima_gray)
# ftimage = np.fft.fftshift(ftimage)
# psd2D = np.square(20 * np.log(np.abs(ftimage)))
# plt.imshow(psd2D,cmap='gray')
# plt.show()
# ftimage1d = np.fft.fft(ima_gray)
# ftimage1d = np.fft.fftshift(ftimage1d)
# psd1D = np.abs(ftimage1d)**2
# ftimage3d = np.fft.fftn(BGRImage)
# ftimage3d = np.fft.fftshift(ftimage3d)
# psd3D = np.abs(ftimage3d)**2


cv2.imshow('noise', iman)
cv2.waitKey(0)
cv2.destroyAllWindows()
