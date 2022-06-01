import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

img = cv2.imread("images project 2/bird.jpg")
# img = cv2.normalize(img, None)
# img = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB).astype(np.double)
img = img / 255

a = 3
b = 4
n2, n1, n0 = img.shape
[u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
u = 2 * u / n2
v = 2 * v / n1
# H given for adding blur
H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
F1 = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
F2 = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
F3 = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))
G1 = np.multiply(F1, H)
G2 = np.multiply(F2, H)
G3 = np.multiply(F3, H)
g1 = np.abs(np.fft.ifft2(G1))
g2 = np.abs(np.fft.ifft2(G2))
g3 = np.abs(np.fft.ifft2(G3))
g = cv2.merge((g1, g2, g3)) #blue image

s1 = np.fft.fftshift(np.fft.fft2(g[:, :, 0]))
s2 =np.fft.fftshift(np.fft.fft2(g[:, :, 1]))
s3 =np.fft.fftshift(np.fft.fft2(g[:, :, 2]))
I1 = s1 / F1
I2 = s2 / F2
I3 = s3 / F3
D1 = np.divide(s1, I1)
D2 = np.divide(s2 , I2)
D3 = np.divide(s3 , I3)
d1 = np.abs(np.fft.ifft2(D1))
d2 = np.abs(np.fft.ifft2(D2))
d3 = np.abs(np.fft.ifft2(D3))
d = cv2.merge((d1, d2, d3))


l = random_noise(g, 'gaussian', mean=0, var=0.09)
l = cv2.normalize(l, None)
q1 = np.fft.fftshift(np.fft.fft2(l[:, :, 0]))
q2 =np.fft.fftshift(np.fft.fft2(l[:, :, 1]))
q3 =np.fft.fftshift(np.fft.fft2(l[:, :, 2]))
I1 = q1 / F1
I2 = q2 / F2
I3 = q3 / F3
Q1 = np.divide(q1, I1)
Q2 = np.divide(q2, I2)
Q3 = np.divide(q3, I3)
p1 = np.abs(np.fft.ifft2(Q1))
p2 = np.abs(np.fft.ifft2(Q2))
p3 = np.abs(np.fft.ifft2(Q3))
P = cv2.merge((p1, p2, p3))


# s = img
# s = img[:,:,0] + n
# s = img[:,:,1] + n
# s = img[:,:,2] + n
cv2.imshow("blue", g)
cv2.imshow("deblue", d/np.max(d))
cv2.imshow("denoise", P/np.max(P))
cv2.imshow("noise", l/np.max(l))
cv2.waitKey(0)
cv2.destroyAllWindows()

