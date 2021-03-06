import cv2
import numpy as np
from skimage.util import random_noise

img = cv2.imread("images project 2/bird.jpg")
# img = cv2.normalize(img, None)
# img = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB).astype(np.double)
img = img / 255 #normalizing

a = 4
b = 5
n2, n1, n0 = img.shape
[u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
u = 2 * u / n2
v = 2 * v / n1
# H given for adding blur
H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
#calculate fourier transform of original image for each channel
F1 = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
F2 = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
F3 = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))
# multiply FFT of image by H
G1 = np.multiply(F1, H)
G2 = np.multiply(F2, H)
G3 = np.multiply(F3, H)
#inverse FFT to get image
g1 = np.abs(np.fft.ifft2(G1))
g2 = np.abs(np.fft.ifft2(G2))
g3 = np.abs(np.fft.ifft2(G3))
g = cv2.merge((g1, g2, g3)) #blur image

# get FFT of blur image
s1 = np.fft.fftshift(np.fft.fft2(g[:, :, 0]))
s2 =np.fft.fftshift(np.fft.fft2(g[:, :, 1]))
s3 =np.fft.fftshift(np.fft.fft2(g[:, :, 2]))
# approximate H
I1 = s1 / F1
I2 = s2 / F2
I3 = s3 / F3
#divide by H
D1 = np.divide(s1, I1)
D2 = np.divide(s2 , I2)
D3 = np.divide(s3 , I3)
d1 = np.abs(np.fft.ifft2(D1))
d2 = np.abs(np.fft.ifft2(D2))
d3 = np.abs(np.fft.ifft2(D3))
d = cv2.merge((d1, d2, d3)) #de_blur image

# add noise to blur image
l = random_noise(g, 'gaussian', mean=0, var=0.03)
l = cv2.normalize(l, None)
#FFT of blur and noisy image
q1 = np.fft.fftshift(np.fft.fft2(l[:, :, 0]))
q2 =np.fft.fftshift(np.fft.fft2(l[:, :, 1]))
q3 =np.fft.fftshift(np.fft.fft2(l[:, :, 2]))
# approximate H
I1 = q1 / F1
I2 = q2 / F2
I3 = q3 / F3
#divide by H
Q1 = np.divide(q1, I1)
Q2 = np.divide(q2, I2)
Q3 = np.divide(q3, I3)
p1 = np.abs(np.fft.ifft2(Q1))
p2 = np.abs(np.fft.ifft2(Q2))
p3 = np.abs(np.fft.ifft2(Q3))
P = cv2.merge((p1, p2, p3)) #denoised and deblur image

# add noise to original image
k = random_noise(img, 'gaussian', mean=0, var=0.03)
# k = cv2.normalize(k, None)
#get the FFT of only noisy image
o1 = np.fft.fftshift(np.fft.fft2(k[:, :, 0]))
o2 = np.fft.fftshift(np.fft.fft2(k[:, :, 1]))
o3 = np.fft.fftshift(np.fft.fft2(k[:, :, 2]))
# approximate the noise
nn = img - k
#FFT of noise
snn1 = abs(np.fft.fftshift(np.fft.fft2(nn[:, :, 0]))) ** 2
snn2 = abs(np.fft.fftshift(np.fft.fft2(nn[:, :, 1]))) ** 2
snn3 = abs(np.fft.fftshift(np.fft.fft2(nn[:, :, 2]))) ** 2
# power spevtrum of original image
sxx1 = abs(F1) ** 2
sxx2 = abs(F2) ** 2
sxx3 = abs(F3) ** 2
# put H equal 1 because no blur added
I1 = 1
I2 = 1
I3 = 1
# calcualte the denominator of transfer function MMSE
dh1 = np.abs(I1) ** 2 + (snn1 / sxx1)
dh2 = np.abs(I2) ** 2 + (snn2 / sxx2)
dh3 = np.abs(I3) ** 2 + (snn3 / sxx3)
#make the transform funciton
Hw1 = np.conj(I1) / dh1
Hw2 = np.conj(I2) / dh2
Hw3 = np.conj(I3) / dh3
# multiply by noisy image FFT
R1 = Hw1 * o1
R2 = Hw2 * o2
R3 = Hw3 * o3
a1 = np.abs(np.fft.ifft2(R1))
a2 = np.abs(np.fft.ifft2(R2))
a3 = np.abs(np.fft.ifft2(R3))
A = cv2.merge((a1, a2, a3)) # denoised by MMSE

# appriximate noise from noisy and blury image
nnB = img - l
#FFT of noise
snnB1 = abs(np.fft.fftshift(np.fft.fft2(nnB[:, :, 0]))) ** 2
snnB2 = abs(np.fft.fftshift(np.fft.fft2(nnB[:, :, 1]))) ** 2
snnB3 = abs(np.fft.fftshift(np.fft.fft2(nnB[:, :, 2]))) ** 2
#approximate H
IB1 = q1 / F1
IB2 = q2 / F2
IB3 = q3 / F3
# calcualte K from mean of s_nn / s_ff
K1 = np.mean((snnB1/sxx1))
K2 = np.mean((snnB2/sxx2))
K3 = np.mean((snnB3/sxx3))
#denominator of transfer funciton
dhB1 = np.abs(IB1) ** 2 + (K1)
dhB2 = np.abs(IB2) ** 2 + (K2)
dhB3 = np.abs(IB3) ** 2 + (K3)
#transfer function
HwB1 = np.conj(IB1) / dhB1
HwB2 = np.conj(IB2) / dhB2
HwB3 = np.conj(IB3) / dhB3
#multiply by FFT of noisy and blur image
RB1 = HwB1 * G1
RB2 = HwB2 * G2
RB3 = HwB3 * G3
aB1 = np.abs(np.fft.ifft2(RB1))
aB2 = np.abs(np.fft.ifft2(RB2))
aB3 = np.abs(np.fft.ifft2(RB3))
AB = cv2.merge((aB1, aB2, aB3)) # denoise and deblur using MMSE


cv2.imshow("blur", g)
cv2.imshow("deblue", d/np.max(d))
cv2.imshow("denoise", P/np.max(P))
cv2.imshow("noise&blur", l/np.max(l))
cv2.imshow("noise", k/np.max(k))
cv2.imshow("MMSE", A/np.max(A))
cv2.imshow("MMSE  B", AB/np.max(AB))
cv2.waitKey(0)
cv2.destroyAllWindows()

# To save the images
# cv2.imwrite('images project 2/ex1_images/blur.jpg', g * 255)
# cv2.imwrite('images project 2/ex1_images/de_blue_DIF.jpg', d/np.max(d) * 255)
# cv2.imwrite('images project 2/ex1_images/denoise_DIF.jpg', P/np.max(P) * 255)
# cv2.imwrite('images project 2/ex1_images/noise&blur.jpg', l/np.max(l) * 255)
# cv2.imwrite('images project 2/ex1_images/only noise.jpg', k/np.max(k) * 255)
# cv2.imwrite('images project 2/ex1_images/noised_MMSE.jpg', A/np.max(A) * 255)
# cv2.imwrite('images project 2/ex1_images/blur&noise_MMSE.jpg', AB/np.max(AB) * 255)