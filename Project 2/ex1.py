import cv2
import numpy as np
import matplotlib.pyplot as plt


x = cv2.imread("images project 2/bird.jpg")
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.double)
n2, n1,n0= x.shape
[u,v] = np.mgrid[-n2/2:n2/2, -n1/2:n1/2]
u = 2*u/n2
v = 2*v/n1
F = np.fft.fft2(x)
a = 0.1
b = 0.2
H = np.sinc((u*a + v*b)) * np.exp(-1j*np.pi*(u*a + v*b))
G = F
G[: ,:, 0] = np.multiply(F[:,:,0], H)
G[: ,:, 1] = np.multiply(F[:,:,1], H)
G[: ,:, 2] = np.multiply(F[:,:,2], H)

g = np.fft.ifft2(G)
plt.imshow(abs(g)/255)
plt.show()
