import math

import cv2
import numpy as np
import scipy.fftpack
from numpy import r_
import matplotlib.pyplot as plt
from numpy import pi

BGRImage = cv2.imread("images project 2/bird.jpg",0)
# RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
# img = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)


def magnitude_spectrume(img):
    ftimage = np.fft.fft2(img)
    ftimage = np.fft.fftshift(ftimage)
    magnitude_spectrum = 20 * np.log(np.abs(ftimage))

    return magnitude_spectrum


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.T, norm='ortho' ).T, norm='ortho' )


def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')

imsize = img.shape
dct = np.zeros(imsize)
img_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:i+8,j:j+8] = dct2( img[i:i+8,j:j+8] )

mag = magnitude_spectrume(dct)
K = 10
alpha = 10

# thresh = 0.012
dct_thresh = dct * (abs(dct) > (K*np.ones(dct)))


for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        img_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )

gaussian = np.random.normal(0, math.sqrt(0.002), K)

# pos = 128

# Extract a block from image
# plt.figure()
# plt.imshow(img[pos:pos+8,pos:pos+8],cmap='gray')
# plt.title( "An 8x8 Image block")
#
# # Display the dct of that block
# plt.figure()
# plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
# plt.title( "An 8x8 DCT block")

# plt.imshow(dct, cmap='gray')
# plt.show()