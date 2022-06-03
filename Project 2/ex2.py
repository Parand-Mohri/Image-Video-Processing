import math

import cv2
import numpy as np
import scipy.fftpack
from numpy import r_, unravel_index, argpartition
import matplotlib.pyplot as plt
from numpy import pi

BGRImage = cv2.imread("images project 2/bird.jpg")
gray = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
# RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
# img = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)

#function to make plots
def toPlot(arr, title, dct=False):
    plt.figure()
    if dct:
        plt.imshow(arr, vmax=np.max(arr) * 0.01, vmin=0,cmap='gray')
    else:
        plt.imshow(arr, cmap='gray')
    plt.title(title)
    plt.show()


# calculate the k highest coefficient from a block
def get_k_block(block, k):
    oput = np.zeros(block.shape)
    block = np.abs(block)
    for i in range(k):
        r, c = np.unravel_index(np.argmax(block, axis=None), block.shape)
        oput[r, c] = block[r, c]
        block[r, c] = 0

    return oput


# get the k highest coefficient from all blocks
def get_k(img_dct, k):
    oput = np.zeros(img_dct.shape)

    for i in r_[:img_dct.shape[0]: 8]:
        for j in r_[:img_dct.shape[1]: 8]:
            oput[i:(i + 8), j:(j + 8)] = get_k_block(img_dct[i:(i + 8), j:(j + 8)], k)

    return oput


# calculte DCT
def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.T, norm='ortho').T, norm='ortho')

# calculate IDCT
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')

# get IDCT for all block of image
def get_idct(img):
    imsize = img.shape
    img_dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            img_dct[i:(i + 8), j:(j + 8)] = idct2(img[i:(i + 8), j:(j + 8)])

    return img_dct

# get DCT for all block of image
def getDCTBlocks(img):
    imsize = img.shape
    dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:i + 8, j:j + 8] = dct2(img[i:i + 8, j:j + 8])

    return dct


# apply teh watermark in a blockwise manner for each block
def watermark_block(img,b, g,alpha):
    w = np.zeros(b.shape)
    # x = abs(b)
    # for gi in g:
    counter = 0
    for i in range(len(w[0])):
        for j in range(len(w[1])):
            if b[i, j] != 0:
                w[i, j] = b[i, j] * (1 + alpha * g[counter])
                counter = counter + 1
            else:
                w[i, j] = img[i,j]

    return w

# add the watermark to all blocks
def waterMark_dct(img,b,g,alpha):
    imsize = b.shape
    wat = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            wat[i:(i + 8), j:(j + 8)] = watermark_block(img[i:(i + 8), j:(j + 8)],b[i:(i + 8), j:(j + 8)], g, alpha)
    return wat
    # print( counter)

# esgtimate the water mark in each block
def estimate_watermark_block(mystry,b,alpha):
    w = np.zeros(b.shape)
    for i in range(len(w[0])):
        for j in range(len(w[1])):
            if b[i, j] != 0:
                w[i, j] =(mystry[i,j] - b[i,j]) / (alpha * b[i,j])

    return w

# esgtimate the water mark in all blocks
def estimate_watermark(img,b,alpha):
    imsize = b.shape
    wat = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            wat[i:(i + 8), j:(j + 8)] = estimate_watermark_block(img[i:(i + 8), j:(j + 8)],b[i:(i + 8), j:(j + 8)], alpha)
    return wat


# def get_mean_watermark(img):
#     imsize = b.shape
#     wat = []
#     for i in r_[:imsize[0]:8]:
#         for j in r_[:imsize[1]:8]:
#             wat.apestimate_watermark_block(img[i:(i + 8), j:(j + 8)],b[i:(i + 8), j:(j + 8)], alpha)
#     return wat



dct = getDCTBlocks(gray)
K = 10
k_dct = get_k(dct, K)
idct_K = get_idct(k_dct)

# noise

#this function was used to generate numbers each time but images in the report were made by using the numbers generate below
# gaussian = np.random.normal(0, math.sqrt(0.02), K)
# print(gaussian)
gaussian = [ 0.08715447,  0.10904472, -0.06745499,  0.09772556,  0.03353753,  0.16826593,
 -0.03927987,  0.05301535,  0.19599344, -0.07863898]
alpha = 0.6
watermark_dct = waterMark_dct(dct,k_dct,gaussian,alpha)
watermark = get_idct(watermark_dct)

difference = watermark_dct - dct

# part 2:
mystry_with_watermark = dct2(watermark)
mystry_without_watermark = dct2(gray)

k_mystery_one = get_k(mystry_with_watermark, K)
k_mystery_two = get_k(mystry_without_watermark, K)

estimate_watermark_one = estimate_watermark(k_mystery_one, k_dct, alpha)
estimate_watermark_two =estimate_watermark(k_mystery_two, k_dct, alpha)

w_hat_bar_one = np.mean(estimate_watermark_one)
w_hat_bar_two = np.mean(estimate_watermark_two)


toPlot(idct_K, 'inv dct image with highest k coeffs')
toPlot(dct, 'dct image', True)
toPlot(watermark, "watermarked image")
toPlot(gray, 'original image')
toPlot(get_idct(difference), "difference image")

plt.figure()
plt.hist(gray.flatten(),bins=255, color='blue')
plt.title("histogram original image")

plt.figure()
plt.hist(watermark.flatten(),bins=255, color='blue')
plt.title("histogram watermark image")

plt.figure()
plt.hist(get_idct(difference).flatten(),bins=255, color='blue')
plt.title("histogram difference image")
plt.show()