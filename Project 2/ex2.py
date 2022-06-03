import math

import cv2
import numpy as np
import scipy.fftpack
from numpy import r_, unravel_index, argpartition
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from numpy import pi

BGRImage = cv2.imread("images project 2/bird.jpg")
gray = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
# RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
# img = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)

def toPlot(arr, title):
    plt.figure()
    plt.imshow(arr, cmap='gray')
    plt.title(title)
    plt.show()



def filter_k(block, k):
    """ Return the K (absolute) largest values in the same location as found in the block """
    oput = np.zeros(block.shape)
    block_abs = np.abs(block)

    for i in range(k):
        r, c = np.unravel_index(np.argmax(block_abs, axis=None), block_abs.shape)
        oput[r, c] = block[r, c]
        block_abs[r, c] = 0

    return oput


def filter_k_highest(img_dct, k):
    oput = np.zeros(img_dct.shape)

    for i in r_[:img_dct.shape[0]: 8]:
        for j in r_[:img_dct.shape[1]: 8]:
            oput[i:(i + 8), j:(j + 8)] = filter_k(img_dct[i:(i + 8), j:(j + 8)], k)

    return oput

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')

def get_idct(img):
    imsize = img.shape
    img_dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            img_dct[i:(i + 8), j:(j + 8)] = idct2(img[i:(i + 8), j:(j + 8)])

    return img_dct


def getDCTBlocks(img):
    imsize = img.shape
    dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:i + 8, j:j + 8] = dct2(img[i:i + 8, j:j + 8])

    return dct


# given the dct blocks, sort array and gather last K indxs and reshape them to get values
def k_largest_coeffs(a, K):
    return np.c_[unravel_index(argpartition(a.ravel(),-K)[-K:],a.shape)]


def threshold(imgDCT, t ):
    return imgDCT * (abs(imgDCT) > (threshold * np.max(imgDCT)))



def findKDCT(dct, img, K):
    size = img.shape
    kIDCT = np.zeros(size)
    # print(size[0], size[1])
    # print(dct.shape[0], dct.shape[1])
    # get the K highest and copy them from dct
    index = k_largest_coeffs(dct, K)
    # print(index.shape[0], index.shape[1])
    filtered = np.zeros(dct.shape)
    print(index.shape)
    print(filtered.shape)
    print(dct.shape)
    filtered[index] = dct[index]


    for i in r_[:size[0]:8]:  # block-wise idct from dct coefficients
        for j in r_[:size[1]:8]:
            kIDCT[i:(i + 8), j:(j + 8)] = idct2(filtered[i:(i + 8), j:(j + 8)])

    return kIDCT, index


# apply teh watermark in a blockwise manner
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


def waterMark_dct(img,b,g,alpha):
    imsize = b.shape
    wat = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            wat[i:(i + 8), j:(j + 8)] = watermark_block(img[i:(i + 8), j:(j + 8)],b[i:(i + 8), j:(j + 8)], g, alpha)
    return wat
    # print( counter)


def estimate_watermark_block(mystry,b,alpha):
    w = np.zeros(b.shape)
    for i in range(len(w[0])):
        for j in range(len(w[1])):
            if b[i, j] != 0:
                w[i, j] =(mystry[i,j] - b[i,j]) / (alpha * b[i,j])

    return w


def estimate_watermark(img,b,alpha):
    imsize = b.shape
    wat = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            wat[i:(i + 8), j:(j + 8)] = estimate_watermark_block(img[i:(i + 8), j:(j + 8)],b[i:(i + 8), j:(j + 8)], alpha)
    return wat


def get_mean_watermark(img):
    # imsize = b.shape
    wat = []
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            wat.apestimate_watermark_block(img[i:(i + 8), j:(j + 8)],b[i:(i + 8), j:(j + 8)], alpha)
    return wat



dct = getDCTBlocks(gray)
K = 10
k_dct = filter_k_highest(dct, K)
idct_K = get_idct(k_dct)

# noise
gaussian = np.random.normal(0, math.sqrt(0.002), K)
print(gaussian)
alpha = 0.6
watermark_dct = waterMark_dct(dct,k_dct,gaussian,alpha)
watermark = get_idct(watermark_dct)

difference = watermark_dct - dct

# part 2:
mystry_with_watermark = dct2(watermark)
mystry_without_watermark = dct2(gray)

k_mystery_one = filter_k_highest(mystry_with_watermark, K)
k_mystery_two = filter_k_highest(mystry_without_watermark, K)

estimate_watermark_one = estimate_watermark(k_mystery_one, k_dct, alpha)
estimate_watermark_two =estimate_watermark(k_mystery_two, k_dct, alpha)

w_hat_bar_one = np.mean(estimate_watermark_one)
w_hat_bar_two = np.mean(estimate_watermark_two)





toPlot(idct_K, 'inv dct image with highest k coeffs')
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