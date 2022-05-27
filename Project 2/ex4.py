import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from skimage.util import random_noise


def resizeImage(img,scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim)


# BGRImage_O = cv2.imread("images project 2/old1.jpg")
# BGRImage_O = resizeImage(BGRImage_O, 115.75)
BGRImage_O = cv2.imread("images project 2/Picture1.jpg")
cv2.imwrite("images project 2/Picture1.jpg", BGRImage_O)
BGRImage_K = cv2.imread("images project 2/kid1.jpg")
BGRImage_W = cv2.imread("images project 2/woman2.jpg")
rowImage_O = BGRImage_O.flatten()
rowImage_W = BGRImage_W.flatten()
rowImage_K = BGRImage_K.flatten()
dataMatrix = np.matrix([rowImage_K,rowImage_O,rowImage_W])
avgRowImage =np.array(np.round_(dataMatrix.sum(axis=0)/3).astype(np.uint8))

# calculating covariance matrix and output eigenvalues and eigenvectors of the latter
eigenValues, eigenVectors = cv2.PCACompute(dataMatrix, mean=None)
img1 = avgRowImage + eigenVectors.sum(axis=0).astype(np.uint8)
print(rowImage_O.size)
print(avgRowImage.size)
img = img1.reshape((6000, 4000, 3))
# print(rowImage_O.shape)
# print(BGRImage_W.shape)
# img = img1.reshape((6000, 4000, 3))

# print(img1.shape)

# cv2.imshow("old", BGRImage_O)
# cv2.imshow("woman", BGRImage_W)
# cv2.imshow("Kid", BGRImage_K)
cv2.imshow("Image1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()