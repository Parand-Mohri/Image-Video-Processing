import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from skimage.util import random_noise


def getDataMatrix(variations):
    dataMatrix = np.matrix([img.flatten() for img in variations])
    return dataMatrix


def findEigVV(variations):
    # calculating covariance matrix and output eigenvalues and eigenvectors of the latter
    eigenValues, eigenVectors = cv2.PCACompute(getDataMatrix(variations), mean=None)
    return eigenValues, eigenVectors


def findEigFaces(variations):
    # get eigenfaces from eigen values
    eigenValues, eigenVectors = findEigVV(variations)
    eigenFaces = [eig.reshape(6000, 4000, 3) for eig in eigenVectors]
    return eigenFaces


def getAvgFace(variations):
    dataMatrix = getDataMatrix(variations)
    avgRowImage = np.array(np.round_(dataMatrix.sum(axis=0) / dataMatrix.shape[0]).astype(np.uint8))
    avgFace = avgRowImage.reshape((6000, 4000, 3))
    return avgFace


variation_K = []
for img in glob.glob("images project 2/ex4_Images/im_K/*.JPG"):
    variation_K.append(cv2.imread(img))

eigenFace_K = findEigFaces(variation_K)


# weights =


# making a new face
# img = avgFace + np.sum(eigenFaces)
# img = getAvgFace([rowImage_K, rowImage_O, rowImage_W])
img = getAvgFace(variation_K)
# cv2.imshow("Image1", avgFace)
cv2.imshow("Image2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
