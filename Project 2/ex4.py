import glob

import cv2
import numpy as np


def getDataMatrix(variations):
    dataMatrix = np.matrix([img.flatten() for img in variations])
    return dataMatrix


def findEigVMean(variations):
    # calculating covariance matrix and output eigenvalues and eigenvectors of the latter
    mean, eigenVectors = cv2.PCACompute(getDataMatrix(variations), mean=None)
    return mean, eigenVectors


def findEigFaces(variations):
    # get eigenfaces from eigen values
    mean, eigenVectors = findEigVMean(variations)
    print(eigenVectors.shape)
    eigenFaces = []
    # eig = eigenVectors[0].reshape(6000, 4000, 3)
    for i in range(eigenVectors.shape[0]):
        eig = eigenVectors[i].reshape(6000, 4000, 3)
        eigenFaces.append(eig)
    return eigenFaces


def getAvgFace(variations):
    mean, e = findEigVMean(variations)
    avgFace = mean.astype(np.uint8).reshape((6000, 4000, 3))
    return avgFace


def getWeight(img, eigvec, mean):
    img = np.squeeze(np.asarray(img), axis=0)
    weight = (img * eigvec) + (img * mean)
    return weight


def getWeights(variations):
    dataMatrix = getDataMatrix(variations)
    mean, eigVect = findEigVMean(variations)
    weights = []
    for i in range(dataMatrix.shape[0]):
        weights.append(getWeight(dataMatrix[i], eigVect[i], mean))
    return weights


def reconstructImage(variations):
    weights = getWeights(variations)
    # eigFaces = findEigFaces(variations)
    mean, eigVec = findEigVMean(variations)
    s_eigfaces = eigVec * weights
    s_eigfaces = s_eigfaces.sum()
    newFace = s_eigfaces + getAvgFace(variations)
    return newFace.astype(np.uint8).reshape((6000, 4000, 3))



# def reconstructImage(variations1, variations2):
#     mean, x = findEigVMean(variations1)
#     y, eigVec = findEigVMean(variations2)
#     s_eigfaces = eigVec * weights
#     s_eigfaces = s_eigfaces.sum()
#     newFace = s_eigfaces + getAvgFace(variations)
#     return newFace.astype(np.uint8).reshape((6000, 4000, 3))


variation_K = []
for img in glob.glob("images project 2/ex4_Images/im_K/*.JPG"):
    variation_K.append(cv2.imread(img))

variation_K_Some = [variation_K[0], variation_K[1]]
# mean,eigenFace_K = findEigFacesMean(variation_K)

# weights =


# making a new face
# img = avgFace + np.sum(eigenFaces)
# img = getAvgFace([rowImage_K, rowImage_O, rowImage_W])
# img = getAvgFace(variation_K)
# img1 = mean.astype(np.uint8).reshape((6000, 4000, 3))
# cv2.imshow("Image1", img)
# eigF = findEigFaces(variation_K)
# print(len(eigF))
# img1 = reconstructImageAllEig(variation_K)
img2 = reconstructImage(variation_K)
# cv2.imshow("Image1", img1)
cv2.imshow("Image2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
