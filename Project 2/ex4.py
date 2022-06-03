import glob

import cv2
import numpy as np


def getDataMatrix(variations):
    # make all images a row and add it to a matrix and return
    data_matrix = np.matrix([img.flatten() for img in variations])
    return data_matrix


def findEigVMean(data_matrix):
    # calculating covariance matrix and output eigenvalues and eigenvectors of the latter
    mean, eigen_vec = cv2.PCACompute(data_matrix, mean=None)
    return mean, eigen_vec


def findEigFaces(eigen_vec):
    # get eigen faces from eigen vectors
    eigen_faces = []
    for i in range(eigen_vec.shape[0]):
        eig = eigen_vec[i].reshape(6000, 4000, 3)
        eigen_faces.append(eig)
    return eigen_faces


def getAvgFace(mean):
    # calculate average face from mean
    avg_face = mean.astype(np.uint8).reshape((6000, 4000, 3))
    return avg_face


# website formula
def getWeight(img, eig_vec, mean):
    # calculate the weight for each image
    img = np.squeeze(np.asarray(img), axis=0)
    m = img - mean
    weight = np.dot(m, eig_vec)
    return weight


def getWeights(data_matrix, eig_vec, mean):
    # calculate all the weights
    weights = []
    for i in range(data_matrix.shape[0]):
        weights.append(getWeight(data_matrix[i], eig_vec[i], mean))
    return weights


def reconstructImage(variations):
    # make new image given variations
    data_matrix = getDataMatrix(variations)
    mean, eig_vec = findEigVMean(data_matrix)
    # avgface = getAvgFace(mean)
    # eigenface = findEigFaces(eig_vec)
    weights = getWeights(data_matrix, eig_vec, mean)
    s_eig_faces = eig_vec * weights
    s_eig_faces = s_eig_faces.sum()
    new_face = s_eig_faces + mean
    return new_face.astype(np.uint8).reshape((6000, 4000, 3))


def reconstructImageDif(variations1, variations2):
    #make new image given two variation using the mean of first and eigenvector of second
    data_matrix_one = getDataMatrix(variations1)
    data_matrix_two = getDataMatrix(variations2)
    mean, x = findEigVMean(data_matrix_one)
    y, eig_vec = findEigVMean(data_matrix_two)
    weights = getWeights(data_matrix_one, eig_vec, mean)
    s_eigfaces = eig_vec * weights
    s_eigfaces = s_eigfaces.sum()
    newFace = s_eigfaces + mean
    return newFace.astype(np.uint8).reshape((6000, 4000, 3))


variation_K = []
for img in glob.glob("images project 2/ex4_Images/im_K/*.JPG"):
    variation_K.append(cv2.imread(img))

variation_O = []
for img in glob.glob("images project 2/ex4_Images/im_O/*.JPG"):
    variation_O.append(cv2.imread(img))

variation_W = []
for img in glob.glob("images project 2/ex4_Images/im_W/*.JPG"):
    variation_W.append(cv2.imread(img))


variation_K_Some = [variation_K[0], variation_K[1]]
variation_O_Some = [variation_O[0], variation_O[1]]
variation_W_Some = [variation_W[0], variation_W[1]]
img_k= reconstructImage(variation_K)
img_w = reconstructImage(variation_W)
img_o = reconstructImage(variation_O)

img_k_some = reconstructImage(variation_K_Some)
img_w_some= reconstructImage(variation_W_Some)
img_o_some = reconstructImage(variation_O_Some)
K_O = reconstructImageDif(variation_K, variation_O)

# cv2.imshow("Image1", img1)
# cv2.imshow("Image2", img2)
# cv2.imshow("avgface_kid", avgface_kid)
# cv2.imshow("avgface_woman", avgface_woman)
# cv2.imshow("avgface_old", avgface_old)
cv2.imshow("img_k", img_k)
cv2.imshow("img_w", img_w)
cv2.imshow("img_o", img_o)
cv2.imshow("img_k_some", img_k_some)
cv2.imshow("img_w_some", img_w_some)
cv2.imshow("img_o_some", img_o_some)
cv2.imshow("kid and old woman", K_O)
cv2.imshow("eigf_k", abs(eigf_k[0]) / np.max(abs(eigf_k[0])))
cv2.imshow("eigf_o", abs(eigf_O[0]) / np.max(abs(eigf_O[0])))
cv2.imshow("eigf_w", abs(eigf_W[0]) / np.max(abs(eigf_W[0])))

cv2.waitKey(0)
cv2.destroyAllWindows()
