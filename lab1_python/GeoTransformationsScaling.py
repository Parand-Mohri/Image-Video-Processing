import cv2
import numpy as np
import math


image = cv2.imread('pizza2.jpg')

# Uses a single scaling value
s = 1.5
scaling_matrix = np.array([[s, 0, 0],
                           [0, s, 0],
                           [0, 0, 1]])
inverse_scaling_matrix = np.array([[1/s, 0, 0],
                                   [0, 1/s, 0],
                                   [0, 0, 1]])

new_image = np.full((math.ceil(image.shape[0]*s), math.ceil(image.shape[1]*s), 3), 255)

# Uses forward mapping when scaled down, and inverse mapping when scaled up
if s <= 1:
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            values = np.array([x, y, 1])
            values = np.matmul(scaling_matrix, values)
            new_image[int(values[0])][int(values[1])] = image[x][y]
else:
    for x in range(new_image.shape[0]):
        for y in range(new_image.shape[1]):
            values = np.array([x, y, 1])
            values = np.matmul(inverse_scaling_matrix, values)
            new_image[x][y] = image[int(values[0])][int(values[1])]

cv2.imshow('original', image)
cv2.imshow('translated Image', new_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()