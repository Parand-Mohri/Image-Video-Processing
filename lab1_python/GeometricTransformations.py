import cv2
import numpy as np
import math

image = cv2.imread('alfie.jpg')

# rotation and translation variables and matrices
t1 = 0
t2 = 40
a = math.pi/4
translation_matrix = np.array([[1, 0, t1],
                               [0, 1, t2],
                               [0, 0, 1]])
rotation_matrix = np.array([[math.cos(a), -math.sin(a), 0],
                            [math.sin(a), math.cos(a), 0],
                            [0, 0, 1]])
translate_rotate_matrix = np.array([[math.cos(a), -math.sin(a), t1],
                                    [math.sin(a), math.cos(a), t2],
                                    [0, 0, 1]])

# USed to display the image correctly, and to ensure the new image includes all of the old image
m, n, _ = image.shape
max_values=[]
max_values.append(np.matmul(rotation_matrix, (m - int((m+1)/2), n - int((n+1)/2), 1)))
max_values.append(np.matmul(rotation_matrix, (0- int((m+1)/2), 0 - int((n+1)/2), 1)))
max_values.append(np.matmul(rotation_matrix, (m- int((m+1)/2), 0 - int((n+1)/2), 1)))
max_values.append(np.matmul(rotation_matrix, (0- int((m+1)/2), n - int((n+1)/2), 1)))
max_val = np.max(max_values, 0)
min_val = np.min(max_values, 0)
new_image = np.full((math.ceil(max_val[0]-min_val[0]),math.ceil(max_val[1]-min_val[1]) , 3), 255).astype(np.uint8)
min_x = min_val[0] + int((m + 1) / 2)
min_y = min_val[1] + int((n + 1) / 2)

# Performs rotation then translation (Uses nearest neighbor)
# Here forward mapping is used
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        values = np.array([x - int((m+1)/2), y - int((n+1)/2), 1])
        values = np.matmul(rotation_matrix, values)
        values = np.matmul(translation_matrix, values)
        if new_image.shape[0] > values[0] + int((m + 1) / 2) - min_x > 0 and new_image.shape[1] > values[1] + int((n + 1) / 2) - min_y > 0:
            new_image[int(values[0] + int((m + 1) / 2) - min_x)][int(values[1] + int((n + 1) / 2) - min_y)] = image[x][y]

#Performs rotation and translation simultaneously (Uses nearest neighbor)
# Here forward mapping is used
new_image2 = np.full((math.ceil(max_val[0]-min_val[0]), math.ceil(max_val[1]-min_val[1]), 3), 255).astype(np.uint8)
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        values = np.array([x - int((m + 1) / 2), y - int((n + 1) / 2), 1])
        values = np.matmul(translate_rotate_matrix, values)
        if new_image2.shape[0] > values[0] + int((m + 1) / 2) - min_x > 0 and new_image2.shape[1] > values[1] + int((n + 1) / 2) - min_y > 0:
            new_image2[int(values[0] + int((m + 1) / 2) - min_x)][int(values[1] + int((n + 1) / 2) - min_y)] = image[x][y]

cv2.imshow('original', image)
cv2.imshow('rotated translated Image with 2 matrix multiplications: rotation, then translation', new_image)
cv2.imshow('rotated translated Image with rotation-translation matrix multiplications', new_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()