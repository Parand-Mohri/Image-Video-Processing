import cv2
import numpy as np

# k-meanas clustering using in-built methods
def k_means(image, k):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

image = cv2.imread('pizza2.jpg')

segmented_image3 = k_means(image, 3)
segmented_image10 = k_means(image, 10)

cv2.imshow('original', image)
cv2.imshow('clustered k=3', segmented_image3)
cv2.imshow('clustered k=10', segmented_image10)
cv2.waitKey(0)
cv2.destroyAllWindows()