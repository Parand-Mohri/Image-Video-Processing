import cv2
import numpy as np

BGRImage = cv2.imread("images project1/pink.jpg")
image_float = BGRImage.astype(np.float64)
r = np.sqrt(((image_float.shape[0]/2.0)**2.0)+((image_float.shape[1]/2.0)**2.0))

polar_image = cv2.linearPolar(image_float, (image_float.shape[0] / 2, image_float.shape[1] / 2), r, cv2.WARP_FILL_OUTLIERS)
cv2.imshow('polar coordinates', polar_image)
cv2.waitKey(0)
cv2.destroyAllWindows()