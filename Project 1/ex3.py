import cv2
import numpy as np

BGRImage_Y = cv2.imread("images project1/flower.jpeg")

# method for polar cordinate
def polar_cordinates(img):
    r = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), r, cv2.WARP_FILL_OUTLIERS)
    return polar_image


polar_image_Y = polar_cordinates(BGRImage_Y)
cv2.imshow('polar coordinates', polar_image_Y)

# cartonize
grayScaleImage = cv2.cvtColor(BGRImage_Y, cv2.COLOR_BGR2GRAY)
smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
getEdge = cv2.adaptiveThreshold(grayScaleImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,119,32)
colorImage = cv2.bilateralFilter(BGRImage_Y, 18, 300, 300)

cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)

cv2.imshow("cartoon image", cartoonImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
