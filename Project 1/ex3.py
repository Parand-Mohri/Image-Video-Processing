import cv2
import numpy as np

BGRImage_Y = cv2.imread("images project1/flower.jpeg")


def polar_cordinates(img):
    r = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), r, cv2.WARP_FILL_OUTLIERS)
    return polar_image


polar_image_Y = polar_cordinates(BGRImage_Y)
# cv2.imshow('polar coordinates', polar_image_Y)


# edges = cv2.Canny(BGRImage_Y,100,200)
# retrieving the edges for cartoon effect
# by using thresholding technique

grayScaleImage = cv2.cvtColor(BGRImage_Y, cv2.COLOR_BGR2GRAY)
smoothGrayScale = cv2.medianBlur(grayScaleImage, 3)
# getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
#                                 cv2.ADAPTIVE_THRESH_MEAN_C,
#                                 cv2.THRESH_BINARY, 119, 1)
getEdge = cv2.adaptiveThreshold(grayScaleImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
# colorImage = cv2.bilateralFilter(smoothGrayScale, 9, 300, 300)
colorImage = cv2.bilateralFilter(BGRImage_Y, 18, 300, 300)

# cartoonImage = cv2.bitwise_and(colorImage, colorImage)



# thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
edges = np.abs(255 - cv2.Canny(smoothGrayScale, 1, 255, edges=getEdge))
cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=edges)
# cartoo = cartoonize(BGRImage_Y,10)
cv2.imshow("x", cartoonImage)
# cv2.imshow("x" , cartoo)
cv2.waitKey(0)
cv2.destroyAllWindows()
