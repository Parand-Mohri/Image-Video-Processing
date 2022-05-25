
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage.measure import label


def open(image, kernel):
    eroded = cv2.erode(image, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def close(image, kernel):
    dilated = cv2.dilate(image, kernel)
    eroded = cv2.erode(dilated, kernel)
    return eroded


def countCircles(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = open(img, kernel)
    closing = close(opening, kernel)
    label_im = label(closing)
    regions = regionprops(label_im)
    masks = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if (num != 0 and (area > 10) and (convex_area / area < 1.06)
                and (convex_area / area > 0.95)):
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
    count = len(masks)
    return count


oranges = cv2.imread("images project 2/oranges.jpg")
orangesTree = cv2.imread("images project 2/orangetree.jpg")
RGBImage1 = cv2.cvtColor(oranges, cv2.COLOR_BGR2RGB)
RGBImage2 = cv2.cvtColor(orangesTree, cv2.COLOR_BGR2RGB)
grey1 = cv2.cvtColor(RGBImage1, cv2.COLOR_RGB2GRAY)
grey2 = cv2.cvtColor(RGBImage2, cv2.COLOR_RGB2GRAY)

(x, y1) = cv2.threshold(grey1, 127, 255, cv2.THRESH_BINARY)
(x, y2) = cv2.threshold(grey2, 127, 255, cv2.THRESH_BINARY)

orangeIm1 = countCircles(y1)
orangeIm2 = countCircles(y2)
print(orangeIm1)
print(orangeIm2)

lights = cv2.imread("images project 2/jar.jpg")
RGBImage3 = cv2.cvtColor(lights, cv2.COLOR_BGR2RGB)
grey3 = cv2.cvtColor(RGBImage3, cv2.COLOR_RGB2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opening = open(grey3, kernel)
closing = close(opening, kernel)
counting = np.zeros(109)
for i in range(1,110):
    circles = cv2.HoughCircles(closing,cv2.HOUGH_GRADIENT,1.5,1,
                            param1=50,param2=30,minRadius=i,maxRadius=i)
    if circles is not None:
        counting[i-1] = circles[0].size / 3


x = np.array(range(109))
plt.xlabel("Radius of circle")
plt.ylabel("number of circles")
plt.plot(x, counting, color = "blue", marker = "o", label = "Array elements")
plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

