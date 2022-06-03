
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage.measure import label

# open the image by first doing erotion and then dilation
def open(image, kernel):
    eroded = cv2.erode(image, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def close(image, kernel):
    dilated = cv2.dilate(image, kernel)
    eroded = cv2.erode(dilated, kernel)
    return eroded


def countCircles(img, x , y , j):
    # get the ellipse used to open the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, x))
    opening = open(img, kernel)
    # label the different naighboring pixels if they have same color
    label_im = label(opening)
    # count the groups find by label
    regions = regionprops(label_im)
    masks = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        # check the area of the region found and if its area is acceptable by size given count them
        if (num != 0 and (area > 10) and (convex_area / area < y)
                and (convex_area / area > j)):
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

# using trshhold given make each pixel balck or white given color
(x, y1) = cv2.threshold(grey1, 127, 255, cv2.THRESH_BINARY)
(x, y2) = cv2.threshold(grey2, 127, 255, cv2.THRESH_BINARY)

# count the circles
orangeIm1 = countCircles(y1, 10, 1.09, 0.9)
orangeIm2 = countCircles(y2, 40, 1.5, 1.1)
print(orangeIm1)
print(orangeIm2)


# part two
lights = cv2.imread("images project 2/jar.jpg")
RGBImage3 = cv2.cvtColor(lights, cv2.COLOR_BGR2RGB)
grey3 = cv2.cvtColor(RGBImage3, cv2.COLOR_RGB2GRAY)
# using trshhold given make each pixel balck or white given color
(x, y3) = cv2.threshold(grey3, 127, 255, cv2.THRESH_BINARY)
# 100 is used to count circles from 1mm to 100mm radius
size = np.linspace(1, 100, 100)
intensity = np.zeros(len(size))
for i in range(len(size)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i + 1, i + 1))
    im = open(y3, kernel)
    # the sum of all intensity pixels is the surface area, as opening is done with bigger ellipse, the intensities computed will be lower
    intensity[i] = np.sum(im)

# using np gradient to get the derivitive meaning the
# rate of change of total intensity after opening with different size ellipses
diff = np.gradient(intensity)

plt.plot(size, diff, '-bo')
plt.title('difference in surface area / radius of SE')
plt.show()

plt.plot(size, intensity)
plt.title('sum of intensities ')
plt.show()


