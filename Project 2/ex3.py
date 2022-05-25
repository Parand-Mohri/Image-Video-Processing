
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.measure import label

oranges = cv2.imread("images project 2/oranges.jpg")
orangesTree = cv2.imread("images project 2/orangetree.jpg")
RGBImage1 = cv2.cvtColor(oranges, cv2.COLOR_BGR2RGB)
RGBImage2 = cv2.cvtColor(orangesTree, cv2.COLOR_BGR2RGB)
grey1 = cv2.cvtColor(RGBImage1, cv2.COLOR_RGB2GRAY)
grey2= cv2.cvtColor(RGBImage2, cv2.COLOR_RGB2GRAY)

(x, y1) = cv2.threshold(grey1, 127, 255, cv2.THRESH_BINARY)
(x, y2) = cv2.threshold(grey2, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
opening = cv2.morphologyEx(y1, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
label_im = label(closing)
regions = regionprops(label_im)
masks = []
bbox = []
list_of_index = []
for num, x in enumerate(regions):
    area = x.area
    convex_area = x.convex_area
    if (num!=0 and (area>10) and (convex_area/area <1.06)
    and (convex_area/area >0.95)):
        masks.append(regions[num].convex_image)
        bbox.append(regions[num].bbox)
        list_of_index.append(num)
count = len(masks)
print(count)
# plt.show()
# plt.imshow(kernel, cmap='gray')
cv2.imshow("orgina;", y2)
cv2.imshow("open", closing)




# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
# edged = cv2.Canny(y, 20, 140)
# contours1, hierarchy1 = cv2.findContours(y1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours2, hierarchy2 = cv2.findContours(y2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours1))
# print(len(contours2))
# plt.imshow(y1, cmap='gray')
# plt.show()
# plt.imshow(y2, cmap='gray')
# plt.show()

# lights = cv2.imread("images project 2/jar.jpg", 0)
# cv2.imshow("lights", lights)
cv2.waitKey(0)
cv2.destroyAllWindows()

