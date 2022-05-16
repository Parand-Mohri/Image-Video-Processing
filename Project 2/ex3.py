
import cv2
import matplotlib.pyplot as plt

BGRImage = cv2.imread("images project 2/oranges.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
grey = cv2.cvtColor(RGBImage, cv2.COLOR_RGB2GRAY)
(x, y) = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(y, 20, 140)
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
plt.imshow(y, cmap='gray')
plt.show()