import cv2

BGRImage = cv2.imread("images project1/birds.jpg")

RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
cv2.imshow('Original image', BGRImage)
cv2.imshow('HSV image', RGBImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
