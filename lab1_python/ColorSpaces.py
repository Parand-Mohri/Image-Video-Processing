import cv2

# OpenCV reads, and shows images in as BGR format
BGRImage = cv2.imread("pizza2.jpg")

RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

# Separate the three channels from BGR image
RImage = BGRImage[:,:,2]
GImage = BGRImage[:,:,1]
BImage = BGRImage[:,:,0]

# Shows images, end by pressing any key on the keyboard
cv2.imshow('BGRImage', BGRImage)
cv2.imshow('RGBImage', RGBImage)
cv2.imshow('Rimage', RImage)
cv2.imshow('Gimage', GImage)
cv2.imshow('Bimage', BImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


