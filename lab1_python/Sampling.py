import cv2

m = 100  # m >= 1

BGRImage = cv2.imread("pizza2.jpg")

#Creates sampled image, sampled every m pixels
SampledImage = BGRImage[::m, ::m, :]

# Resized for easier viewing
[n1, n2, n3] = BGRImage.shape
ResizedSampleImage = cv2.resize(SampledImage, [n2,n1], interpolation=cv2.INTER_NEAREST)

cv2.imshow('BGRImage', BGRImage)
cv2.imshow('SampledImage', SampledImage)
cv2.imshow('ResizedImage', ResizedSampleImage)
cv2.waitKey(0)
cv2.destroyAllWindows()