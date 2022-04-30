import cv2
import numpy as np
from matplotlib import pyplot as plt

BGRImage = cv2.imread("images project1/pink.jpg")


def magnitude_spectrume(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ftimage = np.fft.fft2(img)
    ftimage = np.fft.fftshift(ftimage)
    magnitude_spectrum = 20 * np.log(np.abs(ftimage))

    return magnitude_spectrum


padding = np.pad(BGRImage, ((0, 0), (300, 0), (0, 0)),
                 mode='constant')

magnitude_spectrum_original = magnitude_spectrume(BGRImage)
magnitude_spectrum_padding = magnitude_spectrume(padding)

plt.subplot(121)
plt.imshow(magnitude_spectrum_original, cmap='gray')
plt.title('Magnitude Spectrum original')
plt.subplot(122)
plt.imshow(magnitude_spectrum_padding, cmap='gray')
plt.title('Magnitude Spectrum padding')
plt.show()

cv2.imshow("padding", padding)
cv2.imshow("original", BGRImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
