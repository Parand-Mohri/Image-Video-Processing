import cv2
import numpy as np

# convert BGR to RGB
BGRImage_B = cv2.imread("images project1/birds.jpg")
RGBImage_B = cv2.cvtColor(BGRImage_B, cv2.COLOR_BGR2RGB)

BGRImage_S = cv2.imread("images project1/stone.jpg")
RGBImage_S = cv2.cvtColor(BGRImage_S, cv2.COLOR_BGR2RGB)

# convert to HSV
HSVImage_B = cv2.cvtColor(BGRImage_B, cv2.COLOR_BGR2HSV)
HSVImage_S = cv2.cvtColor(BGRImage_S, cv2.COLOR_BGR2HSV)


# fund I and V
def find_I_V(RGBImage):
    rgb = np.float32(RGBImage) / 255 #normalize
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    I = np.divide((R + G + B), 3)
    V = np.maximum(np.maximum(R, G), B)
    return I, V


I_B, V_B = find_I_V(RGBImage_B)
I_S, V_S = find_I_V(RGBImage_S)

cv2.imshow('Original Bird image', BGRImage_B)
cv2.imshow('Original Stone image', BGRImage_S)
cv2.imshow('HSV Bird image', HSVImage_B)
cv2.imshow('HSV Stone image', HSVImage_S)
cv2.imshow('I Bird image', I_B)
cv2.imshow('I Stone image', I_S)
cv2.imshow('V Bird image', V_B)
cv2.imshow('V Stone image', V_S)
cv2.waitKey(0)
cv2.destroyAllWindows()
