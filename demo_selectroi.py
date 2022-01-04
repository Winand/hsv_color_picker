import cv2

from hsv_color_picker.selection import selectROI

img = cv2.imread("house.jpg")
rc = selectROI(img)
print(rc)
