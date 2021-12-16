import cv2 as cv
import numpy as np

from hsv_color_picker import SliderHSV


color_slider = SliderHSV("HSV slider", size=128)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
hue_width = 10
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    lower_color = [color_slider.hue] + list(color_slider.lower_color)
    lower_color[0] = lower_color[0] - hue_width
    if lower_color[0] < 0:
        lower_color[0] = 180 + lower_color[0]
    upper_color = [color_slider.hue] + list(color_slider.upper_color)
    upper_color[0] = upper_color[0] + hue_width
    if upper_color[0] > 179:
        upper_color[0] = 180 - upper_color[0]
    # print(lower_color, upper_color)
    mask = cv.inRange(hsv, np.uint8(lower_color), np.uint8(upper_color))
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
