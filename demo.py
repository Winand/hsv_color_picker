import cv2 as cv
import numpy as np

from hsv_color_picker import SliderHSV


color_slider = SliderHSV("HSV slider", normalized_display=True)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
hue_width = 10
while True:
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Threshold the HSV image to get only colors in selected range
    lower_color = list(color_slider.lower_color)
    lower_color[0] = color_slider.shift_hue(-hue_width)
    upper_color = list(color_slider.upper_color)
    upper_color[0] = color_slider.shift_hue(+hue_width)
    if lower_color[0] > upper_color[0]:
        # https://stackoverflow.com/q/30331944
        mask1 = cv.inRange(hsv, np.uint8(lower_color), np.uint8([179, *upper_color[1:]]))
        mask2 = cv.inRange(hsv, np.uint8([0, *lower_color[1:]]), np.uint8(upper_color))
        mask = mask1 + mask2
    else:
        mask = cv.inRange(hsv, np.uint8([0,0,0]), np.uint8(upper_color))
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
