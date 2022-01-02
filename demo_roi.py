import cv2

from hsv_color_picker.selection import RectSelection


def put_info(rc, img):
    cv2.putText(img, str(rc), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, 0,
                lineType=cv2.LINE_AA)

img = cv2.imread("house.jpg")
cv2.imshow('ROI', img)
sel = RectSelection('ROI', img, draw_callback=put_info)

while True:
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
