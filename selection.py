"""
see also selectROI
"""
from dataclasses import dataclass, astuple
from typing import Tuple, NamedTuple

import cv2
import numpy as np


# 3.6+ https://www.geeksforgeeks.org/typing-namedtuple-improved-namedtuples/
class Point(NamedTuple):
    "Represents a point. Values can be accessed as .x, .y or by indexing (x=0, y=1)"
    x: int = 0
    y: int = 0


# 3.7+ https://jerrynsh.com/all-you-need-to-know-about-data-classes-in-python/
@dataclass
class Rect:
    """
    Represents a rectangle.
    `points` (prop.) - get coordinates of each corner as a tuple of Points
    """
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @property
    def points(self) -> Tuple[Point]:
        """
        Return coordinates of top-left, top-right, bottom-right and bottom-left corners as Points
        """
        x, y, w, h = astuple(self)
        return Point(x, y), Point(x + w, y), Point(x + w, y + h), Point(x, y + h)


class RectSelection:
    clr_white = (255, 255, 255)
    def __init__(self, window_name, img: np.array, rect: Tuple[int, int, int, int]=None):
        self.selecting = False
        self.wnd = window_name
        self.img = img
        self.rc = Rect(*(rect or (0, 0) + cv2.getWindowImageRect(window_name)[2:]))
        self.sel_rc = Rect()
        self.sel_pt: Point = None  # point of mouse down event
        cv2.setMouseCallback(window_name, self.on_mouse_event)

    def on_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("lmb", self.get_cursor_area(x, y))
            self.selecting = True
            self.sel_pt = Point(x, y)
            self.sel_rc = Rect(x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                print("mm", self.rc)
                self.sel_rc.w = x - self.sel_rc.x
                self.sel_rc.h = y - self.sel_rc.y
                self.draw_rect(self.sel_rc)
            else:
                self.draw_rect(self.sel_rc, hilight=self.get_cursor_area(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False

    def draw_rect(self, rc, hilight: str=None):
        img = self.img.copy()
        tl, tr, br, bl = rc.points
        cv2.rectangle(img, tl, br, self.clr_white)
        for i in (tl, tr, br, bl):
            cv2.circle(img, i, 2, self.clr_white, thickness=-1)

        sides = {'left': (tl, bl), 'top': (tl, tr), 'right': (tr, br), 'bottom': (bl, br)}
        corners = {'topleft': tl, 'topright': tr, 'bottomright': br, 'bottomleft': bl}
        if hilight == 'rect':
            tmp = self.img.copy()
            cv2.rectangle(tmp, tl, br, self.clr_white, thickness=-1)
            img = cv2.addWeighted(img, 0.9, tmp, 0.1, 0)
        elif hilight in sides:
            cv2.line(img, *sides[hilight], self.clr_white, thickness=2)
        elif hilight in corners:
            cv2.circle(img, corners[hilight], 4, self.clr_white, thickness=-1)

        cv2.imshow(self.wnd, img)

    def get_cursor_area(self, x, y) -> str:
        """
        Returns rect part under cursor w/ 2px tolerance:
        * left, top, right, bottom - sides
        * topleft, topright, bottomright, bottomleft - corners
        * area - cursor inside rect
        """
        ret = ""
        tl, _, br, _ = self.sel_rc.points
        if not self.pos_in_rect(x, y, allowance=2):
            return ret

        if abs(tl.y - y) <= 2:
            ret += 'top'
        elif abs(br.y - y) <= 2:
            ret += 'bottom'

        if abs(tl.x - x) <= 2:
            ret += 'left'
        elif abs(br.x - x) <= 2:
            ret += 'right'
        
        if not ret and (tl.x <= x <= br.x and
                        tl.y <= y <= br.y):
            ret = 'rect'
        return ret
    
    def pos_in_rect(self, x, y, allowance=0):
        a = allowance
        tl, _, br, _ = self.sel_rc.points
        return (tl.x - a <= x <= br.x + a and
                tl.y - a <= y <= br.y + a)


if __name__ == '__main__':
    sel = None

    def on_mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global sel
            sel = RectSelection("test_wnd")

    # cv2.namedWindow("test_wnd")
    img = cv2.imread("samples/messi5.jpg")
    cv2.imshow("test_wnd", img)
    sel = RectSelection("test_wnd", img)
    # cv2.setMouseCallback("test_wnd", on_mouse_event)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

# if __name__ == '__main__' :

#     # Read image
#     im = cv2.imread("samples/messi5.jpg")

#     # Select ROI
#     r = cv2.selectROI(im, False, False)
#     print(r)
#     # Crop image
#     imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

#     # Display cropped image
#     cv2.imshow("Image", imCrop)
#     cv2.waitKey(0)

#     cv2.destroyAllWindows()
