"""
see also selectROI
"""
from dataclasses import dataclass, astuple
from enum import Flag, auto
from typing import Optional, Tuple, NamedTuple

import cv2
import numpy as np


# 3.6+ https://www.geeksforgeeks.org/typing-namedtuple-improved-namedtuples/
class Point(NamedTuple):
    "Represents a point. Values can be accessed as .x, .y or by indexing (x=0, y=1)"
    x: int = 0
    y: int = 0

    def __add__(self, other: "Point"):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point"):
        return Vector(self.x - other.x, self.y - other.y)


class Vector(Point):
    "Vector is used in translation operations"

    @property
    def proj_y(self):
        "Projection on Y axis"
        return Vector(y=self.y)

    @property
    def proj_x(self):
        "Projection on X axis"
        return Vector(x=self.x)

    @property
    def neg_x(self):
        "Set X to -X"
        return Vector(-self.x, self.y)

    @property
    def neg_y(self):
        "Set Y to -Y"
        return Vector(self.x, -self.y)


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
    def points(self) -> Tuple[Point, Point, Point, Point]:
        """
        Return coordinates of top-left, top-right, bottom-right and bottom-left corners as Points
        """
        x, y, w, h = astuple(self)
        return Point(x, y), Point(x + w, y), Point(x + w, y + h), Point(x, y + h)

    def __rshift__(self, other: Vector) -> "Rect":
        "Translate"
        return Rect(self.x + other.x, self.y + other.y, self.w, self.h)

    def __add__(self, other: Vector) -> "Rect":
        "Resize"
        return Rect(self.x, self.y, self.w + other.x, self.h + other.y)

    def __sub__(self, other: Vector) -> "Rect":
        "Resize"
        return Rect(self.x, self.y, self.w - other.x, self.h - other.y)

    def normalize(self):
        "Flip coordinates if width or height is negative"
        if self.w < 0:
            self.x += self.w
            self.w = -self.w
        if self.h < 0:
            self.y += self.h
            self.h = -self.h


# https://docs.python.org/3/library/enum.html#flag
class RectElement(Flag):
    left = auto()    # sides
    top = auto()     #
    right = auto()   #
    bottom = auto()  #
    topleft, topright = top | left, top | right  # top corners
    bottomright, bottomleft = bottom | right, bottom | left  # bottom corners
    area = auto()  # inner area


class RectSelection:
    clr_white = (255, 255, 255)
    cursor_tolerance = 3

    def __init__(self, window_name: str, img: np.ndarray, rect: Tuple[int, int, int, int]=None):
        self.moving: Optional[RectElement] = None
        self.wnd = window_name
        self.img = img
        self.rc = Rect(*(rect or (0, 0) + cv2.getWindowImageRect(window_name)[2:]))
        self.sel_rc = Rect()
        self.sel_pt = Point()  # point of mouse down event
        cv2.setMouseCallback(window_name, self.on_mouse_event)

    @staticmethod
    def transformed_rect(rect, el: Optional[RectElement], vec: Vector) -> Rect:
        if el == RectElement.area:
            return rect >> vec
        if el == RectElement.topleft:
            return (rect >> vec) - vec
        if el == RectElement.topright:
            return (rect >> vec.proj_y) + vec.neg_y
        if el == RectElement.bottomright:
            return rect + vec
        if el == RectElement.bottomleft:
            return (rect >> vec.proj_x) + vec.neg_x
        if el == RectElement.top:
            return (rect >> vec.proj_y) - vec.proj_y
        if el == RectElement.right:
            return rect + vec.proj_x
        if el == RectElement.bottom:
            return rect + vec.proj_y
        if el == RectElement.left:
            return (rect >> vec.proj_x) - vec.proj_x
        raise NotImplementedError(el)

    def on_mouse_event(self, event, x: int, y: int, flags, param):
        pt = Point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sel_pt = pt
            self.moving = self.get_cursor_area(x, y)
            if not self.moving:
                self.moving = RectElement.bottomright
                self.sel_rc = Rect(x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                rc = self.transformed_rect(self.sel_rc, self.moving, pt - self.sel_pt)
                self.draw_rect(rc)
            else:
                self.draw_rect(self.sel_rc, hilight=self.get_cursor_area(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.sel_rc = self.transformed_rect(self.sel_rc, self.moving, pt - self.sel_pt)
            self.sel_rc.normalize()
        elif event == cv2.EVENT_RBUTTONDOWN:  # cancel operation
            self.moving = None
            self.draw_rect(self.sel_rc)

    def draw_rect(self, rc: Rect, hilight: RectElement=None):
        img = self.img.copy()
        tl, tr, br, bl = rc.points
        cv2.rectangle(img, tl, br, self.clr_white)
        for i in (tl, tr, br, bl):
            cv2.circle(img, i, 2, self.clr_white, thickness=-1)

        sides = {RectElement.left: (tl, bl), RectElement.top: (tl, tr),
                 RectElement.right: (tr, br), RectElement.bottom: (bl, br)}
        corners = {RectElement.topleft: tl, RectElement.topright: tr,
                   RectElement.bottomright: br, RectElement.bottomleft: bl}
        if hilight == RectElement.area:
            tmp = self.img.copy()
            cv2.rectangle(tmp, tl, br, self.clr_white, thickness=-1)
            img = cv2.addWeighted(img, 0.9, tmp, 0.1, 0)
        elif hilight in sides:
            cv2.line(img, *sides[hilight], self.clr_white, thickness=2)
        elif hilight in corners:
            cv2.circle(img, corners[hilight], 4, self.clr_white, thickness=-1)

        cv2.imshow(self.wnd, img)

    def get_cursor_area(self, x: int, y: int) -> Optional[RectElement]:
        """
        Returns rect part under cursor w/ 2px tolerance:
        * left, top, right, bottom - sides
        * topleft, topright, bottomright, bottomleft - corners
        * area - cursor inside rect
        """
        tl, _, br, _ = self.sel_rc.points
        if not self.pos_in_rect(x, y, allowance=self.cursor_tolerance):
            return

        ret = RectElement(0)
        if abs(tl.y - y) <= 2:
            ret |= RectElement.top
        elif abs(br.y - y) <= 2:
            ret |= RectElement.bottom

        if abs(tl.x - x) <= 2:
            ret |= RectElement.left
        elif abs(br.x - x) <= 2:
            ret |= RectElement.right

        if not ret and (tl.x <= x <= br.x and
                        tl.y <= y <= br.y):
            return RectElement.area
        return ret

    def pos_in_rect(self, x: int, y: int, allowance: int=0):
        a = allowance
        tl, _, br, _ = self.sel_rc.points
        return (tl.x - a <= x <= br.x + a and
                tl.y - a <= y <= br.y + a)


if __name__ == '__main__':
    sel = None

    # def on_mouse_event(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         global sel
    #         sel = RectSelection("test_wnd")

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
