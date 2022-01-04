"""
see also selectROI
"""
from dataclasses import dataclass, astuple
from enum import Flag, auto
from typing import Optional, Tuple, NamedTuple, Union, Callable

import cv2
import numpy as np


# 3.6+ https://www.geeksforgeeks.org/typing-namedtuple-improved-namedtuples/
class Point(NamedTuple):
    "Represents a point. Values can be accessed as .x, .y or by indexing (x=0, y=1)"
    x: int = 0
    y: int = 0

    def __add__(self, other: "Point") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Vector":
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

    def __mul__(self, coef: float):
        # multiply coordinates by coefficient
        return Vector(round(self.x * coef), round(self.y * coef))

    def __eq__(self, other: "Vector") -> bool:
        return self.x == other.x and self.y == other.y


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
        return Point(x, y), Point(x + w - 1, y), \
               Point(x + w - 1, y + h - 1), Point(x, y + h - 1)

    def __rshift__(self, other: Vector) -> "Rect":
        "Translate"
        return Rect(self.x + other.x, self.y + other.y, self.w, self.h)

    def __lshift__(self, other: Vector) -> "Rect":
        "Translate"
        return Rect(self.x - other.x, self.y - other.y, self.w, self.h)

    def __add__(self, other: Vector) -> "Rect":
        "Resize"
        return Rect(self.x, self.y, self.w + other.x, self.h + other.y)

    def __sub__(self, other: Vector) -> "Rect":
        "Resize"
        return Rect(self.x, self.y, self.w - other.x, self.h - other.y)
    
    def __bool__(self) -> bool:
        return self.w != 0 and self.h != 0

    def normalize(self):
        """
        Flip coordinates if width or height is zero or negative.

        NOTE: Valid rectangle cannot have zero size, eg.
        - If `w==1` and we move right side 1px to the left, `w` becomes 0.
        - This means that we need to move left `x` 1px to the left
        - Now `w` becomes 2 'cause left and right sides are on adjacent pixels
        """
        if self.w <= 0:
            self.x += self.w - 1
            self.w = -self.w + 2
        if self.h <= 0:
            self.y += self.h - 1
            self.h = -self.h + 2


# https://docs.python.org/3/library/enum.html#flag
class RectElement(Flag):
    left = auto()    # sides
    top = auto()     #
    right = auto()   #
    bottom = auto()  #
    topleft, topright = top | left, top | right  # top corners
    bottomright, bottomleft = bottom | right, bottom | left  # bottom corners
    area = auto()  # inner area
    from_center = auto()
    # TODO: point element for point selection and moving


class RectSelection:
    clr_white = (255, 255, 255)
    cursor_tolerance = 3

    def __init__(self, window_name: str, img: np.ndarray,
                 rect: Union[Tuple[int, int, int, int], Rect]=None,
                 show_crosshair: bool=False, from_center: bool=False,
                 draw_callback: Callable[[Rect, np.ndarray], None]=None,
                 selection_callback: Callable[[Rect], None]=None):
        self.moving: Optional[RectElement] = None
        self._last_cursor_area: Optional[RectElement] = None
        self.show_crosshair = show_crosshair
        self.from_center = from_center
        self.draw_callback = lambda rc, img: None
        self.selection_callback = lambda rc: None
        self.wnd = window_name
        self.img = img
        self.rc = rect if isinstance(rect, Rect) else \
                  Rect(*(rect or (0, 0) + cv2.getWindowImageRect(self.wnd)[2:]))
        self.sel_rc = Rect()  # selected area
        self.new_rc = Rect()  # origin rect for resizing
        self.sel_pt = Point()  # point of mouse down event
        cv2.setMouseCallback(window_name, self.on_mouse_event)
        if draw_callback:
            self.set_draw_callback(draw_callback)
        if selection_callback:
            self.set_selection_callback(selection_callback)

    @staticmethod
    def transformed_rect(rect: Rect, el: Optional[RectElement], vec: Vector,
                         bounds: Rect) -> Rect:
        rc: Rect
        if el == RectElement.area:
            rc = rect >> vec
        elif el == RectElement.topleft:
            rc = (rect >> vec) - vec
        elif el == RectElement.topright:
            rc = (rect >> vec.proj_y) + vec.neg_y
        elif el == RectElement.bottomright:
            rc = rect + vec
        elif el == RectElement.bottomleft:
            rc = (rect >> vec.proj_x) + vec.neg_x
        elif el == RectElement.top:
            rc = (rect >> vec.proj_y) - vec.proj_y
        elif el == RectElement.right:
            rc = rect + vec.proj_x
        elif el == RectElement.bottom:
            rc = rect + vec.proj_y
        elif el == RectElement.left:
            rc = (rect >> vec.proj_x) - vec.proj_x
        elif el == RectElement.from_center:
            # TODO: combine with other resize flags for mirroring
            rc = (rect << vec) + vec * 2
        elif el is None:
            raise ValueError(el)  # linter
        rc.normalize()
        lt, _, br, _ = rc.points
        b_lt, _, b_br, _ = bounds.points
        # if rect inside bounds v_lt.x,y > 0, v_br.x,y < 0
        v_lt = lt - b_lt 
        v_br = br - b_br
        # Left X is out of bounds AND exceeds MORE than the opposite side
        if v_lt.x < 0 and (abs(v_lt.x) - v_br.x) > 0:  # left X bound
            rc.x = b_lt.x
            # when whole rectangle is moved its size doesn't change;
            # both left and right parts are clipped if `from_center`
            rc.w += v_lt.x * (0 if el == RectElement.area else
                              2 if el == RectElement.from_center else
                              1)
        elif v_br.x > 0:  # right X bound
            rc.w -= v_br.x * (0 if el == RectElement.area else
                              2 if el == RectElement.from_center else
                              1)
            rc.x = b_br.x - rc.w + 1
        # Top Y is out of bounds AND exceeds MORE than the opposite side
        if v_lt.y < 0 and (abs(v_lt.y) - v_br.y) > 0:  # top Y bound
            rc.y = b_lt.y
            rc.h += v_lt.y * (0 if el == RectElement.area else
                              2 if el == RectElement.from_center else
                              1)
        elif v_br.y > 0:  # bottom Y bound
            rc.h -= v_br.y * (0 if el == RectElement.area else
                              2 if el == RectElement.from_center else
                              1)
            rc.y = b_br.y - rc.h + 1
        return rc

    def on_mouse_event(self, event, x: int, y: int, flags, param) -> Optional[bool]:
        """
        Mouse callback.

        Returns: True if accepted event or None
        """
        pt = Point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            click_area = self.get_cursor_area(x, y)
            # click is allowed outside `rc` if it's a click on a part of rect,
            # see `cursor_tolerance`
            if click_area or self.pos_in_rect(x, y, self.rc):
                self.sel_pt = pt
                if click_area:
                    self.moving = click_area
                    self.new_rc = self.sel_rc
                else:
                    self.moving = RectElement.from_center if self.from_center \
                                  else RectElement.bottomright
                    self.new_rc = Rect(x, y, 1, 1)
                self.draw_rect(self.new_rc)  # update display
                return True
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON and self.moving:
                self.draw_rect(
                    self.transformed_rect(self.new_rc, self.moving,
                                          pt - self.sel_pt, bounds=self.rc)
                )
                return True
            else:
                cursor_area = self.get_cursor_area(x, y)
                if cursor_area != self._last_cursor_area:
                    self._last_cursor_area = cursor_area
                    self.draw_rect(self.sel_rc, hilight=cursor_area)
                    return True
        elif event == cv2.EVENT_LBUTTONUP and self.moving:
            self.set_selection(self.transformed_rect(
                self.new_rc, self.moving, pt - self.sel_pt, bounds=self.rc
            ))
            self.moving = None
            self._last_cursor_area = None  # forced update
            return True
        elif event == cv2.EVENT_RBUTTONDOWN:  # cancel operation
            self.moving = None
            self.draw_rect(self.sel_rc, hilight=self.get_cursor_area(x, y))
            self.set_selection(self.sel_rc)  # forced update
            return True

    def draw_rect(self, rc: Rect, hilight: RectElement=None):
        if not rc:
            cv2.imshow(self.wnd, self.img)
            return  # invalid empty rect
        img = self.img.copy()
        tl, tr, br, bl = rc.points
        cv2.rectangle(img, tl, br, self.clr_white)
        for i in (tl, tr, br, bl):
            cv2.circle(img, i, 2, self.clr_white, thickness=-1)
        
        if self.show_crosshair:
            lt, rt, _, lb = rc.points
            w2 = Vector(round(rc.w / 2))
            h2 = Vector(y=round(rc.h / 2))
            cv2.rectangle(img, lt + w2, lb + w2, self.clr_white)
            cv2.rectangle(img, lt + h2, rt + h2, self.clr_white)

        sides = {RectElement.left: (tl, bl), RectElement.top: (tl, tr),
                 RectElement.right: (tr, br), RectElement.bottom: (bl, br)}
        corners = {RectElement.topleft: tl, RectElement.topright: tr,
                   RectElement.bottomright: br, RectElement.bottomleft: bl}
        if hilight == RectElement.area:
            tmp = self.img.copy()
            cv2.rectangle(tmp, tl, br, self.clr_white, thickness=-1)
            img = cv2.addWeighted(img, 0.9, tmp, 0.1, 0)
        elif hilight is None:
            pass  # linter
        elif hilight in sides:
            cv2.line(img, *sides[hilight], self.clr_white, thickness=2)
        elif hilight in corners:
            cv2.circle(img, corners[hilight], 4, self.clr_white, thickness=-1)

        self.draw_callback(rc, img)
        cv2.imshow(self.wnd, img)

    def get_cursor_area(self, x: int, y: int) -> Optional[RectElement]:
        """
        Returns rect part under cursor w/ 2px tolerance:
        * left, top, right, bottom - sides
        * topleft, topright, bottomright, bottomleft - corners
        * area - cursor inside rect
        """
        tol = self.cursor_tolerance
        tl, _, br, _ = self.sel_rc.points
        if not self.pos_in_rect(x, y, self.sel_rc, allowance=tol):
            return

        ret = RectElement(0)
        if abs(tl.y - y) <= tol:
            ret |= RectElement.top
        elif abs(br.y - y) <= tol:
            ret |= RectElement.bottom

        if abs(tl.x - x) <= tol:
            ret |= RectElement.left
        elif abs(br.x - x) <= tol:
            ret |= RectElement.right

        if not ret and (tl.x <= x <= br.x and
                        tl.y <= y <= br.y):
            return RectElement.area
        return ret

    def pos_in_rect(self, x: int, y: int, rect: Rect, allowance: int=0):
        a = allowance
        tl, _, br, _ = rect.points
        return (tl.x - a <= x <= br.x + a and
                tl.y - a <= y <= br.y + a)

    def set_image(self, img: np.ndarray,
                  rect: Union[Tuple[int, int, int, int], Rect]=None):
        self.img = img
        if rect:
            self.rc = rect if isinstance(rect, Rect) else \
                    Rect(*(rect or (0, 0) + cv2.getWindowImageRect(self.wnd)[2:]))
        self.draw_rect(self.sel_rc)

    @property
    def selection(self):
        return self.sel_rc

    def set_draw_callback(self, callback: Callable[[Rect, np.ndarray], None]):
        "Function to call just before updated rectangle is displayed"
        self.draw_callback = callback

    def set_selection_callback(self, callback: Callable[[Rect], None]):
        "Function to call when new rect is selected"
        self.selection_callback = callback

    def set_selection(self, rc: Rect):
        self.sel_rc = rc
        self.selection_callback(rc)


def selectROI(img: np.ndarray, showCrosshair: bool=True, fromCenter: bool=False,
              windowName='ROI selector'):
    """
    Allows users to select a ROI on the given image.
    Implements OpenCV selectROI interface. NOTE: a point has 1x1 size, not 0x0

    Arguments:
    `img` - image to select a ROI
    `showCrosshair` - if true crosshair of selection rectangle will be shown
    `fromCenter` - if true center of selection will match initial mouse position.
                   In opposite case a corner of selection rectangle will
                   correspont to the initial mouse position.
    `windowName` - name of the window where selection process will be shown,
                   default is 'ROI selector'. NOTE: in built-in version of
                   `selectROI` it's the first positional argument

    Returns:
    (x, y, w, h) - selected rectangle
    """
    cv2.imshow(windowName, img)
    sel = RectSelection(windowName, img, show_crosshair=showCrosshair,
                        from_center=fromCenter)

    print(
        "Select a ROI and then press SPACE or ENTER button!\n"
        "Cancel the selection process by pressing c button!"
    )

    while True:
        k = cv2.waitKey(25) & 0xFF
        if k in (27, ord('\r'), ord(' ')):
            return astuple(sel.selection)
        elif k == ord('c'):
            return (0, 0, 0, 0)


if __name__ == '__main__':
    sel = None

    img = cv2.imread("../samples/messi5.jpg")
    cv2.imshow("test_wnd", img)
    sel = RectSelection("test_wnd", img)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
