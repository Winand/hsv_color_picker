from typing import Iterable, Tuple

import cv2
import numpy as np

from cv_utils import put_text_block, Align, def_font
from selection import Rect, RectSelection, Point, Vector


class SliderHSV:
    """
    Widget allows to select hue value and saturation/brightness range.

    `window_name` - name of a window which is used to display widget
    `size` - widget width. Default is 256px
    `slider_height` - hue slider height. Default is 16px
    `normalized_display` - display normalized values. Default is False
                           Hue 0-360, saturation/brightness 0-100%
    """
    sliding = None
    last_cursor_area = ""
    sliding_fixed_y = None
    sliding_fixed_x = None
    sliding_cursor_area = None
    font_params = {**def_font, 'fontScale': 0.75}
    font = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, window_name, size: int=256, slider_height: int=16,
                 normalized_display=False):
        self.window_name = window_name
        self.size = size  # px
        self.slider_height = slider_height  # px
        self.normalized_display = normalized_display
        self.pt = 0, 0
        self.hue = 0
        self._lower_color = [0, 0]
        self._upper_color = [0, 0]

        h_comp = np.uint8([np.linspace([0., 255., 255.], [179., 255., 255.], self.size)])
        h_comp = np.broadcast_to(h_comp, (self.slider_height, self.size, 3))
        self.h_comp = cv2.cvtColor(h_comp, cv2.COLOR_HSV2BGR)
        self.tpl_hsv = np.uint8([  # This is very very slow!
            (0, i, j) for i in np.linspace(0., 255., self.size)
                      for j in np.linspace(0., 255., self.size)
        ]).reshape(self.size, self.size, 3)

        im_stub = np.zeros(1)
        cv2.imshow(window_name, im_stub)
        self.sel = RectSelection(window_name, im_stub, (0, 0, size, size),
                                 update_callback=self.on_sel_update)
        cv2.setMouseCallback(window_name, self.on_mouse_event)
        self.set_value(0)

    def pos_to_hue(self, x):
        return int(x / (self.size - 1) * 179)

    def hue_to_pos(self, h):
        return int(h / 179 * (self.size - 1))

    def pos_to_val(self, x):
        return int(x / (self.size - 1) * 255)

    def vals_to_pos(self, h) -> tuple:
        """
        Scales 0-255 range values to widget size in pixels.
        Note: OpenCV 4.0.1 `rectangle` requires tuples as points

        Arguments:
        `h` - array of values

        Returns: Tuple[int, ...]
        """
        return tuple(np.int16(np.array(h) / 255 * (self.size - 1)))

    def on_mouse_event(self, event, x, y, flags, param):
        # https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
        if self.sel.on_mouse_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                return
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self.size:
                self.sliding = "hue"
                self.set_value(self.pos_to_hue(x))
                self.sel.set_image(self.sv)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.sliding == "hue":
                self.set_value(self.pos_to_hue(x))
                self.sel.set_image(self.sv)
        elif event == cv2.EVENT_LBUTTONUP:
            self.sliding = None
    
    def on_sel_update(self, rect: Rect, img: np.ndarray):
        lt, _, br, _ = rect.points
        text = f"Sg.{self.pos_to_val(lt.y)}\nBg.{self.pos_to_val(lt.x)}"
        color = 0 if lt.x > self.size / 3 else (255, 255, 255)
        put_text_block(img, text, lt + Vector(y=1), self.font_params, color)
        text = f"Sg.{self.pos_to_val(br.y)}\nBg.{self.pos_to_val(br.x)}"
        color = 0 if br.x - 32 > self.size / 3 else (255, 255, 255)
        put_text_block(img, text, br, self.font_params, color,
                       align=Align.bottom | Align.right)

    def set_value(self, hue):
        hue = max(min(hue, 179), 0)
        x_pos = self.hue_to_pos(hue)
        self.sv = self.create_sat_br_rect(hue)
        cv2.line(self.sv, (x_pos, self.size), (x_pos, self.size + self.slider_height), (255, 255, 255), 2)
        disp_hue = f"{hue/179*360:.1f}" if self.normalized_display else str(hue)
        cv2.putText(self.sv, f"Hue {disp_hue}", (0, self.size + self.slider_height - 2), cv2.FONT_HERSHEY_PLAIN, 1, 0, lineType=cv2.LINE_AA)
        self.hue = hue
        self.sel.set_image(self.sv)

    def create_sat_br_rect(self, hue):
        self.tpl_hsv[:, :, 0] = hue
        return cv2.vconcat([
            cv2.cvtColor(self.tpl_hsv, cv2.COLOR_HSV2BGR),
            self.h_comp
        ])

    @property
    def lower_color(self):
        return self.hue, *self._lower_color

    @property
    def upper_color(self):
        return self.hue, *self._upper_color

    def shift_hue(self, shift):
        "Returns shifted hue value, e. g. 179 (hue) + 3 (shift) gives 2"
        #                                    sign
        ret = self.hue + abs(shift) % 180 * (-(shift < 0) or 1)
        if ret < 0:
            ret = 180 + ret
        elif ret > 179:
            ret = ret - 180
        return ret

if __name__ == '__main__':
    slider = SliderHSV("HSV Color Picker", normalized_display=True)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
