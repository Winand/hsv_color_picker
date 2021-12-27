import cv2
import numpy as np


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
        self.set_value(0)
        cv2.setMouseCallback(window_name, self.on_mouse_event)

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
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self.size:
                self.sliding = "hue"
                self.set_value(self.pos_to_hue(x))
            else:
                self.sliding = "sat-br"
                self.sliding_fixed_x = None
                self.sliding_fixed_y = None
                self.sliding_cursor_area = cursor_area = self.get_cursor_area(x, y)
                if cursor_area == 'left':
                    self.sliding_fixed_y = self.vals_to_pos((self._lower_color[0],))[0]
                elif cursor_area == 'top':
                    self.sliding_fixed_x = self.vals_to_pos((self._lower_color[1],))[0]
                elif cursor_area == 'right':
                    self.sliding_fixed_y = self.vals_to_pos((self._upper_color[0],))[0]
                elif cursor_area == 'bottom':
                    self.sliding_fixed_x = self.vals_to_pos((self._upper_color[1],))[0]

                if cursor_area in ('left', 'lefttop', 'top'):
                    self.pt = self.vals_to_pos(self._upper_color[::-1])
                elif cursor_area == 'righttop':
                    self.pt = self.vals_to_pos((self._lower_color[1], self._upper_color[0]))
                elif cursor_area in ('right', 'rightbottom', 'bottom'):
                    self.pt = self.vals_to_pos(self._lower_color[::-1])
                elif cursor_area == 'leftbottom':
                    self.pt = self.vals_to_pos((self._upper_color[1], self._lower_color[0]))
                else:
                    self.pt = self.pos_to_val(x), self.pos_to_val(y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.sliding == "hue":
                self.set_value(self.pos_to_hue(x))
            elif self.sliding == "sat-br":
                if self.sliding_fixed_x is not None:
                    x = self.sliding_fixed_x
                if self.sliding_fixed_y is not None:
                    y = self.sliding_fixed_y
                if self.sliding_cursor_area == 'rect':
                    sx, sy = x - self.pt[0], y - self.pt[1]
                    pt1 = self.vals_to_pos(self._lower_color[::-1])
                    pt1 = (pt1[0] + sx, pt1[1] + sy)
                    pt2 = self.vals_to_pos(self._upper_color[::-1])
                    pt2 = (pt2[0] + sx, pt2[1] + sy)
                    print(pt1, pt2, sx, sy)
                    # FIXME: сдвиг считается от начальной точки, а сдвигается предыдущее положение прямоугольника
                    self.set_rect(pt1, pt2, 'rect')
                else:
                    self.set_rect(self.pt, (self.pos_to_val(x), self.pos_to_val(y)), self.sliding_cursor_area)
            else:
                cursor_area = self.get_cursor_area(x, y)
                if self.last_cursor_area != cursor_area:
                    self.last_cursor_area = cursor_area
                    self.draw_rect(hilight=cursor_area)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.sliding == "sat-br":
                if self.vals_to_pos(self.pt) == (x, y):
                    # Single click resets range (hide rect)
                    self.set_rect((0, 0), (0, 0))
            self.sliding = None

    def get_cursor_area(self, x, y) -> str:
        ret = ""
        pos1 = self.vals_to_pos(self._lower_color[::-1])
        pos2 = self.vals_to_pos(self._upper_color[::-1])
        if not self.pos_in_rect(x, y, allowance=2):
            return ret
        if abs(pos1[0] - x) <= 2:
            ret = 'left'
        elif abs(pos2[0] - x) <= 2:
            ret = 'right'

        if abs(pos1[1] - y) <= 2:
            ret += 'top'
        elif abs(pos2[1] - y) <= 2:
            ret += 'bottom'
        
        if not ret and (pos1[0] <= x <= pos2[0] and
                        pos1[1] <= y <= pos2[1]):
            ret = 'rect'
        return ret
    
    def pos_in_rect(self, x, y, allowance=0):
        a = allowance
        pos1 = self.vals_to_pos(self._lower_color[::-1])
        pos2 = self.vals_to_pos(self._upper_color[::-1])
        return (pos1[0] - a <= x <= pos2[0] + a and
                pos1[1] - a <= y <= pos2[1] + a)

    def draw_rect(self, hilight=None):
        pt1 = self._lower_color[::-1]
        pt2 = self._upper_color[::-1]
        pos_pt1 = self.vals_to_pos(pt1)
        pos_pt2 = self.vals_to_pos(pt2)
        sv = self.sv.copy()
        cv2.rectangle(sv, pos_pt1, pos_pt2, (255, 255, 255))
        cv2.circle(sv, pos_pt1, 2, (255, 255, 255), thickness=-1)
        cv2.circle(sv, (pos_pt2[0], pos_pt1[1]), 2, (255, 255, 255), thickness=-1)
        cv2.circle(sv, pos_pt2, 2, (255, 255, 255), thickness=-1)
        cv2.circle(sv, (pos_pt1[0], pos_pt2[1]), 2, (255, 255, 255), thickness=-1)

        if hilight == 'rect':
            tmp = self.sv.copy()
            cv2.rectangle(tmp, pos_pt1, pos_pt2, (255, 255, 255), thickness=-1)
            sv = cv2.addWeighted(sv, 0.9, tmp, 0.1, 0)
        elif hilight == 'left':
            cv2.line(sv, pos_pt1, (pos_pt1[0], pos_pt2[1]), (255, 255, 255), thickness=2)
        elif hilight == 'top':
            cv2.line(sv, pos_pt1, (pos_pt2[0], pos_pt1[1]), (255, 255, 255), thickness=2)
        elif hilight == 'right':
            cv2.line(sv, (pos_pt2[0], pos_pt1[1]), pos_pt2, (255, 255, 255), thickness=2)
        elif hilight == 'bottom':
            cv2.line(sv, (pos_pt1[0], pos_pt2[1]), pos_pt2, (255, 255, 255), thickness=2)
        elif hilight == 'lefttop':
            cv2.circle(sv, pos_pt1, 4, (255, 255, 255), thickness=-1)
        elif hilight == 'righttop':
            cv2.circle(sv, (pos_pt2[0], pos_pt1[1]), 4, (255, 255, 255), thickness=-1)
        elif hilight == 'rightbottom':
            cv2.circle(sv, pos_pt2, 4, (255, 255, 255), thickness=-1)
        elif hilight == 'leftbottom':
            cv2.circle(sv, (pos_pt1[0], pos_pt2[1]), 4, (255, 255, 255), thickness=-1)

        sh_y1 = (24, 12) if pt2[1] > pt1[1] else (-12, 0)
        sh_y2 = (0, -12) if pt2[1] > pt1[1] else (24, 12)
        c1 = 0 if pos_pt1[0] > self.size / 3 else (255, 255, 255)  # white text in left (darker) part
        c2 = 0 if pos_pt2[0] - 45 > self.size / 3 else (255, 255, 255)
        if self.normalized_display:
            pt1 = f"{pt1[0] / 255:.1%}", f"{pt1[1] / 255:.1%}"
            pt2 = f"{pt2[0] / 255:.1%}", f"{pt2[1] / 255:.1%}"
        cv2.putText(sv, f"S.{pt1[1]}", (pos_pt1[0] + 1, pos_pt1[1] + sh_y1[1]),
                    self.font, 0.75, c1, lineType=cv2.LINE_AA)
        cv2.putText(sv, f"B.{pt1[0]}", (pos_pt1[0] + 1, pos_pt1[1] + sh_y1[0]),
                    self.font, 0.75, c1, lineType=cv2.LINE_AA)
        pt21_text = f"S.{pt2[1]}"
        # Baseline info https://stackoverflow.com/q/51285616
        (w, _), _ = cv2.getTextSize(pt21_text, self.font, 0.75, thickness=1)
        cv2.putText(sv, pt21_text, (pos_pt2[0] - w, pos_pt2[1] + sh_y2[1] - 1),
                    self.font, 0.75, c2, lineType=cv2.LINE_AA)
        pt20_text = f"B.{pt2[0]}"
        (w, _), _ = cv2.getTextSize(pt20_text, self.font, 0.75, thickness=1)
        cv2.putText(sv, pt20_text, (pos_pt2[0] - w, pos_pt2[1] + sh_y2[0] - 1),
                    self.font, 0.75, c2, lineType=cv2.LINE_AA)
        cv2.imshow(self.window_name, sv)

    def set_rect(self, pt1, pt2, hilight=None):
        pt1 = max(min(pt1[0], 255), 0), max(min(pt1[1], 255), 0)
        pt2 = max(min(pt2[0], 255), 0), max(min(pt2[1], 255), 0)
        self._lower_color = min(pt1[1], pt2[1]), min(pt1[0], pt2[0])
        self._upper_color = max(pt1[1], pt2[1]), max(pt1[0], pt2[0])
        self.draw_rect(hilight)

    def set_value(self, hue):
        hue = max(min(hue, 179), 0)
        x_pos = self.hue_to_pos(hue)
        self.sv = self.create_sat_br_rect(hue)
        cv2.line(self.sv, (x_pos, self.size), (x_pos, self.size + self.slider_height), (255, 255, 255), 2)
        disp_hue = f"{hue/179*360:.1f}" if self.normalized_display else str(hue)
        cv2.putText(self.sv, f"Hue {disp_hue}", (0, self.size + self.slider_height - 2), cv2.FONT_HERSHEY_PLAIN, 1, 0, lineType=cv2.LINE_AA)
        self.hue = hue
        self.draw_rect()

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
