import cv2
import numpy as np


class SliderHSV:
    SLIDER_H = 16  # px
    sliding = False

    def __init__(self, window_name, size=255):
        self.window_name = window_name
        self.size = size
        self.pt = 0, 0
        self.hue = 0
        self.lower_color = [0, 0]
        self.upper_color = [0, 0]

        h_comp = np.uint8([np.linspace([0., 255., 255.], [179., 255., 255.], self.size)])
        h_comp = np.broadcast_to(h_comp, (self.SLIDER_H, self.size, 3))
        self.h_comp = cv2.cvtColor(h_comp, cv2.COLOR_HSV2BGR)
        self.tpl_hsv = np.uint8([  # This is very very slow!
            (0, i, j) for i in np.linspace(0., 255., self.size)
                      for j in np.linspace(0., 255., self.size)
        ]).reshape(self.size, self.size, 3)
        self.set_value(0)
        cv2.setMouseCallback(window_name, self.on_mouse_event, param=self.window_name)
        cv2.setMouseCallback("Sat-Bri", self.on_mouse_event)

    def pos_to_hue(self, x):
        return int(x / self.size * 179)

    def hue_to_pos(self, h):
        return int(h / 179 * self.size)

    def pos_to_val(self, x):
        return int(x / self.size * 255)

    def vals_to_pos(self, h):
        return np.int16(np.array(h) / 255 * self.size)

    def on_mouse_event(self, event, x, y, flags, param):
        # https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sliding = True
            if param == self.window_name:
                self.set_value(self.pos_to_hue(x))
            else:
                self.pt = self.pos_to_val(x), self.pos_to_val(y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.sliding:
                if param == self.window_name:
                    self.set_value(self.pos_to_hue(x))
                else:
                    self.set_rect(self.pt, (self.pos_to_val(x), self.pos_to_val(y)))
        elif event == cv2.EVENT_LBUTTONUP:
            self.sliding = False

    def draw_rect(self):
        pt1 = self.lower_color[::-1]
        pt2 = self.upper_color[::-1]
        pos_pt1 = self.vals_to_pos(pt1)
        pos_pt2 = self.vals_to_pos(pt2)
        sv = self.sv.copy()
        cv2.rectangle(sv, pos_pt1, pos_pt2, (255, 255, 255))
        sh_y1 = (24, 12) if pt2[1] > pt1[1] else (-12, 0)
        sh_y2 = (0, -12) if pt2[1] > pt1[1] else (24, 12)
        cv2.putText(sv, f"Sat.{pt1[1]}", (pos_pt1[0], pos_pt1[1] + sh_y1[1]), cv2.FONT_HERSHEY_PLAIN, 0.75, 0, lineType=cv2.LINE_AA)
        cv2.putText(sv, f"Br.{pt1[0]}", (pos_pt1[0], pos_pt1[1] + sh_y1[0]), cv2.FONT_HERSHEY_PLAIN, 0.75, 0, lineType=cv2.LINE_AA)
        cv2.putText(sv, f"Sat.{pt2[1]}", (pos_pt2[0] - 45, pos_pt2[1] + sh_y2[1]), cv2.FONT_HERSHEY_PLAIN, 0.75, 0, lineType=cv2.LINE_AA)
        cv2.putText(sv, f"Br.{pt2[0]}", (pos_pt2[0] - 45, pos_pt2[1] + sh_y2[0]), cv2.FONT_HERSHEY_PLAIN, 0.75, 0, lineType=cv2.LINE_AA)
        cv2.imshow("Sat-Bri", sv)

    def set_rect(self, pt1, pt2):
        pt1 = max(min(pt1[0], 255), 0), max(min(pt1[1], 255), 0)
        pt2 = max(min(pt2[0], 255), 0), max(min(pt2[1], 255), 0)
        self.lower_color = min(pt1[1], pt2[1]), min(pt1[0], pt2[0])
        self.upper_color = max(pt1[1], pt2[1]), max(pt1[0], pt2[0])
        self.draw_rect()

    def set_value(self, hue):
        hue = min(hue, 179)
        hue = max(hue, 0)
        h_comp = self.h_comp.copy()
        x_pos = self.hue_to_pos(hue)
        cv2.line(h_comp, (x_pos, 0), (x_pos, self.SLIDER_H), (255, 255, 255), 2)
        cv2.putText(h_comp, f"Hue {hue}", (0, self.SLIDER_H), cv2.FONT_HERSHEY_PLAIN, 1, 0, lineType=cv2.LINE_AA)
        self.sv = self.create_sat_br_rect(hue)
        cv2.imshow(self.window_name, h_comp)
        cv2.imshow("Sat-Bri", self.sv)
        self.hue = hue
        self.draw_rect()

    def create_sat_br_rect(self, hue):
        self.tpl_hsv[:, :, 0] = hue
        return cv2.vconcat([cv2.cvtColor(self.tpl_hsv, cv2.COLOR_HSV2BGR), self.h_comp])
