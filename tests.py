import unittest as ut

from hsv_color_picker import SliderHSV

class Test(ut.TestCase):
    def test_hue_conversion(self):
        w = SliderHSV('test', size=320)
        self.assertEqual(w.hue_to_pos(0), 0)
        self.assertEqual(w.hue_to_pos(179), 319)
        self.assertEqual(w.pos_to_hue(0), 0)
        self.assertEqual(w.pos_to_hue(319), 179)

    def test_sv_conversion(self):
        w = SliderHSV('test', size=320)
        self.assertEqual(w.vals_to_pos([0, 0]), (0, 0))
        self.assertEqual(w.vals_to_pos([255, 255]), (319, 319))
        self.assertEqual(w.pos_to_val(0), 0)
        self.assertEqual(w.pos_to_val(319), 255)

    def test_shift_hue(self):
        w = SliderHSV('test')
        w.hue = 171
        self.assertEqual(w.shift_hue(10), 1)
        w.hue = 0
        self.assertEqual(w.shift_hue(-10), 170)

if __name__ == '__main__':
    ut.main()
