import unittest as ut

from cv_utils import Align, Vector, alignment_vector
from hsv_color_picker import SliderHSV
from selection import RectSelection, Rect, RectElement


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


class TestVectors(ut.TestCase):
    def test_vectors_equal(self):
        self.assertEqual(Vector(3, 4), Vector(3, 4))
        self.assertNotEqual(Vector(-3, -4), Vector(3, 4))
    
    def test_rect_resize(self):
        bs = Rect(w=99, h=99)
        self.assertEqual(RectSelection.transformed_rect(
            Rect(16, 16, 1, 1), RectElement.bottomright, Vector(x=-1, y=-1), bs
        ), Rect(15, 15, 2, 2))
        # right side becomes left w/ x=16, left becomes right w/ x=16+1
        self.assertEqual(RectSelection.transformed_rect(
            Rect(16, 16, 1, 1), RectElement.left, Vector(x=1), bs
        ), Rect(16, 16, 2, 1))


class TestTextUtils(ut.TestCase):
    def test_alignment(self):
        self.assertEqual(alignment_vector(Align.left, 16, 16),
                         Vector(0, -8))
        self.assertEqual(alignment_vector(Align.top, 16, 16),
                         Vector(-8, 0))
        self.assertEqual(alignment_vector(Align.right, 16, 16),
                         Vector(-16, -8))
        self.assertEqual(alignment_vector(Align.bottom, 16, 16),
                         Vector(-8, -16))
        self.assertEqual(alignment_vector(Align.center, 16, 16),
                         Vector(-8, -8))
        self.assertEqual(alignment_vector(Align.bottom | Align.right, 16, 16),
                         Vector(-16, -16))


if __name__ == '__main__':
    ut.main()
