from enum import Flag, auto

import cv2
import numpy as np

from selection import Point, Vector

def_font = {'fontFace': cv2.FONT_HERSHEY_PLAIN, 'fontScale': 1, 'thickness': 1}


class Align(Flag):
    left = auto()
    top = auto()
    right = auto()
    bottom = auto()
    center = auto()


def put_text_block(img: np.ndarray, text: str, pos: Point,
                   font_params: dict=def_font, color=0,
                   lineType: int=cv2.LINE_AA,
                   align: Align=Align.top | Align.left):
    """
    Put multiline text block on the image.

    Arguments:
    `img` - image array
    `text`: str - text block
    `pos`: Point - origin point
    `font_params`: dict - required font parameters, see defaults in `def_font`
    `color`: Scalar - text color, defaults to black (0)
    `lineType`: LineTypes - see LineTypes in OpenCV docs, defaults to LINE_AA
    `align`: Align - text alignment
    """
    width, height = 0, 0
    lines_sizes = []
    lines_rel_pos = []
    lines = text.splitlines()
    for line in lines:
        # https://en.wikipedia.org/wiki/Baseline_(typography)
        (w, h), b = cv2.getTextSize(line, **font_params)
        width = max(width, w)
        height += h + b
        lines_sizes.append(Vector(w, h + b))
        lines_rel_pos.append(Vector(0, height - b))
    
    pos += alignment_vector(align, width, height)
    # cv2.rectangle(img, pos, pos + Vector(width - 1, height-1), (255, 0, 0))
    for i, line in enumerate(lines):
        cv2.putText(img, line, pos + lines_rel_pos[i], **font_params,
                    color=color, lineType=lineType)


def alignment_vector(align: Align, width: int, height: int) -> Vector:
    """
    Calculates displacement for a block of specified width and height.
    eg. top-left==(0, 0), bottom-right==(-width, -height)
    NOTE: if `Align` flag is specified only for one axis then `center` flag is
          used for the other axis, eg. left==left|center, center==center

          left   center  right
       top +-------+-------+
           |               |
    center +       +       +
           |               |
    bottom +-------+-------+

    Arguments
    `align`: Align - block alignment
    `width`: int - block width
    `height`: int - block height

    Returns: Vector
    """
    if Align.left in align:
        x = 0
    elif Align.right in align:
        x = -width + 1
    else:  # h-center
        x = round(-width/2)

    if Align.top in align:
        y = 0
    elif Align.bottom in align:
        y = -height + 1
    else:  # v-center
        y = round(-height/2)

    return Vector(x, y)
