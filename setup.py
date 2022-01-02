from datetime import date
import setuptools
setuptools.setup(
    name="hsv_color_picker",
    version="{d.year}.{d.month}{d.day:02}".format(d=date.today())[2:],
    packages=["hsv_color_picker"],
)
