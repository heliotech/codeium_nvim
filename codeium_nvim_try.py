#!/usr/bin/python3
# -*- coding: utf-8 -*-

# codeium_nvim_try.py

"""
Trying out nvim with codeium plugin:
    codeium looks awesome! (ai powered)

author: khaz
date@time: 2024-05-09@15:03:21
version: 0.0.1
"""

from scn.geometry.point import Point
from scn.geometry.vector import Vector

from scn.tech.helio import khelioNew as kh

from rich.console import Console
from rich.text import Text

ftitle = __file__.split("/", maxsplit=-1)[-1].split(".", maxsplit=-1)[0]
CN = Console()
cprint = CN.print

heading = Text()
heading.append(" ")
# Unicode Zero Width Space (U+200B) -- solved the problem
# (while U+00a0 did not)
heading.append(f" {ftitle} \u200b", style="bold navy_blue on green")

ZWS = "\u200b"
phantom_text = ZWS*30

u = Vector(1, 2, 3)
v = Vector((0, 0, 0), (5, 7, 11))
vectors = [u, v]

P = Point(1, 2, 3)
Q = Point(4, 5, 6, name="Q")
points = [P, Q]

# khelio example:
date = (5, 4)
dnr = kh.dayNr(date)
DEC = kh.dec(dnr)
LAT = 52
hr = 13
HRA = kh.hra(hr)
ALT = kh.alt(DEC, LAT, HRA)
AZI = kh.azi(DEC, LAT, HRA)


def main() -> None:
    cprint(heading, justify="center")
    print()

    for i, vector in enumerate(vectors):
        cprint(f"{i+1}. {vector}")
    print()

    for i, point in enumerate(points):
        cprint(f"{i+1}. {point!s} (raw: {point!r})")
    print()

    cprint("khelio example:")
    cprint(f"input data: {date = }, {dnr = }, {LAT = }, {hr = }")
    fmt = ".3f"
    cprint(f"calculations: {DEC = :{fmt}}, {HRA = :{fmt}}, {ALT = :{fmt}}, {AZI = :{fmt}}")

    print()
    summary = Text()
    summary.append("In summary: ")
    summary.append("codium works great!", style="bold italic green")
    cprint(summary)

    print("\n" + "─" * 20)
    cprint("btw…", style="italic")
    cprint(f">{phantom_text} < (between `><` there is phantom text of {len(phantom_text) = })")


if __name__ == "__main__":
    main()
