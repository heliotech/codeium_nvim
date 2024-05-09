#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Class representing a cartesian point,
with methods for polar coordinates.

New version, 24.04.06 -- for 3D points

src =
https://realpython.com/python-property/#providing-computed-attributes
"""

from matplotlib.axes import Axes
from typing import Callable
# from numpy import arctan2, array, cos, degrees, linalg, sin, sqrt, radians
import numpy as np
import logging
from numbers import Number
from pprint import pprint
# from inspect import stack as st

from softdev.m_iters import format_numbers as fnrs
from softdev.dbg import get_cprintd
from softdev.custom_logger import get_alogger
from softdev.m_iters import is_iterable
logging.getLogger('matplotlib').setLevel(logging.ERROR)

DBG = True
# DBG = False
ftitle = __file__.split("/", maxsplit=-1)[-1].split(".", maxsplit=-1)[0]
logger = get_alogger("new point", ftitle, level=logging.DEBUG, oneline=True,
                     datetime=False, fname=True, lineno=True, color="yellow")
logger.propagate = False
lgd = logger.debug
cprintd = get_cprintd("PT", ftitle, fg="blue", bg="green")


class Coordinate:
    """ Representation of x, y coordinates (with descriptor protocol (?)) """
    def __init__(self, default=None):
        self.default = default

    def __set_name__(self, owner, name):
        # print(f"`Coordinate.__set_name__(owner={owner}, name={name}`, "
        #       f"setting name = {name}")
        self._name = name

    def __get__(self, instance, owner):
        value = instance.__dict__.get(self._name, self.default)
        if value is self.default:
            return value
        value = value if not int(value) == value else int(value)
        return value

    def __set__(self, instance, value):
        try:
            # lgd(f"setting {value!r} on {instance!r}")
            # lgd(f"{value = }")
            instance.__dict__[self._name] = float(value)
        except (TypeError, ValueError) as e:
            if value is None:
                instance.__dict__[self._name] = None
                return
            else:
                err = f'"{self._name}" must be a number ({value=})'
                raise ValueError(err) from e


# @dataclass
class Point:
    """ Representation of a point, 2D or 3D

    Remarks:
        Ad comparison: if other is a number
            returns all(coord == other for coord in self.coords)
    """

    _x: Coordinate = Coordinate()
    _y: Coordinate = Coordinate()
    z_: Coordinate = Coordinate(default=None)

    name: str = "P"
    color: str = "blue"
    marker: str = "."
    size: int = 60
    fmt: str = ".3f"

    # def __init__(self, x, y, name="P"):
    #     print("`Point.__init__(x, y)`")
    #     self.x = x
    #     self.y = y
    #     self.name = name
    #     self._distanceOrg = None

    def __init__(self, x:Coordinate, y:Coordinate, z: Coordinate = None, *,
                 name="P", color: str = "blue", marker: str = ".",
                 size: int = 60, fmt: str = ".2f"):

        self._x = x
        self._y = y
        self._z = z
        self.name = name
        self.size = size
        self.color = color
        self.fmt = fmt
        if z is not None:
            self.coords = np.array([self.x, self.y, self.z])
        else:
            self.coords = np.array([self.x, self.y])

    import warnings

    warnings.filterwarnings("ignore")

    # def __repr__(self):
    #     # return f"Point({self.x}, {self.y})"
    #     return f"Point{self.as_cartesian}"

    def __str__00(self):

        x_coord = f"x={self.x}"
        y_coord = f", y={self.y}"
        z_coord = f", z={self.z}" if self.x is not None else ""
        return (f"{self.__class__.__name__}({x_coord}{y_coord}{z_coord}, "
                f"name={self.name!r})")

    def __str__(self):

        # return "(" + ", ".join(f"{c:{self.fmt}}" for c in self.coords) + ")"
        return "(" + ", ".join(f"{c:{self.fmt}}" for c in self.coords) + ")"

    def __add__(self, other):
        try:
            # cprintd("the try…", source="__add__", lno=133)
            new_coords = self.coords + other.coords
            # cprintd("…done :)", source="__add__", lno=135)
        except AttributeError:
            # other = Point(*other) if len(other) > 2 else Point(*other + [0])
            if is_iterable(other):
                if len(self) == 2 and len(other) == 2:
                    new_coords = self.coords + np.array(other)
                # new_coords = self.coords + other.coords
            else:
                new_coords = [coord + other for coord in self.coords]
        except ValueError:
            new_z = self.coords[2] if len(self.coords) == 3 else\
                other.coords[2]
            new_coords = np.hstack((self.coords[:2] + other.coords[:2],
                                    new_z))
        except Exception as e:
            cprintd("third except…", source="__add__", lno=144)
            lgd(f"{e.__class__.__name__} / {e}")

        return Point(*new_coords)

    def __eq__(self, other):
        if isinstance(other, Point):
            return all(self.coords == other.coords)
        elif isinstance(other, Number):
            return all(coord == other for coord in self.coords)

        return False

    def __getitem__(self, index):
        return self.coords[index]

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return len(self.coords)

    def __mul__(self, other):
        """ Multiplications of point by another point or by value

            If other is Point: returns scalar multiplication of coords.
        """

        if isinstance(other, Number):
            # newX, newY, newZ = self.x*value, self.y*value, self.z*value
            new_coords = self.coords*other
            return Point(*new_coords, name=f"{self.name}*{other}")

        elif isinstance(other, Point):
            return self.coords.dot(other.coords)

    def __next__(self):
        if self.index >= len(self.coords):
            raise StopIteration
        value = self.coords[self.index]
        self.index += 1
        return value

    def __repr__(self):

        fmt = self.fmt
        try:
            return (f"{self.__class__.__name__}({self.x:{fmt}}, "
                    f"{self.y:{fmt}}, {self.z:{fmt}}, name={self.name!r})")
        except TypeError:
            return (f"{self.__class__.__name__}({self.x:{fmt}}, "
                    f"{self.y:{fmt}}, name={self.name!r})")

    def __setitem__(self, index, value):
        self.coords[index] = value

    def __sub__(self, other):
        try:
            new_coords = self.coords - other.coords
        except AttributeError:
            new_coords = self.coords - np.array(other)
        except ValueError:
            other = np.array(other)
            part = self.coords[:2] - other[:2]
            new_z = self.coords[2] if len(self.coords) == 3 else\
                0 - other[2]
            new_coords = np.hstack((part, [new_z]))

        return Point(*new_coords)

    def __truediv__(self, other):
        """Point division by a number"""

        try:
            # return Point(self.x/other, self.y/other, self.z/other,
            #              f"{self.name}_")
            return Point(*self.coords/other, name=f"{self.name}_")
        except Exception as e:
            print(f"Point: division error, {self}/{other} => "
                  f"{e.__class__.__name__} / {e}")

    def _dist_0(self):
        """ Calculating distance of horizontal projection from (0, 0) """

        return np.sqrt(self.coords[0]**2 + self.coords[1]**2)

    def _scatter2D(self, ax):
        """ Scattering for 2D point/axes case """

        ax.scatter(self.x, self.y, marker=self.marker, color=self.color,
                   s=self.size)

    @property
    def as_cartesian(self):
        return self.coords

    @property
    def as_polar(self):
        return self.distance, *self.angle

    @property
    def angle(self, reference_point=None):
        """ Calculating the angle with respect to the reference point"""

        if reference_point is None:
            dx = self.x
            dy = self.y
            dz = self.z
        else:
            dx = self.x - reference_point.x
            dy = self.y - reference_point.y
            dz = self.z - reference_point.z

        ang_xy = np.degrees(np.arctan2(dy, dx))
        r_xy = np.sqrt(dx**2 + dy**2)
        # return np.arctan2(self.z, r_xy)
        ang_z = np.degrees(np.arctan2(dz, r_xy))

        return ang_xy, ang_z

    @property
    def azimuth(self):
        """Returns the solar azimuth of the point (with respect to (0, 0, 0)"""

        result = 'error'
        # GeoGebra:
        # dFi = If(yY ≥ 0, 180°, If((yY < 0) ∧ (xX > 0), 360°, 0°))
        # atancXY = atand(xX / yY) + dFi
        # (in kh world)
        if self.coords[1] >= 0:
            dFi = np.pi
        elif self.coords[1] < 0 and self.coords[0] > 0:
            dFi = 2*np.pi
        else:
            dFi = 0
        try:
            # printd(f"calculating azimuth: {self.coords[0]}/{self.coords[1]}")
            result = np.rad2deg(np.arctan(self.coords[0]/self.coords[1])+dFi)
        except ZeroDivisionError:
            lgd(f"ZeroDivisionError, result = 270 for {self.name}: "
                f"{self.coords}", err=1)
            # Solution for W, E (and 0):
            result = 270 if self.coords[0] > 0 else (90 if self.coords[0] < 0
                                                     else 0)
        return result

    @property
    def altitude(self):
        """Returns the solar altitude of the point
           (with respect to (0, 0, 0)"""

        # d = np.sqrt(self.coords[0]**2 + self.coords[1]**2)
        d = self._dist_0()
        return np.rad2deg(np.arctan(self.coords[2]/d))

    def copy(self):
        """ Returns copy of self """

        return Point(*self.coords, name=self.name, color=self.color,
                     marker=self.marker, size=self.size, fmt=self.fmt)

    def distance(self, other=None):
        """Distance between self and other, or to (0, 0)"""

        if other is None:
            # other = Point(0, 0, 0)
            other = Point(*(0,)*len(self))

        return np.linalg.norm(self.coords - other.coords)

#    @property
#    def distanceOrg(self):
#        if self._distanceOrg is None:
#            self._distanceOrg = dist((0, 0), (self.x, self.y))
#        return self._distanceOrg

    @classmethod
    def from_spherical(cls, r: float, ORI: float, TIL: float,
                       name: str = None, color: str = "b") -> 'Point':
        """ Constructor -- from spherical coordinates

            Args:
                r - radius,
                ORI (float): xy angle, between y-axis and line from (0,0)
                               to projection of point onto xy-plane,
                               usually between 0 and 180,
                TIL (float): angle between z-axis and vector from point
                               to (0,0), usually between [0, 360].
                """

        rads = np.radians
        # x = -r*sin(rads(ORI))*sin(rads(TIL))
        # y = -r*cos(rads(ORI))*sin(rads(TIL))
        # z = r*cos(rads(TIL))
        x = -r*np.sin(rads(ORI))*np.cos(rads(TIL))
        y = -r*np.cos(rads(ORI))*np.cos(rads(TIL))
        z = r*np.sin(rads(TIL))
        name = name if name is not None else "P_f_sph"
        return cls(x, y, z, name=name, color=color)

    def get_c_inc_Point(self, other):
        """ (self, other) -> (self.azimuth, inc angle)

        Calculates AZI, INC between self and other (before getCTPoint)
        """

        # before:
        # dotProd = np.dot(other.coordsAr, self.coordsAr)
        # magnProd = norm(other.coordsAr)*norm(self.coordsAr)
        # acos = dotProd/magnProd
        # ttAngle = np.rad2deg(np.arccos(acos))
        # new:
        dot_prod = other*self
        # same as P.distance() (looks like…)
        magn = np.linalg.norm(other.coordsAr)*np.linalg.norm(self.coordsAr)
        acos = dot_prod/magn
        INC = np.rad2deg(np.arccos(acos))
        return (self.azimuth, INC)

    @property
    def get_c_point(self):
        """Returns a pair of values: solar azimuth and solar altitude

           i.e. a characteristic point
        """

        return (self.azimuth, self.altitude)

    def get_P_on_orbit(self, radius: float, angle: float, color: str = "b",
                       RefP: "Point" = None,
                       rproj: Callable = None, name: str = None) -> "Point":
        """ !TODO: for old verson! Getting point on orbit of self, radius apart

            Args:
                RefP: Point, reference point, optional"""

        # !TODO: with RefP

        P_tr = self.move(radius, 0, color=color)
        P = P_tr.rotatexy(self, angle=angle, color=color)
        suffix = f"_{{{radius}rot{angle}°}}"
        name = r"$" + self.name + suffix + r"$"
        P.name = name
        return P

    def getRange(self, other, dens):
        """ Getting range between self and other

        Args:
            other (Point)
            dens (int): total number of points in the range
        """

        pt_dif = (other - self)*(1/(dens - 1))
        range_lst = list([self])
        for i in range(1, dens - 1):
            inter_p = self + pt_dif*i
            inter_p.name = f"P{i}"
            range_lst.append(inter_p)
        range_lst.append(other)

        return range_lst

    def middle(self, other, name=None, color=None, size=60):
        """ The middle point between self and the other """

        name = name if name else f"({self.name}+{other.name})/2"
        color = color if color else self.color

        x = (self.x + other.x)/2
        y = (self.y + other.y)/2
        z = (self.z + other.z)/2

        return Point(x, y, z, name=name, color=color, size=size)

    def move(self, dx: float = None, dy: float = None,
             dz: float = None, color: str = "b") -> "Point":
        """ Moving the point by (dx, dy, dz) """

        new_x = self.x + dx if dx else self.x
        new_y = self.y + dy if dy else self.y
        new_z = self.z + dz if dz and self.z is not None else self.z
        return Point(new_x, new_y, new_z, name="P_", color=color)

    def plot_line(self, other, ax, color="blue", **kwargs):
        """ Plotting a line to the other point """

        xes, yes, zes = list(zip(self.coords, other.coords))
        cprintd(f"{self = }, {other = }\n{xes = }, {yes = }, {zes = }",
                source="plot_line")
        ax.plot(xes, yes, zes, color=color, **kwargs)

    @classmethod
    def projected_cart(cls, x: float, y: float, r, method: Callable,
                       name: str = "P", color: str = "blue", marker: str = ".",
                       size: int = 60, fmt: str = ".2f"):
        """ !TODO: for old verson! Creates a point with
        projected coordinates (???)

        Args:
            x, y (float): cartesian coordinates
            r (float): projection radius
            method (function): projection function

        Returns:
            A Point with projected, cartesian coordinates.
        """

        x_coord = x
        y_coord = r - method(r, y)

# class Point:
#     x: Coordinate = Coordinate()
#     y: Coordinate = Coordinate()
#
#     name: str = "P"
#     color: str = "blue"
#     marker: str = "."
#     size: int = 60
#     fmt: str = ".2f"

        return cls(x_coord, y_coord, name=name, color=color, marker=marker,
                   size=size, fmt=fmt)

    @staticmethod
    def random_inbox(x0, x1, y0, y1, n, fmt=".0f"):
        """ !TODO: for old verson! Getting n random points from rectangle """

        import random as rnd

        if n > (x1 - x0 + 1)*(y1 - y0 + 1):
            err = "Unique random numbers is smaller then parameter n"
            raise ValueError(err)

        xes = list(range(x0, x1+1))
        rnd.shuffle(xes)
        yes = list(range(y0, y1+1))
        rnd.shuffle(yes)

        points = [Point(x, y, f"{x, y}")
                  for (x, y) in zip(xes[:n], yes[:n])]

        return points

    @staticmethod
    def random_incircle(x0, y0, r, n, marker=".", size=50, fmt=".0f"):
        """ !TODO: for old verson! Getting n random points from rectangle """

        import random as rnd

        def count_points_in_circle(r):
            count = 0
            for x in range(-r, r + 1):
                for y in range(-r, r + 1):
                    if x**2 + y**2 <= r**2:
                        count += 1
            return count

        totalPoints = count_points_in_circle(r)
        if n > totalPoints:
            err = "Unique random numbers is smaller then parameter n"
            raise ValueError(err)

        angles = rnd.sample(range(360 + 1), n)
        rs = rnd.choices(range(r + 1), k=n)

        P0 = Point(x0, y0)
        points = []
        for i, (r, ang) in enumerate(zip(rs, angles)):
            P = P0 - Point.polar(r, ang)
            P.name = f"<{r, ang}"
            P.marker = marker
            P.size = size
            points.append(P)

        return points

    def rotatexy_(self, other=None, *, angle=0, color="b"):
        """ !TODO: this is xy-rotation!

            clockwise direction """

        Q = other if other is not None else\
            (Point(0, 0) if len(self) == 2 else Point(0, 0, 0))

        P_ = self - Q
        angle_ = np.radians(angle)
        # !DBG: xy-plane!
        P_rotated = Point(P_.x*np.cos(angle_) - P_.y*np.sin(angle_),
                          P_.x*np.sin(angle_) + P_.y*np.cos(angle_))
        P_rotated = P_rotated if len(self) == 2 else\
            Point(P_rotated.x, P_rotated.y, 0)
        try:
            P = P_rotated + Q
        except Exception as e:
            cprintd(f"{P_rotated = }, {Q = }" +
                    str(e.__class__) + "bla", source="rotatexy", lno=431)
            lgd(f"{e.__class__.__name__} / {e}")
            _coords = np.array(self.coords[:2] + Q)
            P = Point(*_coords)
        P.color = color

        return P

    def rotatexy(self, angle=0, *, other=None, color="b"):
        """ !TODO: this is xy-rotation!

            anticlockwise direction """

        Q = other if other is not None else\
            (Point(0, 0) if len(self) == 2 else Point(0, 0, 0))

        P_ = self - Q
        r = P_.distance()
        angle_ = np.radians(angle)
        new_x = r * (-np.sin(angle_))
        new_y = r * (-np.cos(angle_))
        P_rotated = Point(new_x, new_y) if len(self) == 2 else\
            Point(new_x, new_y, 0)

        P = P_rotated + Q
        P.color = self.color
        P.name = f"{self.name}_rot{angle:.0f}"

        return P

    def scatter(self, ax: Axes, s: int = None, marker: str = None,
                linewidths=1, color: str = None, proj: Callable = None):
        """ Scattering point """

        if len(self.as_cartesian) == 2:
            self._scatter2D(ax)
            return

        s = s if s is not None else self.size
        marker = marker if marker is not None else self.marker
        color = color if color is not None else self.color
        # lgd(f"{self!r}")
        # ax.scatter(self.x, self.y, self.z, s=s, marker=marker, color=color,
        #            linewidths=linewidths)
        ax.scatter(self.x, self.y, self.z, marker=marker, color=color,
                   s=self.size, linewidths=linewidths)

    def text(self, ax: Axes, name: str = None, color: str = "white",
             fontdict: dict = None, direction: str = None,
             proj: Callable = None, radius: float = None):
        """ !TODO: for old version! Labeling the point """

        directions = dict(E=0, NE=45, N=90, NW=135,
                          W=180, SW=225, S=270, SE=315)
        direction = directions[direction] if direction is not None\
            else directions['E']
        ha = "left" if 270 <= direction <= 360 or 0 <= direction <= 90\
            else "right"
        va = "bottom" if 0 <= direction <= 180 else "top"
        name = self.name if name is None else name
        dpi = ax.get_figure().dpi
        fontdict = fontdict if fontdict is not None else dict(size=12)
        _fs = fontdict['size']
        fontdict['ha'] = ha
        fontdict['va'] = va
        fontdict['color'] = "black"
        fs = fontdict['size']
        rcf = 1/dpi  # radius coef.
        # not clear about rcf…
        radius = rcf*fs if not radius else radius
        PTxt = self.get_P_on_orbit(radius, direction)
        xtx, ytx, *ztx = PTxt.as_cartesian
        ztx = ztx[0] + (_fs*2/dpi) if ztx else None
        ytx = proj(ytx) if proj is not None else ytx
        if ztx is not None:
            ax.text(xtx, ytx, ztx, name, fontdict=fontdict,
                    bbox=dict(facecolor="wheat", alpha=0.5, edgecolor=color,
                              boxstyle="round"))
        else:
            ax.text(xtx, ytx, name, fontdict=fontdict,
                    bbox=dict(facecolor="wheat", alpha=0.5, edgecolor=color,
                              boxstyle="round"))

    def translate(self, vector, dst=1, name=None):
        """ Translation of the point by dist along vector """

        # lgd(f"{self.x = }, {self.y = }, translating by {vector = !s}")
        # lgd(f"DBG: {vector.x = }, {vector.y = }")
        # lgd(f"DBG: {dist = }")
        name = name if name is not None else f"{self.name}_tr"

        dx = vector.x*dst
        dy = vector.y*dst
        dz = vector.z*dst if vector.z is not None else 0
        # lgd(f"translate: {dx = }, {dy = }, {dz = }")
        # new_coords = self.coords = [self.x + dx, self.y + dy, self.z + dz] if\
        #     len(self.coords) == 3 else self.coords + [dx, dy]
        new_coords = (self.coords + [dx, dy, dz]) if\
            len(self.coords) == 3 else self.coords + [dx, dy]

        return Point(*new_coords, name=name)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.coords[0] = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.coords[1] = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = value
        self.coords[2] = value


def misc():
    # P = Point(3, 4)
    P = Point(-3, -3)
    print(f"{P = }")
    print(f"{P.as_cartesian = }")
    print(f"{P.as_polar = }")
    print("Iteration:")
    for i, coord in enumerate(P):
        print(f"{i}) {coord = }")
    # Q = Point(7, 11)
    Q = Point(7, 0, "Q")
    print("Access by index:")
    print(f"{Q[0] = }")
    print(f"{Q[1] = }")
    Q[0] = 3.5
    Q[1] = 2
    print(f"{Q[0] = }")
    print(f"{Q[1] = }")
    print(f"{Q.distance(P) = }")
    print(f"{P.coords = }")
    print(f"{Q.coords = }")
    print(f"{list([P, Q]) = }")
    print(f"{list(zip(P, Q)) = }")
    print(f"{P*3 = } ({P = })")

    print(f"{Point.random_inbox(3, 5, -1, 8, 5) = }")
    print(f"{Point(1, 2, '(1, 2)', fmt='.3f') = }")


def annotate_to_axes(ax: Axes, pt: Point, annot: str = None) -> None:
    """ Point 3D: extra annotation

    Marking with segments values at each axis,
    plotting dashed lines to (0, 0) and to each axis.
    """

    clr = "black"
    cprintd(f"{pt = }", source="annotate_to_axes")
    # annotation points:
    pt0 = Point(pt.x, pt.y, 0, color=clr)
    pt0x = Point(pt.x, 0, 0, color=clr)
    pt0y = Point(0, pt.y, 0, color=clr)
    pt0z = Point(0, 0, pt.z, color=clr)
    points = [pt0, pt0x, pt0y, pt0z]
    for p in points:
        p.scatter(ax)

    # annotation segments:
    # xes, yes, zes = list(zip(pt, pt0))
    # ax.plot(xes, yes, zes, "--", color="lightgray", lw=0.5)
    pt.plot_line(pt0, ax, clr, lw=1, ls="--")
    pt.plot_line(pt0z, ax, clr, lw=1, ls="--")
    pt0.plot_line(pt0x, ax, clr, lw=1, ls="--")
    pt0.plot_line(pt0y, ax, clr, lw=1, ls="--")

    # text:
    pt.text(ax, f"{pt}", color=clr)


def get_P_on_orbit_demo():

    """
        PURPOSE:
            Get a sattelite point.
            Given the main point and radius, get a sattelite point,
            around the main, radius apart, rotated by given angle.
    """

    cprintd("starting…", source="get_P_on_orbit_demo", lno=527, head=">>>")
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    from matplotlib.patches import Circle
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # initial rotation of the axes:
    ax.view_init(azim=75, elev=30)
    from scn.geometry.misc import plot_axes

    rt = 1
    at = 15
    Test = Point(2, 3)
    Test.scatter(ax, color="red")
    Test_ = Test + [rt, 0]
    Test_.scatter(ax, color="gray")
    Test_r = Test.rotatexy(Test, angle=at)
    Test_r.scatter(ax, color="green", marker="+", s=300)

    x, y = 3, 5
    P = Point(x, y)
    ang = 90
    r = 0.5
    circle = Circle(P, radius=r, edgecolor='black', linestyle="--",
                    facecolor='none')
    Q = P.get_P_on_orbit(2*r, 10)

    plot_axes(ax, P, xminmax=[-1, 1])
    ax.scatter(*P, s=30)
    ax.scatter(*Q, s=15)

    # !DBG:
    P_tr = P + [r, 0]
    P_tr.scatter(ax)

    ax.add_patch(circle)
    plt.axis('scaled')
    dx = dy = 3
    cprintd(f"{(0, x + dx, 0, y + dy) = }")
    ax.axis((0, x + dx, 0, y + dy))

    # projected point demo:
    print("projected demo")
    import scn.geometry.khgeom as khg  # noqa: E402
    x_pp, y_pp = 5, 6
    p_proj = Point.projected_cart(x_pp, y_pp, 11, khg.rndr, color="green")
    p_proj.scatter(ax)
    print(f"p_proj = Point.projected_cart({x_pp}, {y_pp}, 11, khg.rndr) = "
          f"{p_proj}")

    plt.show()


def point_add_demo():
    """ Addition of points, 2D, 3D; new demo """

    cprintd("Addition demo, 3D")
    P = Point(1, 2, 3, "P")
    Q = Point(5, 7, 11, "Q")
    cprintd(f"{P = }, {Q = }")
    cprintd(f"{P + Q = }")
    cprintd("Addition demo, 2D")
    R = Point(13, 17, name="R")
    S = Point(19, 23, name="S")
    cprintd(f"{R = }, {S = }")
    cprintd(f"{R + S = }")
    cprintd("Addition demo, 2D + 3D")
    R2 = Point(29, 31, name="R2")
    S3 = Point(33, 39, 41, name="S3")
    cprintd(f"{R2 = }, {S3 = }")
    cprintd(f"{R2 + S3 = }")
    cprintd("Addition demo, 3D + 2D")
    R3 = Point(29, 31, name="R2")
    S2 = Point(33, 39, 41, name="S2")
    cprintd(f"{R3 = }, {S2 = }")
    cprintd("Addition demo, 2D + 2D")
    O2 = Point(29, 31, name="O2")
    P2 = Point(21, 39, 41, name="P2")
    cprintd(f"{O2= }, {P2 = }")
    cprintd(F"{O2 + P2 = }")


def points_division_demo():
    P = Point(3, 5)
    r = 2
    result = P/r
    print(f"Division of {P = }, by {r = } => {result}")


def point3D_demo():
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import logging

    size = 400
    P = Point(1, 2, 3, "P", color="lightblue", size=size)
    Q = Point(5, 7, 11, "Q", color="red", size=size)
    print(f"{P.copy() = }")
    print(f"{P = }")
    print(f"{Q = }")
    print(f"{P + Q = }")
    print(f"{(P + Q).coords = }")
    print(f"{Q/3 = }")
    # print(f"{Q/P = }")  # unsupported operand
    print(f"{Q*3 = }")
    print(f"{P.angle = }")
    print(f"{P.distance = }")
    print(f"{P.as_polar = }")
    print(f"{P.middle(Q) = }")
    print(f"{Point(0, 0, 0).translate(vector=Point(1, 2, 3)) = }")
    ORI = 180
    TIL = 60
    print(f"{ORI = }, {TIL = }")
    print(f"{Point.from_spherical(1, ORI, TIL) = !s}")
    P_fsph = Point.from_spherical(1, ORI, -TIL)
    print(f"{P_fsph = }")
    # print(f"{

    # Tworzenie figury i osi 3D
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('matplotlib.matplotlib').setLevel(logging.INFO)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection='3d')
    P.scatter(ax)
    Q.scatter(ax)
    Q.plot_line(P, ax, "blue")
    PQ = P.middle(Q, "PQ", size=size)
    PQ.scatter(ax)

    plt.show()




def random_demo():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scn.khmath import Line2D
    from scn.geometry.misc import plot_axes

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8))

    # for rectangle:
    x0, x1 = -3, 5
    DelX = abs(x1 - x0)
    y0, y1 = -2, 7
    DelY = abs(y1 - y0)
    n = 17
    randomPoints = Point.random_inbox(x0, x1, y0, y1, n)
    for P in randomPoints:
        ax0.scatter(*P)
        xytext = Line2D.get_P(P, DelX*0.025, m=1)
        ax0.text(*xytext, P.name)

    plot_axes(ax0, (0, 0))

    # for circle:
    xc, yc = 3, -1
    r = 11

    circle = Circle((xc, yc), radius=r, edgecolor='black', linestyle="--",
                    facecolor='none')
    plot_axes(ax1, Point(0, 0))

    randomPointsCircle = Point.random_incircle(xc, yc, r, n, marker="3",
                                               size=600)
    pprint(randomPointsCircle)
    for P in randomPointsCircle:
        ax1.scatter(*P, marker=P.marker, s=P.size)
        # xytext = P + [P.x + P.x*0.005, P.y + P.y*0.005]
        xytext = P
        # xytext = Line2D.get_P(P, DelX*0.025, m=1)
        ax1.text(*xytext, P.name)

    # setting axes limits:
    # dx = DelX*0.1
    # dy = DelY*0.1
    # ax.axis((x0 - dx, x1 + dx, y0 - dy, y1 + dy))
    ax1.add_patch(circle)
    plt.axis('scaled')
    plt.show()


def rotation_demo():
    P = Point(5, 0, "P")
    P_ = Point(7, 4, "P_")
    Q = Point(8, 5, "Q")
    O = Point(0, 0, "O")
    points = [P, P_, Q, O]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for aPoint in points:
        ax.scatter(*aPoint)

    for i, ang in enumerate([angle for angle in range(15, 360 + 1, 15)]):
        Prot = P.rotatexy(angle=ang)
        ax.scatter(*Prot, c="gray")

        P_rot = P_.rotatexy(Q, angle=ang)
        ax.scatter(*P_rot, c="gray")

    plt.show()


def rotation_single_demo():
    """ Demo of rotation """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scn.khmath import Line2D
    from scn.geometry.misc import plot_axes

    fig, ax = plt.subplots(figsize=(12, 8))

    O = Point(0, 0, "O", marker="+")
    O.scatter(ax)
    plot_axes(ax, O)

    r = 11
    circle = Circle(O, radius=r, edgecolor='black', linestyle="--",
                    linewidth=0.5, facecolor='none')
    ax.add_patch(circle)

    # P = Point.polar(r, 0)
    P = Point.from_spherical(r, 0, 0)
    P.scatter(ax)
    P1 = P.rotatexy(angle=15)
    P1.color = "green"
    P1.scatter(ax)
    P2 = P.rotatexy(angle=-15)
    P2.color = "red"
    P2.scatter(ax)

    r1 = 2
    circle1 = Circle(P1, radius=r1, edgecolor='black', linestyle=":",
                     linewidth=0.5, facecolor='none')
    ax.add_patch(circle1)
    P11 = P1.get_P_on_orbit(r1, 30)
    P11.scatter(ax)
    plt.axis('scaled')

    plt.show()


def main():
    """ Entry method """

    # random_demo()
    # rotation_demo()
    # rotation_single_demo()
    # get_P_on_orbit_demo()
    # points_division_demo()
    # point3D_demo()

    # new demos:
    point_add_demo()


if __name__ == "__main__":
    main()
