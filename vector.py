#!/usr/bin/python3
# -*- coding: utf-8 -*-

# vector.py

"""
Module for vector calculations

author: Sebastian Kazimierski

date@time: 2024-04-06@23:14:54
version: 0.0.1
"""

from matplotlib.axes import Axes
from dataclasses import dataclass
import numpy as np
from numbers import Number

from softdev.m_iters import format_numbers as fnrs
from softdev.m_iters import is_iterable
from scn.geometry.point import Point

from softdev.custom_logger import get_alogger
import logging
from inspect import stack as st


ftitle = __file__.split("/", maxsplit=-1)[-1].split(".", maxsplit=-1)[0]

DBG = True
# DBG = False
ftitle = __file__.split("/", maxsplit=-1)[-1].split(".", maxsplit=-1)[0]
logger = get_alogger("new vector", ftitle, level=logging.DEBUG, oneline=True,
                     datetime=False, fname=True, lineno=True, color="green")
logger.propagate = False
lgd = logger.debug


def isRoughlyZero(number):
    return round(number, 7) == 0  # 7 -- at the beginning


class Vector:
    """ Class for vector representation """

    _nrt = (0, -1, 0)  # 'north' tuple

    def __init__(self, *args, name: str = "v",
                 linestyle: str = "-", lw: int = 1, color: str = "blue"):
        """ Vector starting at p0, ending at p1

        If only p0 is provided: p0 → p1, Point(0, 0, 0) → p0
        """

        if all(isinstance(arg, Number) for arg in args):
            self._p0 = Point(*[0] * len(args), name="p0")
            self._p1 = Point(*args, name="p1")
        elif len(args) == 2 and all(is_iterable(arg) for arg in args):
            self._p0 = Point(*args[0])
            self._p1 = Point(*args[1])
        elif len(args) == 1 and is_iterable(args[0]):
            # coords0 = list(0 for _ in args[0])  # better (ChGPT):
            coords0 = [0] * len(args[0])
            self._p0 = Point(*coords0, name="p0")
            self._p1 = Point(*args[0], name="p1")
        else:
            err = f"Wrong type of arg(s), {args = }"
            raise ValueError(err)
        # self._p0 = p0
        # self._p1 = p1
        self._name = name
        self._linestyle = linestyle
        self._lw = lw
        self._color = color

        if self.p1 is None:
            lgd(f"`self.p1 is None` {self.p0 = }, {self.p0.copy() = }")
            self.p1 = self.p0.copy()
            self.p1.name = "p1"
            self.p0 = Point(*[0 for _ in self.p1], name="p0")

        self._coords = (self.p1 - self.p0).as_cartesian

    def __add__(self, other):
        """we want to add single numbers as a way of changing the length of the
        vector, while it would be nice to be able to do vector addition with
        other vectors.
            >>> from core import Vector
            >>> # test add
            ... v = Vector(0.0, 1.0, 2.0)
            >>> v1 = v + 1
            >>> v1
            Vector(0.0, 1.4472135955, 2.894427191)
            >>> v1.length - v.length
            0.99999999999999956
            >>> v1 + v
            Vector(0.0, 2.4472135955, 4.894427191)
        """

        if isinstance(other, Number):
            # then add to the length of the vector
            # multiply the number by the normalized self, and then
            # add the multiplied vector to self
            x = self.coords[0] + other
            y = self.coords[1] + other
            z = self.coords[2] + other
            return Vector((x, y, z))
        elif isinstance(other, (Point, Vector)):
            # add all the coordinates together
            # there are probably more efficient ways to do this
            new_xy = self.coords[:2] + other.coords[:2]
            try:
                new_z = self.coords[2] + other.coords[2]
                new_coords = np.hstack((new_xy, np.array(new_z)))
            except Exception as e:
                new_coords = new_xy
            return Vector(new_coords)
        else:
            err = (f"cannot add {type(self).__qualname__} and "
                   f"{type(other).__qualname__}")
            raise NotImplementedError(err)

    def __eq__(self, other):
        """
            Comparison method for two vectors.
        """
        # if np.allclose(self.x, other.x) and np.allclose(self.y, other.y)\
        #    and np.allclose(self.z, other.z):
        if np.allclose([self.coords], [other.coords]):
            return True
        else:
            return False

    def __getitem(self, index):
        return self._coords[index]

    def __iter__(self):
        self.index = 0
        return self

    def __mul__(self, other):
        """if with a number, then scalar multiplication of the vector,
            if with a Vector, then dot product, I guess for now, because
            the asterisk looks more like a dot than an X.
            >>> v2 = Vector(-4.0, 1.2, 3.5)
            >>> v1 = Vector(2.0, 1.1, 0.0)
            >>> v2 * 1.25
            Vector(-5.0, 1.5, 4.375)
            >>> v2 * v1 #dot product
            -6.6799999999999997
        """
        if isinstance(other, Number):
            # scalar multiplication for numbers
            #return Vector( *((n * other) for n in self))
            # old approach:
            # newOrigin = (self.p0[0]*other, self.p0[1]*other, self.p0[2]*other)
            # newEnd = (self.p1[0]*other, self.p1[1]*other, self.p1[2]*other)
            # new approach:
            try:
                new_p0 = self.p0*other
            except TypeError:
                new_p0 = None
            new_p1 = self.p1*other
            # x, y, *z = self.coords + other
            if new_p1 is not None:
                return Vector(new_p0, new_p1, color=self.color)
            return Vector(new_p1, color=self.color)

        elif isinstance(other, Vector):
            # dot product for other vectors
            return self.dot(other)

    def __next__(self):
        if self.index >= len(self.coords):
            raise StopIteration
        value = self.coords[self.index]
        self.index += 1

        return value

    def _north(self):
        """ Returning versor representing north """

        return Vector(self._nrt)

    def _plot_edge(self, p0: Point, p1: Point, ax: Axes,
                   lw: int = 1, c: str = "b") -> None:
        """ Plotting an edge """

        data = list(zip(p0, p1))  # zipped data: [(x1, x2), (y1, y2), (z1, z2)]
        ax.plot(*data, lw=lw, color=c)

    def __repr__(self):

        if all(value == 0 for value in self.p0):
            p0_str = ""
        else:
            p0_str = f"p0={self.p0!s}, "
        return (f"{self.__class__.__name__}({p0_str}p1={self._p1}, "
                f"name={self.name!r}, linestyle={self.linestyle!r}, "
                f"lw={self.lw}, color={self.color!r})")

    def __str__(self):

        p0_str = fnrs(self.p0.coords, 3, fixing=False, brackets=("[", "]"))\
            if not self.p0 == 0 else ""
        p1_str = fnrs(self.p1.coords, 3, fixing=False, brackets=("[", "]"))
        if not p0_str:
            return f"{Vector.__qualname__}{p1_str}"
        return f"{Vector.__qualname__}[{p0_str}, {p1_str}]"

    def __sub__(self, other):
        """Subtract a vector or number
            >>> v2 = Vector(-4.0, 1.2, 3.5)
            >>> v1 = Vector(2.0, 1.1, 0.0)
            >>> v2 - v1
            Vector(-6.0, 0.1, 3.5)
        """
        return self.__add__(other * -1)

    def __truediv__(self, number):
        """
            Divides each coordinate by the number.
        """
        # new_origin = (self.p0.x/number, self.p0.y/number,
        #               self.p0.z/number)
        # new_end = (self.p1.x/number, self.p1.y/number,
        #            self.p1.z/number)
        new_p0 = self.p0.coords / number  # origin
        new_p1 = self.p1.coords / number  # end
        return Vector(new_p0, new_p1)

    def ang_(self, v: "Vector" = None, u: "Vector" = None) -> float:
        """Returns angle between vectors (old ver.)"""

        if u is None and v is not None:
            u = v
            v = self
        elif v is None and u is None:
            err = "At leas one argument must be a vector"
            raise ValueError(err)

        dotP = v.dot(u)
        magP = v.length * u.length
        try:
            # lgd(f"trying… {dotP = :.2f}, {magP = :.2}")
            phi = np.arccos(dotP / magP) * (180 / np.pi)
            return phi
        except ZeroDivisionError:
            print('Zero division.')
            if v.length == 0 or u.length == 0:
                print('The magnitutde of at least one of the vectors is 0.')
        except ValueError:  # !!! not sure entirely
            print(ftitle +
                  ' ValueError -- solved by approximation/simplification.')
            print('np.acos(dotP/magP)*(180/np.pi) = '
                  'np.acos({}/{})*({halphpi}) = np.acos({})*({halphpi})'.
                  format(dotP, magP, dotP / magP, halphpi=180 / np.pi))
            if (dotP / magP) <= 1.0000000000000002 and (dotP / magP) > 0:
                return 0
            else:
                raise Exception('negative fraction!')

    def ang(self, v: "Vector" = None, u: "Vector" = None) -> float:
        """Returns angle between vectors (new ver., looks v. good :) )"""

        if u is None and v is not None:
            u = v
            v = self
        elif v is None and u is None:
            err = "At leas one argument must be a vector"
            raise ValueError(err)

        angle1 = np.degrees(np.arctan2(v.y, v.x) - np.arctan2(u.y, u.x))
        return angle1 % 360

    @property
    def azi(self):
        """ Calculating azimuth of the vector """

        self_proj_0 = Vector(Point(self.x, self.y, 0))
        # lgd(f"{self_proj_0 = !s}")

        return self.ang(self_proj_0, self._north())

    @property
    def coords(self):
        return self._coords

    def cross(self, other):
        """Gets the cross product between two vectors
            >>> v
            Vector(5, 1.20747670785, 60.0)
            >>> v1
            Vector(0.0, 2.0, 1.0)
            >>> v1.cross(v)
            Vector(118.792523292, 5.0, -10.0)
        """
        # I hope I did this right
        x = (self.coords[1]*other.coords[2]) - (self.coords[2]*other.coords[1])
        y = (self.coords[2]*other.coords[0]) - (self.coords[0]*other.coords[2])
        z = (self.coords[0]*other.coords[1]) - (self.coords[1]*other.coords[0])
        return Vector((x, y, z))

    def dot(self, other):
        """Gets the dot product of this vector and another.
            >>> v
            Vector(5, 1.20747670785, 60.0)
            >>> v1
            Vector(0.0, 2.0, 1.0)
            >>> v1.dot(v)
            62.41495341569977
        """
        return sum((p[0] * p[1]) for p in zip(self, other))

    def get_range(self, other, dens):
        """
           Returns range of intermediate vectors, between the two given.
        """
        vDiff = other.__sub__(self)
        vd = vDiff / dens
        dResult = self
        result = [dResult]
        while (not dResult == other):
            dResult += vd
            result.append(dResult)
        return result

    @property
    def length(self):
        """get the vector length / amplitude
            >>> v = Vector(0.0, 2.0, 1.0)
            >>> v.length
            2.2360679774997898
        """

        # iterate through the coordinates, square each, and return the root of
        # the sum
        return np.sqrt(sum(n**2 for n in self.coords))

    @length.setter
    def length(self, number):
        """set the vector amplitude
            >>> v = Vector(0.0, 2.0, 1.0)
            >>> v.length
            2.2360679774997898
            >>> v.length = -3.689
            >>> v
            Vector(-0.0, -3.2995419076, -1.6497709538)
        """

        # depends on normalized() and __mult__
        # create a vector as long as the number
        v = self.normalized() * number
        # copy it
        self.match(v)

    def match(self, other):
        """sets the vector to something, either another vector,
        a dictionary, or an iterable.
        If an iterable, it ignores everything
        beyond the first 3 items.
        If a dictionary, it only uses keys 'x','y', and 'z'
            >>> v
            Vector(0.0, 3.2995419076, 1.6497709538)
            >>> v.match({'x':2.0, 'y':1.0, 'z':2.2})
            >>> v
            Vector(2.0, 1.0, 2.2)
        """

        # this basically just makes a new vector and uses it's coordinates to
        # reset the coordinates of this one.
        if isinstance(other, Vector):
            self.coords = other.coords
        elif isinstance(other, dict):
            self.coords = (other['x'], other['y'], other['z'])
        else: # assume it is some other iterable
            self.coords = tuple(other[:3])

    def normalize(self):
        """edits vector in place to amplitude 1.0 and then returns self
            >>> v
            Vector(-0.0, -3.2995419076, -1.6497709538)
            >>> v.normalize()
            Vector(-0.0, -0.894427191, -0.4472135955)
            >>> v
            Vector(-0.0, -0.894427191, -0.4472135955)
        """

        # depends on normalized and match
        self.match(self.normalized())
        return self

    def normalized(self):
        """just returns the normalized version of self without editing self in
        place.
            >>> v.normalized()
            Vector(0.0, 0.894427191, 0.4472135955)
            >>> v
            Vector(0.0, 3.2995419076, 1.6497709538)
        """

        # think how important float accuracy is here!
        if isRoughlyZero(sum(n**2 for n in self)):
            raise ZeroDivisionError
        else:
            vn = self * (1 / self.length)
            vn.name = self.name + '_1'
            #self.coords1 = (self.origin[0]+vn[0], self.origin[1]+vn[1], self.origin[2]+vn[2])
            return vn

    def plot3D(self, ax, tip=False):
        #if ax.is_figure_set():
        # coords = np.array(list(zip(self.p0, self.p1)))
        # lgd(f"{coords = }")
        # ax.plot(*coords, ls=self.linestyle, lw=self.lw, c=self.color)
        # ax.text(*coords/2, self.name)
        # ax.scatter(*self.p1, marker='>', s=80)
        # self._plot_edge(self.p0, self.p1, ax)

        # ax.quiver(*self.p0, *self.p1, lw=3, color=self.color,
        #           arrow_length_ratio=0.1)
        ax.quiver(*self.p0, *self._coords*self.length, lw=3, color=self.color,
                  arrow_length_ratio=0.3)

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, point):
        self._p0 = point

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, point):
        self._p1 = point

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def linestyle(self):
        return self._linestyle

    @linestyle.setter
    def linestyle(self, linestyle):
        self._linestyle = linestyle

    @property
    def lw(self):
        return self._lw

    @lw.setter
    def lw(self, lw):
        self._lw = lw

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    def rotatexy(self, ang):
        """ Rotating self by the ang

        Rotation of the vector in xy-plane (2D).

        Args:
            ang: float - angle of rotation.

        Returns:
            Vector - rotated self.
        """

        p1 = self.p1.rotatexy(angle=ang)

        return Vector(self.p0, p1, name=f"{self.name}_rot{ang}")

    @property
    def vs_nrt(self):
        return self.__class__(self.vs_nrt)

    @vs_nrt.setter
    def vs_nrt(self, value=None):
        err = "vs_nrt cannot be set to a new value"
        raise AttributeError(err)

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        try:
            return self.coords[2]
        except IndexError:
            # 2D case
            return None


def demo():
    print(f"{Vector(1, 2, 3) = }")
    print(f"{Vector(Point(1, 2, 3)) = }")
    v = Vector(Point(1, 2, 3))
    print(f"{v.coords = }")
    u = Vector((0, 0, 0), (1, 2, 3), name="u")
    print(f"{u = }")
    print(f"{Vector((0, 0, 0), (1, 2, 3), name='u') = }")


def main():
    demo()


if __name__ == "__main__":
    main()
