#!/usr/bin/python3
# -*- coding: utf-8 -*-

# minihelio.py

"""
Minihelio module

author: khaz
date@time: 2024-05-09@17:21:53
version: 0.0.1
"""

import numpy as np

ftitle = __file__.split("/", maxsplit=-1)[-1].split(".", maxsplit=-1)[0]

D2R = np.float64(np.pi/180)
mtDaysSum = {1: 0, 2: 31, 3: 59,
             4: 90, 5: 120, 6: 151,
             7: 181, 8: 212, 9: 243,
             10: 273, 11: 304, 12: 334}


def alt(DEC: float, LAT: float, HRA: float, dtype=np.float32) -> float:
    """Solar altitude angle ALT = alt(DEC, LAT, HRA) (degrees)

    Args:
        DEC: float - solar declination angle.
        LAT: float - geographical latitude of location.
        HRA: float - hour angle.
        dtype: type - type of the return value, default np.float32.

    Returns:
        float - solar azimuth angle.
    """

    DEC = np.asarray(DEC, dtype=dtype)
    LAT = np.asarray(LAT, dtype=dtype)
    HRA = np.asarray(HRA, dtype=dtype)
    return np.arcsin(np.cos(LAT*D2R)*np.cos(DEC*D2R)*np.cos(HRA*D2R)
                     + np.sin(LAT*D2R)*np.sin(DEC*D2R))/D2R


def azi(DEC, LAT, HRA, dtype=np.float32):
    """Solar azimuth angle AZI = azi(DEC, LAT, HRA) (in degrees)"""

    DEC = np.asarray(DEC, dtype=dtype)
    LAT = np.asarray(LAT, dtype=dtype)
    HRA = np.asarray(HRA, dtype=dtype)
    ZEN = zen(DEC, LAT, HRA)

    gs = np.sign(HRA)*(np.arccos((np.cos(ZEN*D2R)*np.sin(LAT*D2R)
                               - np.sin(DEC*D2R)) /
                       (np.sin(ZEN*D2R)*np.cos(LAT*D2R)))/D2R)
    if isinstance(gs, complex):
        return gs.real + 180
    else:
        return gs+180


def zen(DEC, LAT, HRA):
    """Solar zenith angle ZEN = zen(DEC, LAT, HRA) (degrees)

    Args:
        DEC: float - solar declination angle.
        LAT: float - geographical latitude of location.
        HRA: float - hour angle.

    Returns:
        float - solar zenith angle.
    """
    return np.arccos(np.cos(LAT*D2R)*np.cos(DEC*D2R)*np.cos(HRA*D2R) +
                     np.sin(LAT*D2R)*np.sin(DEC*D2R))/D2R


def dayNr(*args):
    """Compute number of the day in a common year.

    Args:
        m: int - number of the month.
        d: int - number of the day.
        Or
        (m, d) (tuple(int, int)).

    Returns:
        int - the number of a day in the year.

    Ref.
        Pluta 2006
    """

    if (len(args) == 2):
        mm = args[0]
        dd = args[1]
    elif len(args) == 1 and isinstance(args[0], tuple):
        mm = args[0][0]
        dd = args[0][1]
    elif isinstance(args[0], np.ndarray) or isinstance(args[0], list):
        results = np.array([], dtype=int)
        for tp in args[0]:
            results = np.append(results, oneDayNr(tp[0], tp[1]))
        return results
    else:
        raise TypeError("Wrong date format (should be (m, d) "
                        f"not {args}).")
    return oneDayNr(mm, dd)


def oneDayNr(m, d):
    """'Internal' method, returns number of the day in a year
    for a single day and common year.
    """

    return d + mtDaysSum[m]


def dec(dn):
    """Solar declination DEC = dec(dn) (degrees).

    Args:
        dn: int - number of the day in a year.

    Returns:
        float - value of the solar declination angle.
    """

    if not isinstance(dn, np.ndarray):
        dn = np.array(dn)

    return 23.45 * np.sin((360.0 * (dn + 284.0)/365.0)*D2R)


def hra(h):
    """The hour angle function,

	HRA = hra(tau): HRA - the hour angle, tau - the hour (0, 24].
    """

    if (hasattr(h, '__iter__')):
        return np.array([hra(t) for t in h])
    else:
        dt = 0 if h >= 0 else 24  # correction for h < 0
        return 15*((dt + h) - 12.00)  # degrees
