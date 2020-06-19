"""Test akeyaa.py"""
import random
import numpy as np

import akeyaa

__author__ = "Randal J Barnes"
__version__ = "18 June 2020"


# -----------------------------------------------------------------------------
def test_analyze():
    # TODO: add test code
    pass


# -----------------------------------------------------------------------------
def test_layout_the_targets():
    # TODO: add test code
    pass


# -----------------------------------------------------------------------------
def test_fit_conic_potential():

    # Create a test case
    n = 100
    xmax = 1000
    ymax = 1000

    xtarget = xmax/2
    ytarget = ymax/2

    x = xmax * np.random.random(n)
    y = ymax * np.random.random(n)

    A = 0.001
    B = -0.001
    C = 0.001
    D = 0.10
    E = -0.10
    F = 1000

    dx = x - xtarget
    dy = y - ytarget

    z = (A*dx**2 + B*dy**2 + C*dx*dy + D*dx + E*dy + F) / 0.3048

    xyz = []
    for i in range(n):
        xyz.append((x[i], y[i], z[i]))

    evp, varp = akeyaa.fit_conic_potential((xtarget, ytarget), xyz)

    assert np.allclose(evp, np.array([A, B, C, D, E, F]))


# -----------------------------------------------------------------------------
def test_collate_results():
    # TODO: add test code
    pass