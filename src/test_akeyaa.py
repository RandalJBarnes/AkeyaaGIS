"""Test akeyaa.py"""
import random
import numpy as np

import akeyaa

__author__ = "Randal J Barnes"
__version__ = "24 June 2020"


# -----------------------------------------------------------------------------
def test_analyze():
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
def test_compute_features():
    # TODO: add test code
    pass


# -----------------------------------------------------------------------------
def test_layout_the_grid():
    # TODO: add test code
    pass


# -----------------------------------------------------------------------------
def test_initialize_rasters():
    # TODO: add test code
    pass


# -----------------------------------------------------------------------------
def test_pnorm_pdf():
    """Test pnormpdf by comparing with MATLAB results."""
    mu = np.array([[1.], [2.]])
    sigma = np.array([[2., 1.], [1., 3.]])

    alpha = np.linspace(0, 2*np.pi, 10)
    pdf = pnorm.pdf(alpha, mu, sigma)

    pdf_ans = np.array([
        0.082633326709771, 0.453847423028614, 0.538084798525049,
        0.117852523781454, 0.051437060901120, 0.040453966191302,
        0.037600718897055, 0.030693276068009, 0.035604961954848,
        0.082633326709771
        ])

    assert np.allclose(pdf, pdf_ans)

# -----------------------------------------------------------------------------
def test_pnorm_cdf():
    """Test pnormcdf by comparing with MATLAB results."""
    mu = np.array([[1.], [2.]])
    sigma = np.array([[2., 1.], [1., 3.]])

    lowerbound = np.pi/4
    upperbound = np.pi/2
    cdf = pnorm.cdf(lowerbound, upperbound, mu, sigma)

    cdf_ans = np.array([0.5066762601816892])
    assert np.allclose(cdf, cdf_ans)
