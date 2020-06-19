"""Test pnorm.py"""
import numpy as np

import pnorm

__author__ = "Randal J Barnes"
__version__ = "18 June 2020"


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


def test_pnorm_cdf():
    """Test pnormcdf by comparing with MATLAB results."""
    mu = np.array([[1.], [2.]])
    sigma = np.array([[2., 1.], [1., 3.]])

    lowerbound = np.pi/4
    upperbound = np.pi/2
    cdf = pnorm.cdf(lowerbound, upperbound, mu, sigma)

    cdf_ans = np.array([0.5066762601816892])
    assert np.allclose(cdf, cdf_ans)
