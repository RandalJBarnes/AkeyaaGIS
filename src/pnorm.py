"""Support for the general projected normal distribution.

pdf(theta, mu, sigma):
    Evaluate the probability density function for the general projected
    normal distribution.

cdf(lb, ub, mu, sigma):
    Evaluate the Pr(lb < theta < ub) for a general projected normal
    distribution.

Notes
-----
*  The variance/covariance matrix, sigma, must be positive definite.

*  The general projected normal distribution is a 2D circular distribution.
    The domain is [0, 2pi]. See, for example, Lark [2014].

*  See Justus [1978, (4-11)] or Hernandez et al. [2017, (1)] for details on
   the general projected normal distribution pdf.

References
----------
*  D. Hernandez-Stumpfhauser, F. J. Breidt, and M. J. van der Woerd.
   The General Projected Normal Distribution of Arbitrary Dimension:
   Modeling and Bayesian Inference Bayesian Analysis. Institute of
   Mathematical Statistics, 12:113-133, 2017.

*  C. G. Justus. Winds and Wind System Performance. Solar energy.
   Franklin Institute Press, Philadelphia, Pennsylvania, 1978. ISBN
   9780891680062. 120 pp.

*  R. M. Lark, D. Clifford, and C. N. Waters. Modelling complex geological
   circular data with the projected normal distribution and mixtures of
   von Mises distributions. Solid Earth, Copernicus GmbH, 5:631-639, 2014.

"""
from math import cos, sin, exp, sqrt, pi
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


__author__ = "Randal J Barnes"
__version__ = "18 June 2020"


# -----------------------------------------------------------------------------
def pdf(angles, mu, sigma):
    """General projected normal distribution PDF.

    Evaluate probability density function for the general projected normal
    distribution.

    Parameters
    ----------
    angles : ndarray, shape(M, ), or a list, or scalar.
        The angles at which to evaluate the pdf. The angles are given in
        radians, not degrees.

    mu : ndarray, shape=(2, 1)
        The mean vector.

    sigma : ndarray, shape=(2, 2)
        The variance matrix. This matrix positive definite.

    Returns
    -------
    ndarray, shape (M, )
        The array of pdf values at each of the angles specified in `alpha`.

    Notes
    -----
    This implementation is based on Hernandez et al. [2017] Equation (1).
    However, the exact representation given by Hernandez et al. is prone
    to numerical overflow. To ameliorate the problem we have refactored
    the exponential components of the equation for extreme cases..

    """
    if isinstance(angles, np.ndarray):
        values = np.empty(angles.shape[0])
    elif isinstance(angles, list):
        values = np.empty(len(angles))
    else:
        angles = [angles]
        values = np.empty([1,])

    # Manually compute the det and inv of the 2x2 matrix.
    detS = sigma[0, 0] * sigma[1, 1] - sigma[0, 1] * sigma[1, 0]
    Sinv = np.array([[sigma[1, 1], -sigma[0, 1]], [-sigma[1, 0], sigma[0, 0]]]) / detS

    C = mu.T @ Sinv @ mu
    D = 2 * pi * sqrt(detS)

    for j, theta in enumerate(angles):
        r = np.array([[cos(theta)], [sin(theta)]])
        A = r.T @ Sinv @ r
        B = r.T @ Sinv @ mu
        E = B / sqrt(A)

        # Note: this will still overflow for (E*E - C) > 700, or so.
        if E < 5:
            values[j] = exp(-C / 2) * (1 + E * norm.cdf(E) / norm.pdf(E)) / (A * D)
        else:
            values[j] = E * sqrt(2 * pi) * exp((E * E - C) / 2) / (A * D)

    return values


# -----------------------------------------------------------------------------
def cdf(lowerbound, upperbound, mu, sigma):
    """General projected normal distribution CDF.

    Evaluate the Pr(lb < theta < ub) for a general projected normal
    distribution.

    Parameters
    ----------
    lowerbound : float
        lower integration bound on the angular range. lb < ub.

    upperbound : float
        upper integration bound on the angular range. ub > lb.

    mu : ndarray, shape=(2, 1)
        The mean vector.

    sigma : ndarray, shape=(2, 2)
        The variance matrix.

    Returns
    -------
    float
        Pr(lowerbound < alpha < upperbound)

    """
    try:
        value = quad(lambda theta: pdf(theta, mu, sigma), lowerbound, upperbound)[0]
    except OverflowError:
        value = 1.0
    except ValueError:
        value = 1.0
    except:
        raise

    return value
