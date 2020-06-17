"""AkeyaaGIS entry point."""

from itertools import compress
import math

import numpy as np
import scipy
import statsmodels.api as sm

import pnorm


# -----------------------------------------------------------------------------
def akeyaa(polygon, xyz, radius, required, spacing):
    """Fit the conic discharge potential model.


    """
    results = analyze(polygon, xyz, radius, required, spacing)
    output_array = collate_results(results)

    return output_array


# -----------------------------------------------------------------------------
def analyze(venue, xyz, radius, required, spacing):
    """Compute the Akeyaa analysis across the specified venue.

    Parameters
    ----------
    venue: type


    Returns
    -------
    results : list[tuple] (xytarget, n, evp, varp)

        xytarget : tuple (float, float)
            x- and y-coordinates of target location.
        n : int
            number of naerby wells used in the local analysis.
        evp : (6, 1) ndarray
            expected value vector of the model parameters.
        varp : (6, 6) ndarray
            variance/covariance matrix of the model parameters.

    """
    tree = scipy.spatial.cKDTree([(row[0], row[1]) for row in xyz])
    targets = layout_the_targets(venue, spacing)

    results = []
    for xytarget in targets:

        wells = []
        indx = tree.query_ball_point(xytarget, radius)
        if indx:
            for i in indx:
                wells.append(xyz[i])

        if len(wells) >= required:
            evp, varp = fit_conic_potential(xytarget, wells)
            results.append((xytarget, len(wells), evp, varp))

    return results


# -----------------------------------------------------------------------------
def layout_the_targets(venue, spacing):
    """Determine the evenly-spaced locations of the x and y grid lines.

    The grid lines of target locations are anchored at the centroid of the
    `venue`, axes-aligned, and the separated by `spacing`. The outer extent
    of the grid captures all of the vertices of the `venue`.

    The grid nodes are then filtered so that only nodes inside of the venue
    are retained.

    Parameters
    ----------
    venue: polygon

    spacing : float
        Grid spacing for target locations across the venue. The grid is
        square, so only one `spacing` is needed.

    Returns
    -------
    targets : list[tuple] (xtarget, ytarget)
        x- and y-coordinates of the target points.


    """
    xgrd = [venue.centroid()[0]]
    while xgrd[-1] > venue.extent()[0]:
        xgrd.append(xgrd[-1] - spacing)
    xgrd.reverse()
    while xgrd[-1] < venue.extent()[1]:
        xgrd.append(xgrd[-1] + spacing)

    ygrd = [venue.centroid()[1]]
    while ygrd[-1] > venue.extent()[2]:
        ygrd.append(ygrd[-1] - spacing)
    ygrd.reverse()
    while ygrd[-1] < venue.extent()[3]:
        ygrd.append(ygrd[-1] + spacing)

    xygrd = []
    for x in xgrd:
        for y in ygrd:
            xygrd.append((x, y))
    flag = venue.contains_points(xygrd)

    return list(compress(xygrd, flag))


# -----------------------------------------------------------------------------
def fit_conic_potential(xytarget, xyz):
    """Fit the local conic potential model to the selected heads.

    Parameters
    ----------
    xytarget : tuple (xtarget, ytarget)
        The x- and y-coordinates in "NAD 83 UTM 15N" (EPSG:26915) [m] of
        the target location.

    list[tuple] : ((x, y), z)

        (x, y) : tuple(float, float)
            The x- and y-coordinates in "NAD 83 UTM 15N" (EPSG:26915) [m].
        z : float
            The recorded static water level [ft]

    Returns
    -------
    evp : (6,) ndarray
        The expected value vector for the fitted model parameters.

    varp : (6, 6) ndarray
        The variance/covariance matrix for the fitted model parameters.

    See Also
    --------
    statsmodels.RLM

    Notes
    -----
    The underlying conic potential model is

        z = Ax^2 + By^2 + Cxy + Dx + Ey + F + noise

    where the fitted parameters map as: [A, B, C, D, E, F] = p[0:5].

    """
    x = np.array([row[0] for row in xyz], dtype=float) - xytarget[0]
    y = np.array([row[1] for row in xyz], dtype=float) - xytarget[1]
    z = np.array([row[2] for row in xyz], dtype=float) * 0.3048         # [ft] to [m].

    exog = np.stack([x**2, y**2, x*y, x, y, np.ones(x.shape)], axis=1)

    method_norm = sm.robust.norms.TukeyBiweight()
    rlm_model = sm.RLM(z, exog, method_norm)
    rlm_results = rlm_model.fit()
    evp = rlm_results.params
    varp = rlm_results.bcov_scaled

    return (evp, varp)


# -----------------------------------------------------------------------------
def collate_results(results):
    """Collate the interpreted results.

    Returns
    -------
    structured array

    Notes
    -----
    The underlying conic potential model is

        z = Ax^2 + By^2 + Cxy + Dx + Ey + F + noise

    where the fitted parameters map as: [A, B, C, D, E, F] = p[0:5].

    """

    output_array = np.empty(len(results),
        dtype = [
            ("x", np.float),
            ("y", np.float),
            ("count", np.int),
            ("head", np.float),
            ("ux", np.float),
            ("uy", np.float),
            ("p10", np.float),
            ("grad", np.float),
            ("score", np.float)
        ]
    )

    for i, row in enumerate(results):
        evp = row[2]
        varp = row[3]

        x = row[0][0]
        y = row[0][1]

        count = row[1]

        head = 3.28084 * evp[5]                 # convert [m] to [ft].

        mu = evp[3:5]
        sigma = varp[3:5, 3:5]
        ux = -mu[0] / np.hypot(mu[0], mu[1])
        uy = -mu[1] / np.hypot(mu[0], mu[1])

        theta = math.atan2(mu[1], mu[0])        # angle <from>, not angle <to>.
        lowerbound = theta - np.pi / 18.0       # +/- 10 degrees.
        upperbound = theta + np.pi / 18.0
        p10 = pnorm.cdf(lowerbound, upperbound, mu, sigma)

        grad = np.hypot(mu[0], mu[1])

        laplacian = 2*(evp[0]+evp[1])
        stdev = 2*np.sqrt(varp[0, 0] + varp[1, 1] + 2*varp[0, 1])
        score = min(max(laplacian/stdev, -3), 3)

        output_array[i] = (x, y, count, head, ux, uy, p10, grad, score)

    return output_array
