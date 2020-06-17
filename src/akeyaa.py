"""AkeyaaGIS entry point."""

from itertools import compress
import math

import numpy as np
import scipy
import statsmodels.api as sm

import pnorm


# -----------------------------------------------------------------------------
def analyze(venue, welldata, radius, required, spacing):
    """Compute the Akeyaa analysis at locations across the specified venue.

    Parameters
    ----------
    venue : a concrete instance of a geometry.Shape, e.g. Polygon. We are
        using the following methods: centroid, extent, and contains_points.

    welldata : array, shape=(n, 3), dtype=float
        well data: x- and y- locations [m], and the measured static water
        level [ft]. A well may have more than one entry, if it has more than
        one measured static water level.

    radius : float
        Search radius for neighboring wells. radius >= 1.

    required : int
        Required number of neighboring wells. If fewer are found, the
        target location is skipped. required >= 6.

    spacing : float
        Grid spacing for target locations across the venue. The grid is
        square, so only one `spacing` is needed. spacing >= 1.

    Returns
    -------
    structured array
        ("x", np.float),
        ("y", np.float),
        ("count", np.int),
        ("head", np.float),
        ("ux", np.float),
        ("uy", np.float),
        ("p10", np.float),
        ("grad", np.float),
        ("score", np.float)

    See Also
    --------
    geometry.py, pnorm.py

    """
    tree = scipy.spatial.cKDTree([(row[0], row[1]) for row in welldata])
    targets = layout_the_targets(venue, spacing)

    results = []
    for xytarget in targets:

        xyz = []
        indx = tree.query_ball_point(xytarget, radius)
        if indx:
            for i in indx:
                xyz.append(welldata[i])

        if len(xyz) >= required:
            evp, varp = fit_conic_potential(xytarget, xyz)
            results.append((xytarget, len(xyz), evp, varp))

    return collate_results(results)


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
    venue : a concrete instance of a geometry.Shape.

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

    list[tuple] : (x, y, z)

        x : The x-coordinates in "NAD 83 UTM 15N" (EPSG:26915) [m].

        y : The y-coordinates in "NAD 83 UTM 15N" (EPSG:26915) [m].

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
    *   The local conic potential model is computed using a robust linear
        model which is fit using iteratively reweighted least squares with
        Tukey biweights.

    *   The underlying conic potential model is

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

    Parameters
    ----------
    results : list[tuple] (xytarget, n, evp, varp)

        xytarget : tuple (float, float)
            x- and y-coordinates of target location.
        n : int
            number of naerby wells used in the local analysis.
        evp : (6, 1) ndarray
            expected value vector of the model parameters.
        varp : (6, 6) ndarray
            variance/covariance matrix of the model parameters.

    Returns
    -------
    structured array
        ("x", np.float),
        ("y", np.float),
        ("count", np.int),
        ("head", np.float),
        ("ux", np.float),
        ("uy", np.float),
        ("p10", np.float),
        ("grad", np.float),
        ("score", np.float)

    """
    output_array = np.empty(
        len(results),
        dtype=[
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
        evp = row[2]                            # expected value vector
        varp = row[3]                           # variance/covariance matrix

        x = row[0][0]                           # NAD 83 UTM zone 15N
        y = row[0][1]

        count = row[1]                          # number of neighbors

        head = 3.28084 * evp[5]                 # convert [m] to [ft].

        mu = evp[3:5]
        sigma = varp[3:5, 3:5]
        ux = -mu[0] / np.hypot(mu[0], mu[1])    # x-component of unit vector
        uy = -mu[1] / np.hypot(mu[0], mu[1])    # y-component of unit vector

        theta = math.atan2(mu[1], mu[0])        # angle <from>, not angle <to>.
        lowerbound = theta - np.pi / 18.0       # +/- 10 degrees.
        upperbound = theta + np.pi / 18.0
        p10 = pnorm.cdf(lowerbound, upperbound, mu, sigma)

        grad = np.hypot(mu[0], mu[1])           # magnitude of the head gradient

        laplacian = 2*(evp[0]+evp[1])
        stdev = 2*np.sqrt(varp[0, 0] + varp[1, 1] + 2*varp[0, 1])
        score = min(max(laplacian/stdev, -3), 3)

        output_array[i] = (x, y, count, head, ux, uy, p10, grad, score)

    return output_array
