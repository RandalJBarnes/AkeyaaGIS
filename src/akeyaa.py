"""AkeyaaGIS entry point."""
import math
import numpy as np
import scipy
import statsmodels.api as sm
import arcpy
import pnorm


__author__ = "Randal J Barnes"
__version__ = "19 June 2020"


# -----------------------------------------------------------------------------
def analyze(polygon, welldata, radius, required, spacing):
    """Compute the Akeyaa analysis at locations across the specified polygon.

    Parameters
    ----------
    polygon : arcpy.Polygon

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
        Grid spacing for target locations across the polygon. The grid is
        square, so only one `spacing` is needed. spacing >= 1.

    Returns
    -------
    structured numpy array : one row for each target location

        ("x", np.float),        Easting (NAD 83 UTM zone 15N) [m]
        ("y", np.float),        Northing (NAD 83 UTM zone 15N) [m]
        ("count", np.int),      Number of neighbors [#]
        ("head", np.float),     Local piezometric head [ft]
        ("ux", np.float),       x-component of flow unit vector [.]
        ("uy", np.float),       y-component of flow unit vector [.]
        ("p10", np.float),      pr(theta within +/- 10 degrees) [.]
        ("grad", np.float),     Magnitude of the head gradient [.]
        ("score", np.float)     Laplacian z-score [.]

    See Also
    --------
    pnorm.py

    """
    tree = scipy.spatial.cKDTree([(row[0], row[1]) for row in welldata])

    results = []
    for xytarget in layout_the_targets(polygon, spacing):

        xyz = []
        for i in tree.query_ball_point(xytarget, radius):
            xyz.append(welldata[i])

        if len(xyz) >= required:
            evp, varp = fit_conic_potential(xytarget, xyz)
            results.append((xytarget, len(xyz), evp, varp))

    return collate_results(results)


# -----------------------------------------------------------------------------
def layout_the_targets(polygon, spacing):
    """Determine the evenly-spaced locations of the x and y grid lines.

    The grid lines of target locations are anchored at the centroid of the
    `polygon`, axes-aligned, and the separated by `spacing`. The outer extent
    of the grid captures all of the vertices of the `polygon`.

    The grid nodes are then filtered so that only nodes inside of the polygon
    are retained.

    Parameters
    ----------
    polygon : a concrete instance of a geometry.Shape.

    spacing : float
        Grid spacing for target locations across the polygon. The grid is
        square, so only one `spacing` is needed.

    Returns
    -------
    targets : list[tuple] (xtarget, ytarget)
        x- and y-coordinates of the target points.

    """
    xgrd = [polygon.centroid.X]
    while xgrd[-1] > polygon.extent.XMin:
        xgrd.append(xgrd[-1] - spacing)
    xgrd.reverse()
    while xgrd[-1] < polygon.extent.XMax:
        xgrd.append(xgrd[-1] + spacing)

    ygrd = [polygon.centroid.Y]
    while ygrd[-1] > polygon.extent.YMin:
        ygrd.append(ygrd[-1] - spacing)
    ygrd.reverse()
    while ygrd[-1] < polygon.extent.YMax:
        ygrd.append(ygrd[-1] + spacing)

    xygrd = []
    for x in xgrd:
        for y in ygrd:
            if polygon.contains(arcpy.Point(x, y)):
                xygrd.append((x, y))

    return xygrd


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
    structured numpy array : one row for each target location

        ("x", np.float),        Easting (NAD 83 UTM zone 15N) [m]
        ("y", np.float),        Northing (NAD 83 UTM zone 15N) [m]
        ("count", np.int),      Number of neighbors [#]
        ("head", np.float),     Local piezometric head [ft]
        ("ux", np.float),       x-component of flow unit vector [.]
        ("uy", np.float),       y-component of flow unit vector [.]
        ("p10", np.float),      pr(theta within +/- 10 degrees) [.]
        ("grad", np.float),     Magnitude of the head gradient [.]
        ("score", np.float)     Laplacian z-score [.]

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
        x = row[0][0]
        y = row[0][1]

        count = row[1]

        evp = row[2]
        varp = row[3]

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

        laplacian = 2*(evp[0]+evp[1])           # Laplacian, not recharge.
        stdev = 2*np.sqrt(varp[0, 0] + varp[1, 1] + 2*varp[0, 1])
        score = min(max(laplacian/stdev, -3), 3)

        output_array[i] = (x, y, count, head, ux, uy, p10, grad, score)

    return output_array
