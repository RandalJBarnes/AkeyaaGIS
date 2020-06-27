"""AkeyaaGIS"""
import math
import sys
import numpy as np
import scipy
import statsmodels.api as sm
import arcpy


__version__ = "27 June 2020"


# -----------------------------------------------------------------------------
def run_akeyaa(polygon, welldata, radius, required, spacing, base_filename):
    """Carries out an Akeyaa analysis over the specified polygon.

    In addition to the returned output array, this function creates a suite of
    output files. All of the output files share a common ``base_filename``,
    which is passed in as an argument.

    The base_filename can (should) include the necessary path information. This
    means that all of the files created by this function are pout into a single
    common folder (directory).

    The feature class files are created by arcpy.da.NumPyArrayToFeatureClass.
    The associated filenames start with the base_filename, have a suffix "_fc"
    attached, and end with the ArcGIS-assigned file extension.  These include:

        base_filename_fc.cpg
        base_filename_fc.dbf
        base_filename_fc.prj
        base_filename_fc.shp
        base_filename_fc.shx

    ESRI GridFloat files are created for each of the feature rasters. There are
    two files created for each feature: an ASCII .hdr file and a binary .flt
    file.

    Parameters
    ----------
    polygon : arcpy.Polygon
        The Akeyaa analysis is carried out at target locations within the
        polygon. The target locations are selected as the nodes of a square
        grid covering the polygon.

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

    base_filename : str
        Path and filename prefix for the feature class and ESRI GridFloat files.

    Returns
    -------
    output_array : numpy structured array
        ("x", np.float),        Easting (NAD 83 UTM zone 15N) [m]
        ("y", np.float),        Northing (NAD 83 UTM zone 15N) [m]
        ("count", np.int),      Number of neighbors [#]
        ("head", np.float),     Local piezometric head [ft]
        ("ux", np.float),       x-component of flow unit vector [.]
        ("uy", np.float),       y-component of flow unit vector [.]
        ("p10", np.float),      pr(theta within +/- 10 degrees) [.]
        ("grad", np.float),     Magnitude of the head gradient [.]
        ("score", np.float)     Laplacian z-score [.]

    Notes
    -------
    *   This module requires:
            arcgispro >= 2.5  (and all that this entails)
            statsmodels >= 0.11.1

    *   output feature class:
            "x"         Easting (NAD 83 UTM zone 15N) [m]
            "y"         Northing (NAD 83 UTM zone 15N) [m]
            "count"     Number of neighbors [#]
            "head"      Local piezometric head [ft]
            "ux"        x-component of flow unit vector [.]
            "uy"        y-component of flow unit vector [.]
            "p10"       pr(theta within +/- 10 degrees) [.]
            "grad"      Magnitude of the head gradient [.]
            "score"     Laplacian z-score [.]

    Notes
    -----
    *   The feature class and .flt files are created using the NAD 83 UTM
        zone 15N (EPSG:26915) projected coordinate system. This spatial
        reference information is encoded in the feature class files, but it is
        not encoded in the ESRI GridFloat files.

    *   The ESRI GridFloat files use a 32-bit floating point format for the
        gridded data, not the IEEE 754 format sued throughout python 3. As such,
        NoDataValues for the ESRI GridFloat files is np.finfo(np.float32).max
        rather than np.nan.

    *   If we could figure out how to correctly and completely construct, write,
        and read binary arcpy.Raster files then numerous ESRI GridFloat files
        would be unnecessary.

    """
    xgrd, ygrd, output_list, index_list = analyze(polygon, welldata, radius, required, spacing)
    spatial_reference = arcpy.SpatialReference(26915)  # NAD 83 UTM zone 15N (EPSG:26915).

    # Create the feature class output and save it to disk.
    akeyaa_array = np.array(
        output_list,
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
    arcpy.da.NumPyArrayToFeatureClass(
        akeyaa_array,
        base_filename + "_fc",
        ("x", "y"),
        spatial_reference
    )

    # Create the ESRI GridFloat files for the feature rasters.
    features = ["count", "head", "ux", "uy", "p10", "grad", "score"]

    grid = np.empty((len(ygrd), len(xgrd)), dtype=np.float32)
    missing = np.finfo(np.float32).max
    if sys.byteorder == 'little':
        byteorder = "LSBFIRST"
    elif sys.byteorder == 'big':
        byteorder = "MSBFIRST"
    else:
        raise TypeError

    for band, name in enumerate(features):
        with open(base_filename + "_" + name + ".hdr", "w") as fid:
            fid.write(f"NCOLS {len(xgrd)} \n")
            fid.write(f"NROWS {len(ygrd)} \n")
            fid.write(f"XLLCORNER {min(xgrd)} \n")
            fid.write(f"YLLCORNER {min(ygrd)} \n")
            fid.write(f"CELLSIZE {spacing} \n")
            fid.write(f"NODATA_VALUE {missing} \n")
            fid.write(f"BYTEORDER {byteorder}")

        grid[:] = missing
        for index, output_row in zip(index_list, output_list):
            grid[index[0], index[1]] = output_row[band + 2]

        with open(base_filename + "_" + name + ".flt", "wb") as fid:
            grid.tofile(fid)

    return akeyaa_array


# -----------------------------------------------------------------------------
def analyze(polygon, welldata, radius, required, spacing):
    """Compute the AkeyaaGIS features at grid nodes across the specified polygon.

    There are seven AkeyaaGIS features: count, head, ux, uy, p10, grad, score.

    The Akeyaa analysis is carried out at target locations within a polygon.
    The target locations are selected as the nodes of a square grid covering the
    polygon.

    The square grid of target locations is anchored at the centroid of the
    polygon, and the grid lines are separated by `spacing`. If a target location
    is not inside of ``polygon`` it is ignored.

    For each target location inside the polygon, all ``welldata'' (wells)
    within a  horizontal distance of ``radius`` of the target location are
    identified. If a target location has fewer than ``required`` identified
    (neighboring) wells it is ignored.

    Wells from outside of the polygon may also be used in the computations.

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
    xgrd : List[float]
        x-grid values (columns) in increasing order.

    ygrd : List[float]
        y-grid values (rows) in decreasing order.

    output_list : List[(x, y, len(xyz), head, ux, uy, p10, grad, score)]

        x : float
            target location easting in "NAD 83 UTM 15N" (EPSG:26915) [m].

        y : float
            target location northing in "NAD 83 UTM 15N" (EPSG:26915) [m].

        count : int
            Number of neighbors [#].

        head : float
            Local piezometric head [ft].

        ux : float
            x-component of flow direction unit vector [.].

        uy : float
            y-component of flow direction unit vector [.].

        p10 : float
            pr(theta within +/- 10 degrees) [.].

        grad : float
            Magnitude of the head gradient [.].

        score : float
            Laplacian z-score [.].

    index_list : List[(i, j)]
        i : int
            row index of the target grid [#].

        j : int
            column index of the target grid [#].

    """
    tree = scipy.spatial.cKDTree([(row[0], row[1]) for row in welldata])
    xgrd, ygrd = layout_the_grid(polygon, spacing)

    output_list = []
    index_list = []
    for i, y in enumerate(ygrd):
        for j, x in enumerate(xgrd):
            if polygon.contains(arcpy.Point(x, y)):
                xytarget = (x, y)

                xyz = []
                for k in tree.query_ball_point(xytarget, radius):
                    xyz.append(welldata[k])

                if len(xyz) >= required:
                    evp, varp = fit_conic_potential(xytarget, xyz)
                    head, ux, uy, p10, grad, score = compute_features(evp, varp)
                    output_list.append((x, y, len(xyz), head, ux, uy, p10, grad, score))
                    index_list.append((i, j))

    return xgrd, ygrd, output_list, index_list


# -----------------------------------------------------------------------------
def layout_the_grid(polygon, spacing):
    """Determine the evenly-spaced locations of the x and y grid lines.

    The grid lines of target locations are anchored at the centroid of the
    `polygon`, axes-aligned, and the separated by `spacing`. The outer extent
    of the grid captures all of the vertices of the `polygon`.

    Parameters
    ----------
    polygon : a concrete instance of a geometry.Shape.

    spacing : float
        Grid spacing for target locations across the polygon. The grid is
        square, so only one `spacing` is needed.

    Returns
    -------
    xgrd : List[float]
        x-grid values (columns) in _IN_creasing order.

    ygrd : List[float]
        y-grid values (rows) in _DE_creasing order.

    """
    xgrd = [math.floor(polygon.centroid.X)]
    while xgrd[-1] > polygon.extent.XMin:
        xgrd.append(xgrd[-1] - spacing)
    xgrd.reverse()
    while xgrd[-1] < polygon.extent.XMax:
        xgrd.append(xgrd[-1] + spacing)

    ygrd = [math.floor(polygon.centroid.Y)]
    while ygrd[-1] < polygon.extent.YMax:
        ygrd.append(ygrd[-1] + spacing)
    ygrd.reverse()
    while ygrd[-1] > polygon.extent.YMin:
        ygrd.append(ygrd[-1] - spacing)

    return (xgrd, ygrd)


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
    z = np.array([row[2] for row in xyz], dtype=float) * 0.3048     # [ft] to [m].

    exog = np.stack([x**2, y**2, x*y, x, y, np.ones(x.shape)], axis=1)

    method_norm = sm.robust.norms.TukeyBiweight()
    rlm_model = sm.RLM(z, exog, method_norm)
    rlm_results = rlm_model.fit()
    evp = rlm_results.params
    varp = rlm_results.bcov_scaled

    return (evp, varp)

# -----------------------------------------------------------------------------
def compute_features(evp, varp):
    """Compute the interpreted features.

    Parameters
    ----------
    evp : (6, 1) ndarray
        expected value vector of the model parameters.

    varp : (6, 6) ndarray
        variance/covariance matrix of the model parameters.

    Returns
    -------
    head : float
        Local piezometric head [ft]

    ux : float
        x-component of flow unit vector [.]

    uy : float
        y-component of flow unit vector [.]

    p10 : float,
        pr(theta within +/- 10 degrees) [.]

    grad : float
        Magnitude of the head gradient [.]

    score : float
        Laplacian z-score [.]

    """
    head = 3.28084 * evp[5]                 # convert [m] to [ft].

    mu = evp[3:5]
    sigma = varp[3:5, 3:5]

    ux = -mu[0] / np.hypot(mu[0], mu[1])
    uy = -mu[1] / np.hypot(mu[0], mu[1])

    theta = math.atan2(mu[1], mu[0])        # angle <from>, not angle <to>.
    lowerbound = theta - math.pi / 18.0       # +/- 10 degrees.
    upperbound = theta + math.pi / 18.0
    p10 = pnormcdf(lowerbound, upperbound, mu, sigma)

    grad = np.hypot(mu[0], mu[1])

    laplacian = 2*(evp[0]+evp[1])           # Laplacian, not recharge.
    stdev = 2*math.sqrt(varp[0, 0] + varp[1, 1] + 2*varp[0, 1])
    score = min(max(laplacian/stdev, -3), 3)

    return (head, ux, uy, p10, grad, score)


# -----------------------------------------------------------------------------
def pnormpdf(angles, mu, sigma):
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
    *   The variance/covariance matrix, sigma, must be positive definite.

    *   The general projected normal distribution is a 2D circular distribution.
        The domain is [0, 2pi]. See, for example, Lark [2014].

    *   See Justus [1978, (4-11)] or Hernandez et al. [2017, (1)] for details
        on the general projected normal distribution pdf.

    *   This implementation is based on Hernandez et al. [2017] Equation (1).
        However, the exact representation given by Hernandez et al. is prone
        to numerical overflow. To ameliorate the problem we have refactored
        the exponential components of the equation for extreme cases..

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
    if isinstance(angles, np.ndarray):
        values = np.empty(angles.shape[0])
    elif isinstance(angles, list):
        values = np.empty(len(angles))
    else:
        angles = [angles]
        values = np.empty([1,])

    # Manually compute the det and inv of the 2x2 matrix.
    detS = sigma[0, 0] * sigma[1, 1] - sigma[0, 1] * sigma[1, 0]
    Sinv = (
        np.array([[sigma[1, 1], -sigma[0, 1]], [-sigma[1, 0], sigma[0, 0]]])
            / detS
    )

    C = mu.T @ Sinv @ mu
    D = 2 * math.pi * math.sqrt(detS)

    for j, theta in enumerate(angles):
        r = np.array([[math.cos(theta)], [math.sin(theta)]])
        A = r.T @ Sinv @ r
        B = r.T @ Sinv @ mu
        E = B / math.sqrt(A)

        # Note: this will still overflow for (E*E - C) > 700, or so.
        if E < 5:
            values[j] = (
                math.exp(-C / 2) * (1 + E * scipy.stats.norm.cdf(E) /
                scipy.stats.norm.pdf(E)) / (A * D)
            )
        else:
            values[j] = (
                E * math.sqrt(2 * math.pi) * math.exp((E * E - C) / 2) / (A * D)
            )

    return values


# -----------------------------------------------------------------------------
def pnormcdf(lowerbound, upperbound, mu, sigma):
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
        value = scipy.integrate.quad(lambda theta: pnormpdf(theta, mu, sigma), lowerbound, upperbound)[0]
    except OverflowError:
        value = 1.0
    except ValueError:
        value = 1.0
    except:
        raise

    return value
