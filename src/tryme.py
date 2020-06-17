import numpy as np

import arcpy

import akeyaa
import geometry


def main():
    out_table = r"D:\Google Drive\Projects\AkeyaaGIS\data\DakotaResults"

    vertices = np.loadtxt(r"data\DakotaPolygon.csv", delimiter=",")
    polygon = geometry.Polygon(vertices)

    xyz = np.loadtxt(r"data\DakotaWells.csv", delimiter=",")

    in_array = akeyaa.akeyaa(polygon, xyz, radius=3000, required=25, spacing=1000)

    arcpy.da.NumPyArrayToFeatureClass(
        in_array,
        out_table,
        ("x", "y"),
        arcpy.SpatialReference(26915)       # NAD 83 UTM zone 15N (EPSG:26915).
    )

    return in_array


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()