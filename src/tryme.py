import arcpy
import akeyaa

__version__ = "24 June 2020"


#-------------------------------------------------------------------------------
# The constants CWIGDB and CTYGDB are specific paths on the local machine.
# These paths must be modified to point to the gdb's as installed on the local
# machine. Similarly, DESTINATION is the local path to the output table.
#-------------------------------------------------------------------------------

# Minnesota County Well Index (CWI). Obtained from the Minnesota Department
# of Health. The CWI is administered by the Minnesota Geological Survey.
# https://www.mngs.umn.edu/cwi.html
# This gdb uses 'NAD 1983 UTM zone 15N' (EPSG:26915).
CWIGDB = r"D:\Google Drive\CWI\CWI_20200420\water_well_information.gdb"

# County Boundaries, Minnesota. Obtained from the Minnesota Geospatial Commons.
# https://gisdata.mn.gov/dataset/bdry-counties-in-minnesota
# This gdb uses 'NAD 1983 UTM zone 15N' (EPSG:26915).
CTYGDB = r"D:\Google Drive\GIS\fgdb_bdry_counties_in_minnesota\bdry_counties_in_minnesota.gdb"

# Where to put the feature class files after they are created.
FC_DEST = r"D:\Google Drive\Projects\AkeyaaGIS\data\DakotaResults"


# -----------------------------------------------------------------------------
def main():
    polygon, xyz = get_dakota_county_data()
    akeyaa_array, akeyaa_raster = akeyaa.run_akeyaa(
        polygon, xyz, radius=3000, required=25, spacing=1000, fc_dest=FC_DEST
    )

    # arcpy.da.NumPyArrayToFeatureClass(
    #     akeyaa_array,
    #     fc_dest,
    #     ("x", "y"),
    #     spatial_reference
    # )

    return akeyaa_array, akeyaa_raster


# -----------------------------------------------------------------------------
def get_dakota_county_data():
    """Get the polygon and well data from Dakota County.

    This function has one purpose only: to create a test case. This function
    will be discarded after we hook into ArcGIS Pro.

    """
    CTY_SOURCE = CTYGDB + r"\mn_county_boundaries"  # County boundaries
    ALLWELLS = CWIGDB + r"\allwells"                # MN county well index
    C5WL = CWIGDB + r"\C5WL"                        # MN static water levels

    # Get the Dakota County polygon.
    source = CTY_SOURCE
    what = ["CTY_NAME", "CTY_FIPS", "SHAPE@"]
    where = "(CTY_FIPS = 37)"

    results = []
    with arcpy.da.SearchCursor(source, what, where) as cursor:
        for row in cursor:
            results.append(row)
    dakota_polygon = results[0][2]

    # Get the Dakota County wells.
    in_table = arcpy.AddJoin_management(ALLWELLS, "RELATEID", C5WL, "RELATEID", False)
    dakota_table = arcpy.SelectLayerByLocation_management(in_table, 'WITHIN', dakota_polygon)

    field_names = ["allwells.SHAPE", "C5WL.MEAS_ELEV"]
    where_clause = (
        "(C5WL.MEAS_ELEV is not NULL) AND "
        "(allwells.AQUIFER is not NULL) AND "
        "(allwells.UTME is not NULL) AND "
        "(allwells.UTMN is not NULL)"
    )
    dakota_wells = []
    with arcpy.da.SearchCursor(dakota_table, field_names, where_clause) as cursor:
        for row in cursor:
            dakota_wells.append((row[0][0], row[0][1], row[1]))

    return dakota_polygon, dakota_wells


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()
