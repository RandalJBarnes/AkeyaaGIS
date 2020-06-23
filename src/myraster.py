import numpy as np
import arcpy



rasInfo = arcpy.RasterInfo()

rasInfo.setBandCount(1)
rasInfo.setCellSize((1000, 1000))
rasInfo.setExtent(arcpy.Extent(474000., 4924000., 521000., 4974000.))
rasInfo.setPixelType("F64")
rasInfo.setNoDataValues(np.nan)
rasInfo.setSpatialReference(arcpy.SpatialReference(26915))

outRas = arcpy.Raster(rasInfo)
