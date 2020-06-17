=========
AkeyaaGIS
=========

Compute flow directions using a locally-fitted conic discharge potential model,
as a fully integrated ArcGIS Pro toolbox.


Method
------
The Akeyaa analysis is carried out at target locations within a selected venue (e.g. a polygon).

The target locations are selected as the nodes of a square grid covering the venue.

The square grid of target locations is anchored at the centroid of the venue, and the grid lines
are separated by `spacing`. If a target location is not inside of the venue it is discarded.

For each target location, all wells that are within a horizontal distance of `radius` of the
target location are identified.

If a target location has fewer than `required` identified (neighboring) wells it is discarded.

The Akeyaa analysis is carried out at each of the remaining target locations using the `method`
for fitting the conic potential model.
