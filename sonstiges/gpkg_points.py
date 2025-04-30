from osgeo import ogr
import os
ogr.UseExceptions()


run24 = "/media/riannek/minimax/gleis/run24-2024-08-13"
run14 = "/media/riannek/minimax/gleis/run14-2024-08-14"

run24_temp = "/media/riannek/minimax/gleis/temp_run24"
run14_temp = "/media/riannek/minimax/gleis/temp_run14"

gpkg_points = []

gpkg = ogr.Open(os.path.join(run14_temp, "temp.gpkg"))
layer = gpkg.GetLayerByName("multipoints")

for feature in layer:
    geom = feature.GetGeometryRef()
    if geom.GetGeometryName() == "MULTIPOINT":
        for i in range(geom.GetGeometryCount()):
            point = geom.GetGeometryRef(i)
            gpkg_points.append(point.GetPoint())
    else:
        gpkg_points.append(geom.GetPoint())

gpkg_points = np.array(gpkg_points)  