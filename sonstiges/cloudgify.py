# Based on code by (c) Florent Poux
# https://towardsdatascience.com/how-to-generate-gifs-from-3d-models-with-python/

import pyvista as pv
import numpy as np
import os

def cloudgify(cloud, scalars, filename="cloud.gif", point_size=1.0, factor=3.0, up=1.0, viewup=[0, 0, 1], background_color='k', n_points=80, opacity=0.65, rgb=False, eye_dome_lighting=True, style='points'):
    if not (filename.endswith('.gif') or filename.endswith('.mp4')):
        raise ValueError("Filename must end with .gif or .mp4")
    if not isinstance(cloud, pv.PolyData):
        raise ValueError("cloud must be a pyvista PolyData object")

    if scalars=="dist center":
        scalars = np.linalg.norm(cloud.points - cloud.center, axis=1)
    pl = pv.Plotter(off_screen=True, image_scale=1)
    pl.add_mesh(
        cloud,
        style='Points',
        render_points_as_spheres=True,
        emissive=False,
        color='#fff7c2',
        scalars=scalars,
        opacity=opacity,
        point_size=point_size,
        show_scalar_bar=False,
        rgb=rgb,
        )

    pl.background_color = background_color
    if eye_dome_lighting:
        pl.enable_eye_dome_lighting()
    pl.show(auto_close=False)

    #print(os.path.abspath(filename))

    path = pl.generate_orbital_path(n_points=n_points, shift=cloud.length*up, viewup=viewup, factor=factor)

    if filename.endswith('gif'):
        pl.open_gif(filename)
    else:
        pl.open_movie(filename)
    pl.orbit_on_path(path, write_frames=True, viewup=viewup)
    pl.close()

    return