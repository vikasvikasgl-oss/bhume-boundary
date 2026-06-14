import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from bhume import load
from solver import get_boundary_points, vectorized_grid_search
from scipy.ndimage import distance_transform_edt
from bhume.geo import open_imagery

village = load('data/12429_malatavadi_chandgad_kolhapur')
with open_imagery(village.boundaries_path) as src:
    meta = src.meta
    transform = src.transform
    inv_transform = ~transform
    width = meta['width']
    height = meta['height']
    boundaries = src.read(1)
    inverted = (boundaries == 0).astype(np.float32)
    dt_pixels = distance_transform_edt(inverted)
    pixel_size = abs(transform[0])
    dt_meters = dt_pixels * pixel_size

plots_3857 = village.plots.to_crs("EPSG:3857")
geom = plots_3857.loc['1763', 'geometry']
pts = get_boundary_points(geom, step_m=max(2.0, pixel_size))

a, b, c = inv_transform.a, inv_transform.b, inv_transform.c
d, e, f = inv_transform.d, inv_transform.e, inv_transform.f

shifts_to_test = [
    (0.0, 0.0, "zero shift"),
    (9.8, -5.2, "original predicted shift"),
    (13.92, 0.05, "true shift"),
    (9.0, -15.0, "coarse shift")
]

x = pts[:, 0]
y = pts[:, 1]
cap_dist = 6.0 * pixel_size

for dx, dy, label in shifts_to_test:
    x_shift = x + dx
    y_shift = y + dy
    cols = (a * x_shift + b * y_shift + c).astype(np.int32)
    rows = (d * x_shift + e * y_shift + f).astype(np.int32)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if np.sum(valid) == 0:
        cost = 9999.0
    else:
        dists = dt_meters[rows[valid], cols[valid]]
        cost = np.mean(np.minimum(dists, cap_dist))
    print(f"{label} (dx={dx:.2f}, dy={dy:.2f}): cost={cost:.3f}")
