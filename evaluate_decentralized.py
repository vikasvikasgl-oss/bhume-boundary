import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.geo import open_imagery
from scipy.ndimage import distance_transform_edt
from solver import get_boundary_points, vectorized_grid_search

def inspect_details():
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

    cap_dist = float(6.0 * pixel_size)
    control_cost_thresh = float(2.3 * pixel_size)
    improvement_thresh = float(0.6 * pixel_size)
    min_contrast_thresh = float(0.5 * pixel_size)

    plots_3857 = village.plots.to_crs("EPSG:3857")
    plot_points = {}
    for pn, row in plots_3857.iterrows():
        geom = row['geometry']
        pts = get_boundary_points(geom, step_m=max(2.0, pixel_size))
        plot_points[pn] = pts

    dx_cand_coarse = np.arange(-15.0, 15.0 + 0.01, 1.0).astype(np.float32)
    dy_cand_coarse = np.arange(-15.0, 15.0 + 0.01, 1.0).astype(np.float32)

    for pn in ['1763', '1966']:
        pts = plot_points[pn]
        best_dx, best_dy, best_cost, contrast = vectorized_grid_search(
            pts, dx_cand_coarse, dy_cand_coarse, dt_meters, inv_transform, width, height, cap_dist
        )
        _, _, cost_zero, _ = vectorized_grid_search(
            pts, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32),
            dt_meters, inv_transform, width, height, cap_dist
        )
        
        print(f"Plot {pn}: cost_zero={cost_zero:.3f}, best_cost={best_cost:.3f}, improvement={cost_zero - best_cost:.3f}")
        print(f"  control_cost_thresh={control_cost_thresh:.3f}, improvement_thresh={improvement_thresh:.3f}")

inspect_details()
