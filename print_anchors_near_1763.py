import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from bhume import load
from solver import get_boundary_points, vectorized_grid_search
from scipy.ndimage import distance_transform_edt
from bhume.geo import open_imagery

village = load('data/12429_malatavadi_chandgad_kolhapur')
plots_3857 = village.plots.to_crs("EPSG:3857")

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
min_contrast_thresh = float(0.5 * pixel_size)

plot_points = {}
for pn, row in plots_3857.iterrows():
    geom = row['geometry']
    pts = get_boundary_points(geom, step_m=max(2.0, pixel_size))
    plot_points[pn] = pts

dx_cand_coarse = np.arange(-15.0, 15.0 + 0.01, 1.0).astype(np.float32)
dy_cand_coarse = np.arange(-15.0, 15.0 + 0.01, 1.0).astype(np.float32)

coarse_results = {}
for pn, row in plots_3857.iterrows():
    pts = plot_points[pn]
    if len(pts) == 0:
        continue
    best_dx, best_dy, best_cost, contrast = vectorized_grid_search(
        pts, dx_cand_coarse, dy_cand_coarse, dt_meters, inv_transform, width, height, cap_dist
    )
    
    recorded_area = row.get('recorded_area_sqm')
    pot_kharaba_ha = row.get('pot_kharaba_ha')
    total_recorded_ha = 0.0
    if recorded_area is not None:
        total_recorded_ha += recorded_area / 10000.0
    if pot_kharaba_ha is not None:
        total_recorded_ha += pot_kharaba_ha
    map_area_ha = row['map_area_sqm'] / 10000.0
    area_ratio = map_area_ha / total_recorded_ha if total_recorded_ha > 0 else 1.0

    coarse_results[pn] = {
        'dx': best_dx, 'dy': best_dy, 'cost': best_cost, 'contrast': contrast, 'area_ratio': area_ratio,
        'centroid': plots_3857.loc[pn, 'geometry'].centroid, 'n_pts': len(pts)
    }

anchor_candidates = []
for pn, res in coarse_results.items():
    if abs(res['dx']) >= 14.5 or abs(res['dy']) >= 14.5:
        continue
    if res['cost'] < 4.0 and res['contrast'] > min_contrast_thresh and 0.85 <= res['area_ratio'] <= 1.15 and res['n_pts'] >= 15:
        anchor_candidates.append(pn)
        
anchor_candidates.sort(key=lambda pn: coarse_results[pn]['contrast'], reverse=True)
num_anchors = int(len(plots_3857) * 0.15)
num_anchors = max(10, min(num_anchors, len(anchor_candidates)))
anchor_plots = anchor_candidates[:num_anchors]

target_centroid = plots_3857.loc['1763', 'geometry'].centroid

anchors_with_dists = []
for ap in anchor_plots:
    ap_centroid = coarse_results[ap]['centroid']
    dist = np.hypot(ap_centroid.x - target_centroid.x, ap_centroid.y - target_centroid.y)
    anchors_with_dists.append((ap, dist, coarse_results[ap]['dx'], coarse_results[ap]['dy']))

anchors_with_dists.sort(key=lambda x: x[1])
print("Closest clean anchors to 1763:")
for ap, dist, dx, dy in anchors_with_dists[:15]:
    print(f"Anchor {ap}: dist={dist:.1f}m, dx={dx:.2f}m, dy={dy:.2f}m")
