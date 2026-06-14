import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.geo import open_imagery
from scipy.ndimage import distance_transform_edt
from solver import get_boundary_points

def find_global_shift_regularized(village_dir, l2_penalty=0.003):
    village = load(village_dir)
    print(f"\n=== Regularized global shift search for {Path(village_dir).name} (L2 penalty={l2_penalty}) ===")
    
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
    
    sampled_pts = []
    count = 0
    for pn, row in plots_3857.iterrows():
        recorded_area = row.get('recorded_area_sqm')
        pot_kharaba_ha = row.get('pot_kharaba_ha')
        total_recorded_ha = 0.0
        if recorded_area is not None:
            total_recorded_ha += recorded_area / 10000.0
        if pot_kharaba_ha is not None:
            total_recorded_ha += pot_kharaba_ha
        map_area_ha = row['map_area_sqm'] / 10000.0
        area_ratio = map_area_ha / total_recorded_ha if total_recorded_ha > 0 else 1.0
        
        if 0.9 <= area_ratio <= 1.1:
            pts = get_boundary_points(row['geometry'], step_m=max(3.0, pixel_size))
            if len(pts) >= 15:
                sampled_pts.append(pts)
                count += 1
                if count >= 150:
                    break

    dx_cand = np.arange(-25.0, 25.0 + 0.01, 1.0).astype(np.float32)
    dy_cand = np.arange(-25.0, 25.0 + 0.01, 1.0).astype(np.float32)
    
    dx_grid, dy_grid = np.meshgrid(dx_cand, dy_cand)
    dxs = dx_grid.flatten()
    dys = dy_grid.flatten()
    
    a, b, c = inv_transform.a, inv_transform.b, inv_transform.c
    d, e, f = inv_transform.d, inv_transform.e, inv_transform.f
    
    cap_dist = 6.0 * pixel_size
    
    total_costs = np.zeros_like(dxs, dtype=np.float32)
    for pts in sampled_pts:
        px = pts[:, 0]
        py = pts[:, 1]
        
        x_shifts = px[:, np.newaxis] + dxs[np.newaxis, :]
        y_shifts = py[:, np.newaxis] + dys[np.newaxis, :]
        
        cols = (a * x_shifts + b * y_shifts + c).astype(np.int32)
        rows = (d * x_shifts + e * y_shifts + f).astype(np.int32)
        
        in_bounds = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        cols_clipped = np.clip(cols, 0, width - 1)
        rows_clipped = np.clip(rows, 0, height - 1)
        
        dists = dt_meters[rows_clipped, cols_clipped]
        dists = np.where(in_bounds, dists, cap_dist)
        dists = np.minimum(dists, cap_dist)
        
        costs = np.mean(dists, axis=0)
        total_costs += costs
        
    total_costs /= len(sampled_pts)
    
    # Compute penalized loss
    penalties = l2_penalty * (dxs**2 + dys**2)
    losses = total_costs + penalties
    
    best_idx = np.argmin(losses)
    best_dx = dxs[best_idx]
    best_dy = dys[best_idx]
    print(f"Penalized global shift: dx={best_dx:.2f}m, dy={best_dy:.2f}m (cost={total_costs[best_idx]:.3f}, loss={losses[best_idx]:.3f})")
    
    sorted_idxs = np.argsort(losses)
    print("Top 5 penalized shifts:")
    for idx in sorted_idxs[:5]:
        print(f"  Shift ({dxs[idx]:.1f}, {dys[idx]:.1f}): cost={total_costs[idx]:.3f}, penalty={penalties[idx]:.3f}, loss={losses[idx]:.3f}")

# Test different L2 penalties
for p in [0.001, 0.003, 0.005, 0.01]:
    find_global_shift_regularized('data/12429_malatavadi_chandgad_kolhapur', l2_penalty=p)
    
print("\n--- Vadnerbhairav ---")
find_global_shift_regularized('data/34855_vadnerbhairav_chandavad_nashik', l2_penalty=0.003)
