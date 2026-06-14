import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.geo import open_imagery
from scipy.ndimage import distance_transform_edt
from solver import get_boundary_points

def print_profile(village_dir):
    village = load(village_dir)
    print(f"\n=== Cost Profile for {Path(village_dir).name} ===")
    
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

    a, b, c = inv_transform.a, inv_transform.b, inv_transform.c
    d, e, f = inv_transform.d, inv_transform.e, inv_transform.f
    cap_dist = 6.0 * pixel_size
    
    shifts_to_test = [
        (0.0, 0.0, "zero shift"),
        (2.0, 0.0, "dx=2"),
        (4.0, 0.0, "dx=4"),
        (6.0, 0.0, "dx=6"),
        (8.0, 0.0, "dx=8"),
        (10.0, 0.0, "dx=10"),
        (12.0, 0.0, "dx=12"),
        (14.0, 0.0, "dx=14"),
        (20.0, -4.0, "false snap"),
    ]
    
    for dx, dy, label in shifts_to_test:
        total_cost = 0.0
        for pts in sampled_pts:
            px = pts[:, 0] + dx
            py = pts[:, 1] + dy
            cols = (a * px + b * py + c).astype(np.int32)
            rows = (d * px + e * py + f).astype(np.int32)
            valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            if np.sum(valid) == 0:
                cost = cap_dist
            else:
                dists = dt_meters[rows[valid], cols[valid]]
                cost = np.mean(np.minimum(dists, cap_dist))
            total_cost += cost
        total_cost /= len(sampled_pts)
        print(f"  {label} (dx={dx:.1f}, dy={dy:.1f}): cost={total_cost:.4f}")

print_profile('data/12429_malatavadi_chandgad_kolhapur')
print_profile('data/34855_vadnerbhairav_chandavad_nashik')
