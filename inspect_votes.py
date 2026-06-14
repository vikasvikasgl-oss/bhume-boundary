import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.affinity import translate

sys.path.append(str(Path(__file__).parent))
from bhume import load, score
from bhume.geo import open_imagery
from scipy.ndimage import distance_transform_edt
from solver import get_boundary_points, vectorized_grid_search

def test_decentralized_solver(village_dir, pass2_range=5.0, idw_power=2):
    village = load(village_dir)
    print(f"\n=== Testing solver with pass2_range={pass2_range}, idw_power={idw_power} for {Path(village_dir).name} ===")
    
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
    
    coarse_results = {}
    for pn, row in plots_3857.iterrows():
        pts = plot_points[pn]
        
        recorded_area = row.get('recorded_area_sqm')
        pot_kharaba_ha = row.get('pot_kharaba_ha')
        total_recorded_ha = 0.0
        if recorded_area is not None:
            total_recorded_ha += recorded_area / 10000.0
        if pot_kharaba_ha is not None:
            total_recorded_ha += pot_kharaba_ha
        map_area_ha = row['map_area_sqm'] / 10000.0
        area_ratio = map_area_ha / total_recorded_ha if total_recorded_ha > 0 else 1.0

        if len(pts) == 0:
            coarse_results[pn] = {
                'dx': 0.0, 'dy': 0.0, 'cost': 999.0, 'contrast': 0.0, 'area_ratio': area_ratio,
                'centroid': plots_3857.loc[pn, 'geometry'].centroid, 'n_pts': 0
            }
            continue

        best_dx, best_dy, best_cost, contrast = vectorized_grid_search(
            pts, dx_cand_coarse, dy_cand_coarse, dt_meters, inv_transform, width, height, cap_dist
        )
        coarse_results[pn] = {
            'dx': best_dx, 'dy': best_dy, 'cost': best_cost, 'contrast': contrast, 'area_ratio': area_ratio,
            'centroid': plots_3857.loc[pn, 'geometry'].centroid, 'n_pts': len(pts)
        }

    anchor_candidates = []
    for pn, res in coarse_results.items():
        if res['cost'] < 4.0 and res['contrast'] > min_contrast_thresh and 0.85 <= res['area_ratio'] <= 1.15 and res['n_pts'] >= 15:
            anchor_candidates.append(pn)
            
    anchor_candidates.sort(key=lambda pn: coarse_results[pn]['contrast'], reverse=True)
    num_anchors = int(len(plots_3857) * 0.15)
    num_anchors = max(10, min(num_anchors, len(anchor_candidates)))
    anchor_plots = anchor_candidates[:num_anchors]

    anchor_coords = np.array([[coarse_results[ap]['centroid'].x, coarse_results[ap]['centroid'].y] for ap in anchor_plots])
    anchor_dxs = np.array([coarse_results[ap]['dx'] for ap in anchor_plots])
    anchor_dys = np.array([coarse_results[ap]['dy'] for ap in anchor_plots])
    
    priors = {}
    for pn, row in plots_3857.iterrows():
        if pn in anchor_plots:
            priors[pn] = {
                'dx': coarse_results[pn]['dx'],
                'dy': coarse_results[pn]['dy'],
                'min_dist_anchor': 0.0
            }
        elif len(anchor_plots) >= 3:
            pt = coarse_results[pn]['centroid']
            dists = np.hypot(anchor_coords[:, 0] - pt.x, anchor_coords[:, 1] - pt.y)
            weights = 1.0 / (dists**idw_power + 1e-4)
            weights /= np.sum(weights)
            
            prior_dx = np.sum(weights * anchor_dxs)
            prior_dy = np.sum(weights * anchor_dys)
            priors[pn] = {
                'dx': prior_dx,
                'dy': prior_dy,
                'min_dist_anchor': float(np.min(dists))
            }
        else:
            priors[pn] = {
                'dx': 0.0,
                'dy': 0.0,
                'min_dist_anchor': 9999.0
            }

    # Pass 2: Fine Prior-Regularized Search
    fine_results = {}
    for pn, row in plots_3857.iterrows():
        pts = plot_points[pn]
        if len(pts) == 0:
            fine_results[pn] = {'dx': 0.0, 'dy': 0.0, 'cost': 999.0, 'cost_zero': 999.0}
            continue
            
        prior_dx = priors[pn]['dx']
        prior_dy = priors[pn]['dy']
        
        # Search window of pass2_range with 0.25m step
        dx_cand_fine = np.arange(prior_dx - pass2_range, prior_dx + pass2_range + 0.01, 0.25).astype(np.float32)
        dy_cand_fine = np.arange(prior_dy - pass2_range, prior_dy + pass2_range + 0.01, 0.25).astype(np.float32)
        
        best_dx, best_dy, best_cost, _ = vectorized_grid_search(
            pts, dx_cand_fine, dy_cand_fine, dt_meters, inv_transform, width, height, cap_dist
        )
        
        _, _, cost_zero, _ = vectorized_grid_search(
            pts, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32),
            dt_meters, inv_transform, width, height, cap_dist
        )
        
        fine_results[pn] = {
            'dx': best_dx,
            'dy': best_dy,
            'cost': best_cost,
            'cost_zero': cost_zero
        }

    preds_gdf = plots_3857.copy()
    geoms = []
    statuses = []
    confidences = []
    notes = []
    
    for pn, row in plots_3857.iterrows():
        pts = plot_points[pn]
        area_ratio = coarse_results[pn]['area_ratio']
        min_dist = priors[pn]['min_dist_anchor']
        
        if len(pts) == 0:
            status = 'flagged'
            confidence = 0.0
            geom_final = row['geometry']
            note = "flagged: no boundary points to align"
        else:
            dx = fine_results[pn]['dx']
            dy = fine_results[pn]['dy']
            best_cost = fine_results[pn]['cost']
            cost_zero = fine_results[pn]['cost_zero']
            
            is_control = (cost_zero < control_cost_thresh) or (cost_zero <= best_cost + improvement_thresh)
            
            if area_ratio < 0.75 or area_ratio > 1.3:
                status = 'flagged'
                confidence = 0.0
                geom_final = row['geometry']
                note = f"flagged: area ratio"
            elif min_dist > 1500.0:
                status = 'flagged'
                confidence = 0.0
                geom_final = row['geometry']
                note = f"flagged: isolated"
            elif is_control:
                status = 'flagged'
                confidence = 0.0
                geom_final = row['geometry']
                note = f"flagged: already correct"
            else:
                status = 'corrected'
                geom_final = translate(row['geometry'], dx, dy)
                
                cost_factor = np.clip(1.0 - (best_cost / (4.0 * pixel_size)), 0.0, 1.0)
                area_factor = np.clip(1.0 - abs(1.0 - area_ratio) * 2.0, 0.0, 1.0)
                
                if pn in anchor_plots:
                    confidence = 0.85 + 0.12 * (cost_factor * area_factor)
                else:
                    dist_factor = np.exp(-min_dist / 400.0)
                    cost_impr = cost_zero - best_cost
                    if cost_impr > improvement_thresh:
                        confidence = 0.65 + 0.20 * (cost_factor * area_factor * dist_factor)
                    else:
                        confidence = 0.45 + 0.20 * (area_factor * dist_factor)
                        
                confidence = float(np.clip(confidence, 0.1, 1.0))
                note = f"corrected"
                
        geoms.append(geom_final)
        statuses.append(status)
        confidences.append(confidence)
        notes.append(note)
        
    preds_gdf['geometry'] = geoms
    preds_gdf['status'] = statuses
    preds_gdf['confidence'] = confidences
    preds_gdf['method_note'] = notes
    
    preds = preds_gdf.to_crs("EPSG:4326")
    print(score(preds, village))

test_decentralized_solver('data/12429_malatavadi_chandgad_kolhapur', pass2_range=4.0, idw_power=2)
test_decentralized_solver('data/12429_malatavadi_chandgad_kolhapur', pass2_range=5.0, idw_power=2)
test_decentralized_solver('data/12429_malatavadi_chandgad_kolhapur', pass2_range=4.0, idw_power=3)
test_decentralized_solver('data/12429_malatavadi_chandgad_kolhapur', pass2_range=5.0, idw_power=3)
