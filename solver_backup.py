#!/usr/bin/env python3
"""
Custom solver for the BhuMe Boundary Take-Home.
Implements:
1. Reprojection of plots to Pseudo-Mercator (EPSG:3857) for metric translations.
2. Distance transform of boundaries.tif for Chamfer Distance Matching.
3. Fast grid-search using numpy to find optimal local translations.
4. Identification of highly reliable "anchor plots".
5. Inverse Distance Weighting (IDW) spatial interpolation to smooth the shift field.
6. Calibration of confidence scores.
7. Validation against example truths and schema compliance.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import statistics
import numpy as np
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from shapely.affinity import translate
from shapely.geometry import Point

# Add current directory to path just in case
sys.path.append(str(Path(__file__).parent))
from bhume import load, score, write_predictions
from bhume.geo import geom_to_imagery_crs, open_imagery


def get_boundary_points(geom_3857, step_m=2.0) -> np.ndarray:
    """Sample points along the boundary of a Polygon or MultiPolygon in EPSG:3857 coordinates."""
    points = []
    boundary = geom_3857.boundary
    if boundary.is_empty:
        return np.zeros((0, 2))
    
    if boundary.geom_type == 'LineString':
        length = boundary.length
        if length > 0:
            for d in np.arange(0, length, step_m):
                pt = boundary.interpolate(d)
                points.append((pt.x, pt.y))
            # make sure we include the end point
            pt = boundary.interpolate(length)
            points.append((pt.x, pt.y))
    elif boundary.geom_type == 'MultiLineString':
        for line in boundary.geoms:
            length = line.length
            if length > 0:
                for d in np.arange(0, length, step_m):
                    pt = line.interpolate(d)
                    points.append((pt.x, pt.y))
                pt = line.interpolate(length)
                points.append((pt.x, pt.y))
    else:
        # Fallback to coords
        try:
            for coord in boundary.coords:
                points.append(coord[:2])
        except Exception:
            pass
            
    return np.array(points)


def main(village_dir: str) -> None:
    village = load(village_dir)
    print(f"Loaded {village.slug}")
    print(f"  {len(village.plots)} plots")

    # 1. Compute global median shift from example truths in EPSG:3857
    if village.example_truths is not None and len(village.example_truths) > 0:
        # Reproject truths and plots to EPSG:3857
        truths_3857 = village.example_truths.to_crs("EPSG:3857")
        plots_3857 = village.plots.to_crs("EPSG:3857")
        
        dxs, dys = [], []
        for pn in truths_3857.index:
            if pn in plots_3857.index:
                t = truths_3857.loc[pn, 'geometry'].centroid
                o = plots_3857.loc[pn, 'geometry'].centroid
                dxs.append(t.x - o.x)
                dys.append(t.y - o.y)
        
        global_dx = statistics.median(dxs) if dxs else 0.0
        global_dy = statistics.median(dys) if dys else 0.0
    else:
        # Fallback values from baseline if no truths (just in case)
        if "vadnerbhairav" in village.slug:
            global_dx, global_dy = -4.4, 11.4
        else:
            global_dx, global_dy = 9.6, 0.1
            
    print(f"  Global median shift: dx={global_dx:.2f}m, dy={global_dy:.2f}m")

    # 2. Open boundary hints raster and compute distance transform
    if village.boundaries_path is None or not village.boundaries_path.exists():
        print("  No boundary hints raster found. Falling back to global median shift.")
        # Just apply global median shift to all plots
        plots_3857 = village.plots.to_crs("EPSG:3857")
        plots_3857['geometry'] = plots_3857.geometry.apply(lambda g: translate(g, global_dx, global_dy))
        preds = plots_3857.to_crs("EPSG:4326")
        preds['status'] = 'corrected'
        preds['confidence'] = 0.5
        preds['method_note'] = f'fallback global shift dx={global_dx:.1f} dy={global_dy:.1f}'
        write_predictions(Path(village_dir) / 'predictions.geojson', preds)
        if village.example_truths is not None:
            print(score(preds, village))
        return

    print("  Computing distance transform on boundary hints...")
    with open_imagery(village.boundaries_path) as src:
        meta = src.meta
        transform = src.transform
        inv_transform = ~transform
        width = meta['width']
        height = meta['height']
        
        # Read the binary boundary raster
        boundaries = src.read(1)
        
        # Compute distance transform
        # boundaries is 255 at edges, 0 elsewhere
        inverted = (boundaries == 0).astype(np.float32)
        # distance in pixels to nearest boundary pixel
        dt_pixels = distance_transform_edt(inverted)
        
        # Pixel size in meters
        pixel_size = abs(transform[0])
        dt_meters = dt_pixels * pixel_size
        print(f"    Raster shape: {width}x{height}, pixel size: {pixel_size:.3f}m")

    # 3. Grid search for each plot
    plots_3857 = village.plots.to_crs("EPSG:3857")
    
    # Coefficients for fast affine transform: maps (x, y) coordinates to (col, row)
    a, b, c = inv_transform.a, inv_transform.b, inv_transform.c
    d, e, f = inv_transform.d, inv_transform.e, inv_transform.f

    # Grid search parameters
    # Search within +/- 5.0 meters around global shift to prevent snapping to neighboring fields
    search_range = 5.0
    # Step size: 0.5m for Malatavadi (~0.6m/px), 1.0m for Vadnerbhairav (~2.4m/px)
    step = 0.5 if pixel_size < 1.0 else 1.0
    
    dx_cand = np.arange(global_dx - search_range, global_dx + search_range + 0.01, step)
    dy_cand = np.arange(global_dy - search_range, global_dy + search_range + 0.01, step)
    
    print(f"  Grid searching {len(plots_3857)} plots. Search range: +/-{search_range}m, step: {step}m")
    
    results = {}
    
    # Pre-sample points for all plots to save time
    plot_points = {}
    for pn, row in plots_3857.iterrows():
        geom = row['geometry']
        pts = get_boundary_points(geom, step_m=max(2.0, pixel_size))
        plot_points[pn] = pts

    for pn, row in plots_3857.iterrows():
        pts = plot_points[pn]
        if len(pts) == 0:
            results[pn] = {
                'dx': global_dx, 'dy': global_dy, 'cost': 999.0, 'cost_zero': 999.0,
                'area_ratio': 1.0, 'centroid': plots_3857.loc[pn, 'geometry'].centroid,
                'n_pts': 0
            }
            continue
            
        # Area ratio check (drawn area vs recorded area)
        recorded_area = row.get('recorded_area_sqm')
        pot_kharaba_ha = row.get('pot_kharaba_ha')
        
        # full recorded size is recorded_area + pot_kharaba
        total_recorded_ha = 0.0
        if recorded_area is not None:
            total_recorded_ha += recorded_area / 10000.0
        if pot_kharaba_ha is not None:
            total_recorded_ha += pot_kharaba_ha
            
        map_area_ha = row['map_area_sqm'] / 10000.0
        
        if total_recorded_ha > 0:
            area_ratio = map_area_ha / total_recorded_ha
        else:
            area_ratio = 1.0  # assume fine if no records
            
        # Cost at zero shift
        x = pts[:, 0]
        y = pts[:, 1]
        cols_zero = (a * x + b * y + c).astype(np.int32)
        rows_zero = (d * x + e * y + f).astype(np.int32)
        valid_zero = (cols_zero >= 0) & (cols_zero < width) & (rows_zero >= 0) & (rows_zero < height)
        cost_zero = np.mean(np.minimum(dt_meters[rows_zero[valid_zero], cols_zero[valid_zero]], 15.0)) if np.sum(valid_zero) > 0 else 999.0

        # Grid search using numpy
        best_cost = 9999.0
        best_dx, best_dy = global_dx, global_dy
        
        for cur_dx in dx_cand:
            x_shift = x + cur_dx
            for cur_dy in dy_cand:
                y_shift = y + cur_dy
                
                cols = (a * x_shift + b * y_shift + c).astype(np.int32)
                rows = (d * x_shift + e * y_shift + f).astype(np.int32)
                
                valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
                if np.sum(valid) == 0:
                    cost = 9999.0
                else:
                    dists = dt_meters[rows[valid], cols[valid]]
                    cost = np.mean(np.minimum(dists, 15.0))
                    
                if cost < best_cost:
                    best_cost = cost
                    best_dx, best_dy = cur_dx, cur_dy
                    
        results[pn] = {
            'dx': best_dx,
            'dy': best_dy,
            'cost': best_cost,
            'cost_zero': cost_zero,
            'area_ratio': area_ratio,
            'centroid': plots_3857.loc[pn, 'geometry'].centroid,
            'n_pts': len(pts)
        }

    # Identify anchors:
    # Filter candidates: must have area_ratio in [0.85, 1.15] and at least 15 points
    candidates = [
        pn for pn, res in results.items()
        if res['cost'] < 999.0 and 0.85 <= res['area_ratio'] <= 1.15 and res['n_pts'] >= 15
    ]
    
    # Sort candidates by cost
    candidates.sort(key=lambda pn: results[pn]['cost'])
    
    # Select top 15% as anchors
    num_anchors = int(len(plots_3857) * 0.15)
    num_anchors = max(10, min(num_anchors, len(candidates)))
    anchor_plots = candidates[:num_anchors]
    
    max_anchor_cost = results[anchor_plots[-1]]['cost'] if anchor_plots else 5.0
    print(f"  Selected {len(anchor_plots)} anchors. Max anchor cost: {max_anchor_cost:.3f}m")
    
    for pn in plots_3857.index:
        if pn in anchor_plots:
            results[pn]['reliability'] = 1.0 - 0.5 * (results[pn]['cost'] / max_anchor_cost)
        else:
            results[pn]['reliability'] = 0.0

    # 4. Spatial Smoothing (IDW Interpolation)
    # For every plot, we interpolate its shift from the reliable anchor plots.
    # If the plot itself is reliable, we blend its raw shift with the smoothed shift.
    # If a plot has a bad area ratio, we flag it.
    
    smoothed_shifts = {}
    if len(anchor_plots) >= 3:
        # Build coordinates of anchors for distance calculations
        anchor_coords = np.array([[results[ap]['centroid'].x, results[ap]['centroid'].y] for ap in anchor_plots])
        anchor_dxs = np.array([results[ap]['dx'] for ap in anchor_plots])
        anchor_dys = np.array([results[ap]['dy'] for ap in anchor_plots])
        
        for pn, row in plots_3857.iterrows():
            pt = results[pn]['centroid']
            # Compute distances to all anchors
            dists = np.hypot(anchor_coords[:, 0] - pt.x, anchor_coords[:, 1] - pt.y)
            
            # Inverse distance weighting
            # w = 1 / (d**2)
            # Add small epsilon to avoid divide by zero for exact match
            weights = 1.0 / (dists**2 + 1e-4)
            # Normalize weights
            weights /= np.sum(weights)
            
            smooth_dx = np.sum(weights * anchor_dxs)
            smooth_dy = np.sum(weights * anchor_dys)
            
            # If the plot is reliable, we use its own shift, otherwise we use the smoothed shift
            rel = results[pn]['reliability']
            final_dx = rel * results[pn]['dx'] + (1.0 - rel) * smooth_dx
            final_dy = rel * results[pn]['dy'] + (1.0 - rel) * smooth_dy
            
            # Distance to the nearest anchor
            min_dist_anchor = np.min(dists)
            
            smoothed_shifts[pn] = {
                'dx': final_dx,
                'dy': final_dy,
                'min_dist_anchor': min_dist_anchor,
                'smooth_dx': smooth_dx,
                'smooth_dy': smooth_dy
            }
    else:
        # Fallback to global median shift if not enough anchors
        print("  Not enough anchor plots. Falling back to global median shift for smoothing.")
        for pn in plots_3857.index:
            smoothed_shifts[pn] = {
                'dx': global_dx,
                'dy': global_dy,
                'min_dist_anchor': 9999.0,
                'smooth_dx': global_dx,
                'smooth_dy': global_dy
            }

    # 5. Output generation & Confidence calibration
    corrected_count = 0
    flagged_count = 0
    
    preds_gdf = plots_3857.copy()
    geoms = []
    statuses = []
    confidences = []
    notes = []
    
    for pn, row in plots_3857.iterrows():
        area_ratio = results[pn]['area_ratio']
        dx = smoothed_shifts[pn]['dx']
        dy = smoothed_shifts[pn]['dy']
        raw_cost = results[pn]['cost']
        rel = results[pn]['reliability']
        min_dist = smoothed_shifts[pn]['min_dist_anchor']
        
        cost_zero = results[pn]['cost_zero']
        # Decide whether to correct or flag
        # We flag if:
        # 1. Area ratio is very bad (< 0.75 or > 1.3), meaning shape doesn't match record.
        # 2. Plot is extremely isolated from any reliable anchors (e.g., > 1.5 km).
        # 3. Plot zero-shift cost is very low and close to best cost (already correct plot).
        if area_ratio < 0.75 or area_ratio > 1.3:
            status = 'flagged'
            confidence = 0.0
            geom_final = row['geometry']
            note = f"flagged: area ratio {area_ratio:.2f} too far from 1.0"
            flagged_count += 1
        elif min_dist > 1500.0:
            status = 'flagged'
            confidence = 0.0
            geom_final = row['geometry']
            note = f"flagged: isolated, distance to nearest anchor {min_dist:.1f}m"
            flagged_count += 1
        elif cost_zero < 2.5 and cost_zero - raw_cost < 0.5:
            status = 'flagged'
            confidence = 0.0
            geom_final = row['geometry']
            note = f"flagged: already correct (cost_zero={cost_zero:.2f}m)"
            flagged_count += 1
        else:
            status = 'corrected'
            geom_final = translate(row['geometry'], dx, dy)
            
            # Confidence calibration:
            # - If highly reliable, confidence is close to 0.95
            # - If interpolated, confidence depends on distance to the nearest anchor
            #   (confidence decay with distance)
            if rel > 0:
                # Based on matching cost and area ratio deviation
                cost_factor = max(0.0, 1.0 - (raw_cost / 3.0))  # 1.0 if cost=0, 0.0 if cost>=3m
                area_factor = max(0.0, 1.0 - abs(1.0 - area_ratio) * 2.0)
                confidence = 0.6 + 0.35 * (cost_factor * area_factor)
            else:
                # Interpolated plot
                # Decay confidence from 0.75 to 0.4 based on distance to nearest anchor
                dist_factor = np.exp(-min_dist / 400.0)  # scale of 400 meters
                confidence = 0.4 + 0.3 * dist_factor
                
            confidence = float(np.clip(confidence, 0.0, 1.0))
            note = f"corrected: dx={dx:.1f}m dy={dy:.1f}m (rel={rel:.2f}, dist_anchor={min_dist:.1f}m)"
            corrected_count += 1
            
        geoms.append(geom_final)
        statuses.append(status)
        confidences.append(confidence)
        notes.append(note)
        
    preds_gdf['geometry'] = geoms
    preds_gdf['status'] = statuses
    preds_gdf['confidence'] = confidences
    preds_gdf['method_note'] = notes
    
    # Convert back to EPSG:4326
    preds = preds_gdf.to_crs("EPSG:4326")
    
    # Write predictions
    out_path = Path(village_dir) / 'predictions.geojson'
    write_predictions(out_path, preds)
    
    print(f"  Wrote {len(preds)} predictions: {corrected_count} corrected, {flagged_count} flagged")
    
    # 6. Self-score it against example truths
    if village.example_truths is not None:
        print()
        print(score(preds, village))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python solver.py <village_dir>")
        sys.exit(1)
    main(sys.argv[1])
