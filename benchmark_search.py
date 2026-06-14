import numpy as np
import time

def benchmark():
    n_plots = 2500
    n_pts_per_plot = 50
    n_candidates = 676  # 26 x 26
    
    # Simulate data
    x = np.random.randn(n_plots, n_pts_per_plot) * 100
    y = np.random.randn(n_plots, n_pts_per_plot) * 100
    
    dt_meters = np.random.rand(3000, 3000).astype(np.float32)
    width, height = 3000, 3000
    
    dx_cand = np.linspace(-25, 25, 26)
    dy_cand = np.linspace(-25, 25, 26)
    
    a, b, c = 0.8, 0.0, 10.0
    d, e, f = 0.0, -0.8, 2000.0
    
    t0 = time.time()
    
    # Simple loop over plots, vectorized over candidates
    results = []
    for i in range(n_plots):
        px = x[i]
        py = y[i]
        
        # Grid search: we can vectorize this
        best_cost = 9999.0
        best_dx, best_dy = 0.0, 0.0
        
        # We can construct shifted points for all candidates
        # px: (50,)
        # dx_cand: (26,)
        # dy_cand: (26,)
        # We want to check all 26x26 shifts
        # To make it fast, we can do loops in Python but use numpy inside
        for cur_dx in dx_cand:
            x_shift = px + cur_dx
            for cur_dy in dy_cand:
                y_shift = py + cur_dy
                
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
                    
    t1 = time.time()
    print(f"Time taken for {n_plots} plots with {n_candidates} candidates: {t1 - t0:.3f} seconds")

benchmark()
