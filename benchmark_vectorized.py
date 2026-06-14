import numpy as np
import time

def benchmark_vectorized():
    n_plots = 2500
    n_pts_per_plot = 50
    n_candidates = 676  # 26 x 26
    
    # Simulate data
    x = np.random.randn(n_plots, n_pts_per_plot) * 100
    y = np.random.randn(n_plots, n_pts_per_plot) * 100
    
    dt_meters = np.random.rand(3000, 3000).astype(np.float32) * 15.0
    width, height = 3000, 3000
    
    dx_cand = np.linspace(-25, 25, 26).astype(np.float32)
    dy_cand = np.linspace(-25, 25, 26).astype(np.float32)
    
    dx_grid, dy_grid = np.meshgrid(dx_cand, dy_cand)
    dxs = dx_grid.flatten()
    dys = dy_grid.flatten()
    
    a, b, c = 0.8, 0.0, 10.0
    d, e, f = 0.0, -0.8, 2000.0
    
    t0 = time.time()
    
    results = []
    for i in range(n_plots):
        px = x[i]
        py = y[i]
        
        # Vectorized grid search
        # px: (50,)
        # dxs: (676,)
        x_shifts = px[:, np.newaxis] + dxs[np.newaxis, :]  # (50, 676)
        y_shifts = py[:, np.newaxis] + dys[np.newaxis, :]  # (50, 676)
        
        cols = (a * x_shifts + b * y_shifts + c).astype(np.int32)
        rows = (d * x_shifts + e * y_shifts + f).astype(np.int32)
        
        in_bounds = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        cols_clipped = np.clip(cols, 0, width - 1)
        rows_clipped = np.clip(rows, 0, height - 1)
        
        dists = dt_meters[rows_clipped, cols_clipped]
        dists = np.where(in_bounds, dists, 15.0)
        dists = np.minimum(dists, 15.0)
        
        costs = np.mean(dists, axis=0)  # (676,)
        
        best_idx = np.argmin(costs)
        best_dx = dxs[best_idx]
        best_dy = dys[best_idx]
        best_cost = costs[best_idx]
        
        results.append((best_dx, best_dy, best_cost))
        
    t1 = time.time()
    print(f"Vectorized time taken for {n_plots} plots: {t1 - t0:.3f} seconds")

benchmark_vectorized()
