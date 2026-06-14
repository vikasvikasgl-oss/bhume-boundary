import geopandas as gpd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.score import _utm_for

def inspect_spatial_truths(village_dir):
    village = load(village_dir)
    utm = _utm_for(village.example_truths.geometry.iloc[0])
    truth_u = village.example_truths.to_crs(utm)
    official_u = village.plots.to_crs(utm)
    
    print(f"\n=== Spatial Truths for {Path(village_dir).name} ===")
    for pn in truth_u.index:
        t_geom = truth_u.loc[pn, 'geometry']
        o_geom = official_u.loc[pn, 'geometry']
        dx = t_geom.centroid.x - o_geom.centroid.x
        dy = t_geom.centroid.y - o_geom.centroid.y
        print(f"Plot {pn}: Centroid=({o_geom.centroid.x:.1f}, {o_geom.centroid.y:.1f}), shift=({dx:.2f}, {dy:.2f})")

inspect_spatial_truths('data/34855_vadnerbhairav_chandavad_nashik')
inspect_spatial_truths('data/12429_malatavadi_chandgad_kolhapur')
