import geopandas as gpd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.score import _utm_for

village = load('data/12429_malatavadi_chandgad_kolhapur')
utm = _utm_for(village.example_truths.geometry.iloc[0])

official_pt = village.plots.loc['1177', 'geometry'].centroid
truth_pt = village.example_truths.loc['1177', 'geometry'].centroid

# Reproject to UTM
official_utm = village.plots.to_crs(utm).loc['1177', 'geometry'].centroid
truth_utm = village.example_truths.to_crs(utm).loc['1177', 'geometry'].centroid

print("Official Centroid (WGS84):", official_pt.x, official_pt.y)
print("Truth Centroid (WGS84):", truth_pt.x, truth_pt.y)
print("Official Centroid (UTM):", official_utm.x, official_utm.y)
print("Truth Centroid (UTM):", truth_utm.x, truth_utm.y)

dx = truth_utm.x - official_utm.x
dy = truth_utm.y - official_utm.y
print(f"Required shift for 1177 (UTM): dx={dx:.2f}m, dy={dy:.2f}m")
print(f"Status in truth dataset: {village.example_truths.loc['1177'].get('status')}")
