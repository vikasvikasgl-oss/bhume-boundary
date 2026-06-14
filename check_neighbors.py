import sys
from pathlib import Path
import geopandas as gpd

sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.score import _utm_for

village = load('data/12429_malatavadi_chandgad_kolhapur')
utm = _utm_for(village.plots.geometry.iloc[0])

plots_utm = village.plots.to_crs(utm)
geom_1763 = plots_utm.loc['1763', 'geometry']
geom_1966 = plots_utm.loc['1966', 'geometry']
geom_1177 = plots_utm.loc['1177', 'geometry']

print(f"Distance 1763 to 1966: {geom_1763.distance(geom_1966):.1f}m")
print(f"Distance 1763 to 1177: {geom_1763.distance(geom_1177):.1f}m")
print(f"Distance 1966 to 1177: {geom_1966.distance(geom_1177):.1f}m")
