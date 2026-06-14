import geopandas as gpd
from pathlib import Path

def inspect_village_truths(village_dir):
    print(f"\n=== Truths for {Path(village_dir).name} ===")
    truths_path = Path(village_dir) / 'example_truths.geojson'
    if not truths_path.exists():
        print("No truths found")
        return
    gdf = gpd.read_file(truths_path)
    print(gdf[['plot_number', 'status', 'geometry']])
    if 'confidence' in gdf.columns:
        print(gdf['confidence'])

inspect_village_truths('data/34855_vadnerbhairav_chandavad_nashik')
inspect_village_truths('data/12429_malatavadi_chandgad_kolhapur')
