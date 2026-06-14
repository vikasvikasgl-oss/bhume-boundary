import geopandas as gpd
from shapely.geometry import Polygon
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from bhume import load
from bhume.io import read_predictions
from bhume.score import _utm_for, _iou

def print_details(village_dir):
    village = load(village_dir)
    preds = read_predictions(Path(village_dir) / 'predictions.geojson')
    
    utm = _utm_for(village.example_truths.geometry.iloc[0])
    truth_u = village.example_truths.to_crs(utm)
    official_u = village.plots.to_crs(utm)
    pred_u = preds.to_crs(utm)
    
    print(f"=== Details for {village.slug} ===")
    for pn in village.example_truths.index:
        t = truth_u.loc[pn, 'geometry']
        o = official_u.loc[pn, 'geometry']
        iou_official = _iou(o, t)
        
        if pn in pred_u.index:
            p_row = preds.loc[pn]
            p_geom = pred_u.loc[pn, 'geometry']
            iou_pred = _iou(p_geom, t)
            improvement = iou_pred - iou_official
            status = p_row['status']
            conf = p_row.get('confidence')
            note = p_row.get('method_note', '')
            print(f"Plot {pn}: official_IoU={iou_official:.3f}, pred_IoU={iou_pred:.3f}, improvement={improvement:+.3f}, status={status}, conf={conf:.3f}")
            print(f"  Note: {note}")
        else:
            print(f"Plot {pn}: official_IoU={iou_official:.3f}, NOT IN PREDICTIONS")

print_details('data/34855_vadnerbhairav_chandavad_nashik')
print()
print_details('data/12429_malatavadi_chandgad_kolhapur')
