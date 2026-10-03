from pathlib import Path
import json,numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
checked=0;differences=[];hist={}
for task in ['object','behavior']:
 for split in ['val','test']:
  records=json.loads((HERE/'results/accuracy'/task/split/'640_rect_fp32/per_image.json').read_text())
  source=HERE/'data/object'/split/'images' if task=='object' else ROOT/'work/recovered_behavior_data'/('valid' if split=='val' else 'test')/'images'
  histogram={}
  for record in records:
   with Image.open(source/record['image']) as im:w,h=im.size
   ratio=640/max(w,h);resized=np.array([round(h*ratio),round(w*ratio)]);expected=(np.ceil(resized/32).astype(int)*32).tolist()
   if record['input_shape']!=expected:differences.append({'task':task,'split':split,'image':record['image'],'native':[h,w],'validation':record['input_shape'],'predictor_expected':expected})
   histogram[str(record['input_shape'])]=histogram.get(str(record['input_shape']),0)+1;checked+=1
  hist[task+'_'+split]=histogram
report={'checked_images':checked,'shape_mismatches':differences,'input_shape_histograms':hist,'scope':'Minimum-stride tensor-shape comparison to predictor resize rounding. Not bitwise interpolation equivalence.'}
(HERE/'geometry_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'checked':checked,'mismatches':len(differences),'histograms':hist},indent=2))
assert not differences
