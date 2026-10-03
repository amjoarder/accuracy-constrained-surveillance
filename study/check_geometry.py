"""Verify rectangular predictor shapes from published source-image dimensions."""
from pathlib import Path
import argparse,json
import numpy as np
HERE=Path(__file__).resolve().parent
def verify(base=HERE):
 geometry=json.loads((base/'manifests/image_geometry.json').read_text());checked=0;differences=[];hist={}
 for task in ['object','behavior']:
  for split in ['val','test']:
   records=json.loads((base/'results/accuracy'/task/split/'640_rect_fp32/per_image.json').read_text())
   shapes=geometry[task+'_'+split];assert set(shapes)=={r['image'] for r in records};histogram={}
   for record in records:
    h,w=shapes[record['image']];assert h>0 and w>0
    ratio=640/max(w,h);resized=np.array([round(h*ratio),round(w*ratio)]);expected=(np.ceil(resized/32).astype(int)*32).tolist()
    if record['input_shape']!=expected:differences.append({'task':task,'split':split,'image':record['image'],'native':[h,w],'validation':record['input_shape'],'predictor_expected':expected})
    histogram[str(record['input_shape'])]=histogram.get(str(record['input_shape']),0)+1;checked+=1
   hist[task+'_'+split]=histogram
 return {'checked_images':checked,'shape_mismatches':differences,'input_shape_histograms':hist,'scope':'Minimum-stride tensor-shape comparison from published native dimensions, read from original imagery before publication. Not bitwise interpolation equivalence.'}
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,help='Optional report path; default does not modify published records');args=parser.parse_args()
 report=verify()
 if args.output:args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'checked':report['checked_images'],'mismatches':len(report['shape_mismatches']),'histograms':report['input_shape_histograms']},indent=2))
 assert report['checked_images']==1627 and not report['shape_mismatches']
if __name__=='__main__':main()
