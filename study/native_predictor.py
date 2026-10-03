"""Post-freeze diagnostic: AP at the actual native-image prediction interface.

Four reference/selected test evaluations; no selection or threshold tuning.
Uses the pinned validator's own IoU matching and AP routines.
"""
from pathlib import Path
import gc,json,hashlib,sys
import cv2,numpy as np,torch,yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class
from run_study import ROOT,HERE,RESULT,atomic,state,CONFIGS,now,sha

def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__ or 'Research utility; see reproduction instructions for inputs.')
 parser.add_argument('--include-strict',action='store_true')
 parser.parse_args()
 assert len(list((RESULT/'full_profile').rglob('complete.json')))>=21
 pair=json.loads((RESULT/'selection.json').read_text())['primary'];configs={c['id']:c for c in CONFIGS};summary={}
 for task in ['object','behavior']:
  summary[task]={}
  dataset=yaml.safe_load((HERE/'data'/f'{task}.yaml').read_text());images=Path(dataset['path'])/dataset['test'];files=sorted(f for f in images.iterdir() if f.suffix.lower() in ['.jpg','.jpeg','.png'])
  cases=[('reference','640_fp32'),('selected',pair[task]['id'])]
  if '--include-strict' in sys.argv and task=='object':cases.append(('strict_100',json.loads((RESULT/'selection.json').read_text())['policies']['1.0'][task]['id']))
  for policy,cid in cases:
   out=RESULT/'predictor_check'/task/policy;out.mkdir(parents=True,exist_ok=True);marker=out/'complete.json'
   if marker.exists():summary[task][policy]=json.loads(marker.read_text());continue
   state('native_predictor',task+' '+policy)
   model=YOLO(ROOT/f'{task}_detection/best.pt');matcher=DetectionValidator();matcher.iouv=torch.linspace(.5,.95,10);c=configs[cid];stats={k:[] for k in ['tp','conf','pred_cls','target_cls','pred_image','target_image']};records=[]
   for i,path in enumerate(files):
    im=cv2.imread(str(path));assert im is not None;h,w=im.shape[:2]
    lp=(images.parent/'labels'/path.name).with_suffix('.txt')
    rows=[list(map(float,line.split())) for line in lp.read_text().splitlines() if line.strip()];assert all(len(r)==5 for r in rows)
    gt=np.array(rows,dtype=np.float32).reshape(-1,5);xyxy=np.zeros((len(gt),4),dtype=np.float32)
    if len(gt):
     xyxy[:,0]=(gt[:,1]-gt[:,3]/2)*w;xyxy[:,1]=(gt[:,2]-gt[:,4]/2)*h;xyxy[:,2]=(gt[:,1]+gt[:,3]/2)*w;xyxy[:,3]=(gt[:,2]+gt[:,4]/2)*h
    result=model.predict(im,imgsz=c['size'],rect=c['rect'],half=c['half'],conf=.001,iou=.7,max_det=300,device=0,verbose=False)[0];detections=result.boxes.data.cpu()
    tp=matcher._process_batch(detections,torch.from_numpy(xyxy),torch.from_numpy(gt[:,0])) if len(gt) and len(detections) else torch.zeros((len(detections),10),dtype=torch.bool)
    stats['tp'].append(tp.numpy());stats['conf'].append(detections[:,4].numpy());stats['pred_cls'].append(detections[:,5].numpy());stats['target_cls'].append(gt[:,0]);stats['pred_image'].append(np.full(len(detections),i));stats['target_image'].append(np.full(len(gt),i))
    records.append({'image':path.name,'targets':len(gt),'predictions':len(detections),'original_shape':[h,w]})
   arrays={k:np.concatenate(v,axis=0) for k,v in stats.items()};values=ap_per_class(arrays['tp'],arrays['conf'],arrays['pred_cls'],arrays['target_cls'],plot=False);ap=values[5]
   report={'task':task,'policy':policy,'configuration':cid,'images':len(files),'instances':int(len(arrays['target_cls'])),'metrics':{'metrics/mAP50(B)':float(ap[:,0].mean()),'metrics/mAP50-95(B)':float(ap.mean())},'checkpoint_sha256':sha(ROOT/f'{task}_detection/best.pt'),'completed_utc':now(),'selection_unchanged':True,'scope':'Post-freeze test diagnostic using native cv2 images and actual predictor preprocessing, pinned library IoU matching/AP; no output-stage extra object NMS.'}
   np.savez_compressed(out/'matching_arrays.npz',**arrays);atomic(out/'per_image.json',records);atomic(marker,report);summary[task][policy]=report;print(task,policy,report['metrics'],flush=True)
   del model;gc.collect();torch.cuda.empty_cache()
 atomic(RESULT/'predictor_check/summary.json',summary)
if __name__=='__main__':main()
