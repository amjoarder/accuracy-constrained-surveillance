from pathlib import Path
import collections,datetime,gc,hashlib,json,os,time
import cv2,numpy as np,torch
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from run_study import ROOT,HERE,RESULT,atomic,state,CONFIGS

EXTERNAL=RESULT/'external_airtlab';EXTERNAL.mkdir(exist_ok=True)
def scores_summary(records):
 truth=np.array([r['label'] for r in records]);out={}
 for key in ['object_proxy','behavior','OR','AND']:
  scores=np.array([r['scores'][key] for r in records]);pred=scores>=.5
  tp=int(np.sum(pred&(truth==1)));fp=int(np.sum(pred&(truth==0)));fn=int(np.sum(~pred&(truth==1)));tn=int(np.sum(~pred&(truth==0)))
  out[key]={'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn),'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'accuracy':(tp+tn)/len(truth),'ROC_AUC':float(roc_auc_score(truth,scores)),'false_positive_fraction':fp/(fp+tn)}
 return out

def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__ or 'Research utility; see reproduction instructions for inputs.')
 parser.parse_args()
 assert (HERE/'external_manifest.json').exists()
 assert len(list((RESULT/'full_profile').rglob('complete.json')))>=21,'Finish primary timing before external inference'
 selected=json.loads((RESULT/'selection.json').read_text())['primary'];configs={c['id']:c for c in CONFIGS}
 policies={'reference':{'object':'640_fp32','behavior':'640_fp32'},'selected':{task:selected[task]['id'] for task in ['object','behavior']},'original_stretch':{'object':'640_fp32','behavior':'640_fp32'}}
 manifest=json.loads((HERE/'external_manifest.json').read_text());files=manifest['files'];assert len(files)==350
 torch.set_num_threads(4)
 for policy,pair in policies.items():
  destination=EXTERNAL/policy;destination.mkdir(exist_ok=True)
  models={task:YOLO(ROOT/f'{task}_detection/best.pt') for task in ['object','behavior']}
  for i,file in enumerate(files):
   rel=Path(file['path']);camera=rel.parent.name;label=int('non-violent' not in rel.parts);name=f'{label}_{camera}_{rel.stem}';marker=destination/(name+'.json')
   if marker.exists():continue
   state('external_transfer',f'{policy} {i+1}/350')
   path=HERE/'external_airtlab'/rel;assert hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
   cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert n>=3
   indices=np.rint(np.linspace(0,n-1,3)).astype(int);samples=[];started=time.perf_counter()
   for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read();assert ok,str(path)
    values={}
    for task in ['object','behavior']:
     c=configs[pair[task]];image=cv2.resize(frame,(640,640)) if policy=='original_stretch' else frame
     r=models[task].predict(image,imgsz=c['size'],half=c['half'],rect=c['rect'],device=0,conf=.001,iou=.7,max_det=300,verbose=False)[0]
     values[task]=float(r.boxes.conf.max().item()) if len(r.boxes) else 0.
    samples.append({'frame':int(index),'object':values['object'],'behavior':values['behavior']})
   cap.release()
   scores={'object_proxy':max(r['object'] for r in samples),'behavior':max(r['behavior'] for r in samples),'OR':max(max(r['object'],r['behavior']) for r in samples),'AND':max(min(r['object'],r['behavior']) for r in samples)}
   atomic(marker,{'repository_path':file['path'],'label':label,'camera':camera,'scene':f'{label}_{rel.stem}','policy':policy,'settings':pair,'video_sha256':file['sha256'],'frame_count':n,'decoded_shape':list(frame.shape[:2]),'samples':samples,'scores':scores,'wall_seconds':time.perf_counter()-started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
   if (i+1)%25==0:print(policy,i+1,'/350',flush=True)
  del models;gc.collect();torch.cuda.empty_cache()
 outputs={policy:sorted([json.loads(f.read_text()) for f in (EXTERNAL/policy).glob('*.json')],key=lambda r:r['repository_path']) for policy in policies}
 assert all(len(rows)==350 for rows in outputs.values())
 assert [r['repository_path'] for r in outputs['reference']]==[r['repository_path'] for r in outputs['selected']]
 summaries={p:{'all':scores_summary(rows),'by_camera':{cam:scores_summary([r for r in rows if r['camera']==cam]) for cam in ['cam1','cam2']}} for p,rows in outputs.items()}
 scene_indices=collections.defaultdict(list)
 for i,r in enumerate(outputs['reference']):scene_indices[r['scene']].append(i)
 assert len(scene_indices)==175 and all(len(indices)==2 for indices in scene_indices.values())
 scenes=list(scene_indices);rng=np.random.default_rng(20261003);differences={k:[] for k in ['object_proxy','behavior','OR','AND']}
 for _ in range(400):
  chosen=rng.integers(0,175,175);indices=[i for c in chosen for i in scene_indices[scenes[c]]]
  for key in differences:
   metrics=[]
   for policy in ['reference','selected']:
    rows=outputs[policy];truth=np.array([rows[i]['label'] for i in indices]);pred=np.array([rows[i]['scores'][key]>=.5 for i in indices]);metrics.append(float(balanced_accuracy_score(truth,pred)))
   differences[key].append(metrics[1]-metrics[0])
 intervals={k:(100*np.percentile(v,[2.5,97.5])).tolist() for k,v in differences.items()}
 report={'dataset':'AIRTLab','repository':manifest['repository'],'revision':manifest['revision'],'clips':350,'positive_clips':230,'negative_clips':120,'acted_scene_clusters':175,'frames_per_clip':3,'policies':policies,'threshold':.5,'no_training_or_benchmark_based_tuning':True,'summaries':summaries,'paired_scene_cluster_bootstrap_balanced_accuracy_difference_pp':intervals,'bootstrap_replicates':400,'limitations':'Sparse frame-pooled clip screening, not learned temporal recognition, event localization, or camera-hour false alarms. Known paired camera views are clustered; unknown actor/session dependence and original training-source overlap remain possible. Object proxy does not define violent ground truth.'}
 atomic(EXTERNAL/'summary.json',report);state('external_complete','350 clips,175 paired-camera scene groups; three fixed policies complete');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
