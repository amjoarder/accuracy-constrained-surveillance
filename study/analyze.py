from pathlib import Path
import json,math,hashlib,collections,itertools
import numpy as np
from ultralytics.utils.metrics import ap_per_class
from run_study import ROOT,HERE,RESULT,atomic,state,CONFIGS
from policy import choose

def metrics(arrays,weights=None):
 if weights is None:
  pred=np.arange(len(arrays['conf']));targets=np.arange(len(arrays['target_cls']))
 else:
  pred=np.repeat(np.arange(len(arrays['conf'])),weights[arrays['pred_image'].astype(int)])
  targets=np.repeat(np.arange(len(arrays['target_cls'])),weights[arrays['target_image'].astype(int)])
 if not len(targets):return None
 values=ap_per_class(arrays['tp'][pred],arrays['conf'][pred],arrays['pred_cls'][pred],arrays['target_cls'][targets],plot=False)
 ap=values[5]
 return np.array([ap[:,0].mean(),ap.mean()])

def paired_interval(task,selected,reps=400):
 paths=[RESULT/'accuracy'/task/'test'/c for c in ['640_fp32',selected]]
 arrays=[dict(np.load(p/'matching_arrays.npz')) for p in paths]
 records=json.loads((paths[0]/'per_image.json').read_text());n=len(records)
 other=json.loads((paths[1]/'per_image.json').read_text());index={r['image']:i for i,r in enumerate(records)}
 assert set(index)=={r['image'] for r in other}
 remap=np.array([index[r['image']] for r in other])
 for key in ['pred_image','target_image']:arrays[1][key]=remap[arrays[1][key].astype(int)]
 estimates=[]
 for p,a in zip(paths,arrays):
  m=metrics(a);stored=json.loads((p/'complete.json').read_text())['metrics']
  assert np.allclose(m,[stored['metrics/mAP50(B)'],stored['metrics/mAP50-95(B)']],atol=1e-10)
  estimates.append(m)
 rng=np.random.default_rng(20261003);differences=[]
 for _ in range(reps):
  weights=np.bincount(rng.integers(0,n,n),minlength=n)
  a,b=[metrics(x,weights) for x in arrays]
  if a is not None and b is not None:differences.append(b-a)
 return {'configuration':selected,'baseline':'640_fp32','resamples':reps,'seed':20261003,'images':n,'difference_AP50_percentage_points':float(100*(estimates[1][0]-estimates[0][0])),'difference_AP95_percentage_points':float(100*(estimates[1][1]-estimates[0][1])),'conditional_image_bootstrap_interval_AP50_pp':(100*np.percentile(np.array(differences)[:,0],[2.5,97.5])).tolist(),'conditional_image_bootstrap_interval_AP95_pp':(100*np.percentile(np.array(differences)[:,1],[2.5,97.5])).tolist(),'interpretation':'Paired resampling of images with fixed per-image matches. Conditional image-set sensitivity only; not recording-independent population intervals or evidence that source leakage is absent.'}

def main():
 state('analysis','Aggregating completed accuracy/timing trials and conditional image-resampling sensitivity')
 selection=json.loads((RESULT/'selection.json').read_text());accuracy={};cost={};full={}
 for task in ['object','behavior']:
  accuracy[task]={};cost[task]={}
  for split in ['val','test']:
   accuracy[task][split]={c['id']:json.loads((RESULT/'accuracy'/task/split/c['id']/'complete.json').read_text()) for c in CONFIGS}
  for c in CONFIGS:cost[task][c['id']]=json.loads((RESULT/'branch_cost'/task/c['id']/'complete.json').read_text())
 for policy in sorted((RESULT/'full_profile').iterdir()):
  repeats=[json.loads((policy/f'repeat_{r}'/'complete.json').read_text()) for r in range(3)]
  assert all(r['frames']==239 for r in repeats)
  latencies=[v for r in repeats for clip in r['clips'] for v in clip['per_frame_ms']]
  full[policy.name]={'frames':717,'repeats':3,'fps_by_repeat':[r['aggregate_fps'] for r in repeats],'pooled_fps':717/sum(r['seconds'] for r in repeats),'mean_repeat_fps':float(np.mean([r['aggregate_fps'] for r in repeats])),'sd_repeat_fps':float(np.std([r['aggregate_fps'] for r in repeats],ddof=1)),'median_frame_ms':float(np.median(latencies)),'p95_frame_ms':float(np.percentile(latencies,95)),'peak_allocated_mib':max(r['peak_allocated_mib'] for r in repeats),'peak_reserved_mib':max(r['peak_reserved_mib'] for r in repeats),'snapshots':sum(clip['snapshots'] for r in repeats for clip in r['clips']),'config':repeats[0]['config']}
 transfer={}
 rows=selection['candidates'];refs={t:next(r for r in rows[t] if r['id']=='640_fp32') for t in rows}
 ap50_rows={t:[dict(r,ap95=1.) for r in rows[t]] for t in rows}
 ap50_only=choose(ap50_rows,{'object':'640_fp32','behavior':'640_fp32'},.95)
 pairs=list(itertools.product(rows['object'],rows['behavior']))
 feasible=[pair for pair in pairs if all(sum(r[key]/refs[t][key] for t,r in zip(['object','behavior'],pair))/2>=.95 for key in ['ap50','ap95'])]
 pooled_pair=min(feasible,key=lambda pair:sum(r['ms'] for r in pair))
 ablations={'AP50_only':{t:ap50_only[t]['id'] for t in rows},'pooled_retention':dict(zip(['object','behavior'],[r['id'] for r in pooled_pair]))}
 guarded={}
 for task in ['object','behavior']:
  reference={r['id']:r for r in accuracy[task]['val']['640_fp32']['per_class']}
  feasible=[]
  for c in CONFIGS:
   classes={r['id']:r for r in accuracy[task]['val'][c['id']]['per_class']}
   if all(classes[classid][metric]+1e-12>=.95*ref[metric] for classid,ref in reference.items() for metric in ['ap50','ap95'] if ref[metric]>0):feasible.append(c['id'])
  guarded[task]=min(feasible,key=lambda cid:cost[task][cid]['mean_ms'])
 ablations['class_guard']=guarded
 for tolerance,policy in selection['policies'].items():
  transfer[tolerance]={}
  for task in ['object','behavior']:
   chosen=accuracy[task]['test'][policy[task]['id']]['metrics'];base=accuracy[task]['test']['640_fp32']['metrics']
   ratio={key:chosen[key]/base[key] for key in ['metrics/mAP50(B)','metrics/mAP50-95(B)']}
   transfer[tolerance][task]={'config':policy[task]['id'],'test_retention':ratio,'passes_same_test_retention':all(r>=float(tolerance) for r in ratio.values())}
 bootstrap={}
 for task in ['object','behavior']:
  marker=RESULT/f'bootstrap_{task}.json'
  if marker.exists():bootstrap[task]=json.loads(marker.read_text())
  else:
   bootstrap[task]=paired_interval(task,selection['primary'][task]['id']);atomic(marker,bootstrap[task])
 summary={'selection':selection,'accuracy':accuracy,'branch_cost':cost,'full_profile':full,'retention_transfer':transfer,'constraint_ablations':ablations,'bootstrap':bootstrap,'primary_speedup_vs_uniform640fp32':full['selected_095']['pooled_fps']/full['uniform_640_fp32']['pooled_fps'],'precision_only_speedup':full['uniform_640_fp16']['pooled_fps']/full['uniform_640_fp32']['pooled_fps'],'scope':'No retraining. Task-specific box accuracy and measured sequential offline-output cost; not joint violence/event recognition. Test results are not used to revise the frozen selection.'}
 atomic(RESULT/'analysis.json',summary);state('analysis_complete','Results aggregated; proceed to scientific review and manuscript/figures')
 print(json.dumps({'selection':selection['primary'],'transfer':transfer['0.95'],'profiles':full,'bootstrap':bootstrap},indent=2))
if __name__=='__main__':main()
