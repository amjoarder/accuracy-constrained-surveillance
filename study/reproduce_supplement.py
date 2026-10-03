"""Recompute supplementary AP, timing and external counts using NumPy only.

Usage: python study/reproduce_supplement.py
Published arrays include the original evaluation ranking for portable confidence-tie reproduction.
No GPU, source images, videos, model weights or Ultralytics installation required.
"""
from pathlib import Path
import argparse,json
import numpy as np

def ap_metrics(a):
 order=a.get('evaluation_order')
 if order is None:
  if np.__version__!='1.26.4':raise RuntimeError('Legacy arrays require NumPy 1.26.4')
  order=np.argsort(-a['conf'])
 assert np.array_equal(np.sort(order),np.arange(len(a['conf'])))
 assert np.all(np.diff(a['conf'][order])<=0)
 tp=a['tp'][order];pred=a['pred_cls'][order]
 classes,counts=np.unique(a['target_cls'],return_counts=True);aps=np.zeros((len(classes),10))
 for i,(classid,count) in enumerate(zip(classes,counts)):
  matched=tp[pred==classid]
  if not len(matched):continue
  tpc=matched.cumsum(0);fpc=(1-matched).cumsum(0)
  recall=tpc/(count+1e-16);precision=tpc/(tpc+fpc)
  for j in range(10):
   mr=np.concatenate(([0.],recall[:,j],[1.]));mp=np.concatenate(([1.],precision[:,j],[0.]))
   mp=np.flip(np.maximum.accumulate(np.flip(mp)))
   integrate=getattr(np,'trapezoid',np.trapz);aps[i,j]=integrate(np.interp(np.linspace(0,1,101),mr,mp),np.linspace(0,1,101))
 return np.array([aps[:,0].mean(),aps.mean()])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--supplement',type=Path,default=Path(__file__).resolve().parent/'results');args=parser.parse_args();base=args.supplement
 analysis=json.loads((base/'analysis.json').read_text());checks=0
 trials=list((base/'accuracy').glob('*/*/*'))+list((base/'predictor_check').glob('*/*'))
 for trial in trials:
  record=json.loads((trial/'complete.json').read_text());a=dict(np.load(trial/'matching_arrays.npz'));actual=ap_metrics(a)
  expected=[record['metrics']['metrics/mAP50(B)'],record['metrics']['metrics/mAP50-95(B)']]
  assert np.allclose(actual,expected,atol=1e-10,rtol=0),(trial,actual,expected);checks+=1
 assert checks==37
 for policy,p in analysis['full_profile'].items():
  rows=[json.loads((base/'full_profile'/policy/f'repeat_{i}/complete.json').read_text()) for i in range(3)]
  assert sum(r['frames'] for r in rows)==717
  assert abs(p['pooled_fps']-717/sum(r['seconds'] for r in rows))<1e-10
 external=json.loads((base/'external_airtlab/summary.json').read_text());clips=0
 for policy in external['policies']:
  rows=[json.loads(f.read_text()) for f in (base/'external_airtlab'/policy).glob('*.json')];assert len(rows)==350;clips+=len(rows)
  truth=np.array([r['label'] for r in rows])
  for cue,expected in external['summaries'][policy]['all'].items():
   pred=np.array([r['scores'][cue]>=.5 for r in rows]);tp=int(np.sum(pred&(truth==1)));fp=int(np.sum(pred&(truth==0)));fn=int(np.sum(~pred&(truth==1)));tn=int(np.sum(~pred&(truth==0)))
   assert [tp,fp,fn,tn]==[expected[k] for k in ['TP','FP','FN','TN']]
   assert abs(.5*(tp/(tp+fn)+tn/(tn+fp))-expected['balanced_accuracy'])<1e-12
   scores=np.array([r['scores'][cue] for r in rows]);pos=scores[truth==1];neg=scores[truth==0]
   auc=np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:]));assert abs(auc-expected['ROC_AUC'])<1e-12
  for r in rows:
   values=r['samples'];computed={'object_proxy':max(v['object'] for v in values),'behavior':max(v['behavior'] for v in values),'OR':max(max(v['object'],v['behavior']) for v in values),'AND':max(min(v['object'],v['behavior']) for v in values)}
   assert computed==r['scores']
   assert [v['frame'] for v in values]==np.rint(np.linspace(0,r['frame_count']-1,3)).astype(int).tolist()
 print(json.dumps({'AP_trials_recomputed':checks,'grid_trials':32,'native_predictor_trials':5,'timing_policies_recomputed':len(analysis['full_profile']),'external_records_checked':clips,'passed':True},indent=2))
if __name__=='__main__':main()
