"""Power-outage-resumable validation selection and deployment experiments.

Each complete.json is written atomically only after its predictions/metrics exist.
No test outcomes are used until selection.json is permanently frozen.
"""
from pathlib import Path
import argparse,collections,datetime,gc,hashlib,json,os,platform,random,shutil,sys,time
import cv2,numpy as np,torch,ultralytics,yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from policy import choose,pareto

ROOT=Path(os.environ.get('VIOLENCE_STUDY_ROOT', str(Path(__file__).resolve().parents[2])));HERE=Path(__file__).resolve().parent
DATA=HERE/'data';RESULT=HERE/'results';RESULT.mkdir(exist_ok=True)
CONFIGS=[{'id':f'{s}_{p}','size':s,'half':p=='fp16','rect':False} for s in [320,480,640] for p in ['fp32','fp16']]+[{'id':f'640_rect_{p}','size':640,'half':p=='fp16','rect':True} for p in ['fp32','fp16']]
torch.set_num_threads(4)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 # Windows readers/antivirus can briefly deny delete sharing on the old file.
 # Preserve atomic replacement and retry; never remove the last valid checkpoint.
 for attempt in range(20):
  try:os.replace(temp,path);break
  except PermissionError:
   if attempt==19:raise
   time.sleep(.05*(attempt+1))
def state(stage,detail):
 atomic(HERE/'state.json',{'updated_utc':now(),'stage':stage,'detail':detail,'resume_command':'work/research_env/Scripts/python.exe work/deployment_study/run_study.py --phase all','complete_markers':sorted(str(p.relative_to(HERE)) for p in RESULT.rglob('complete.json'))})
def sync():torch.cuda.synchronize()

def prepare():
 if (DATA/'complete.json').exists():return
 state('prepare','Normalizing object validation rows and excluding exact training/validation overlap')
 obj=ROOT/'work/recovered_data/merged_object_detection'
 train_hashes={sha(f) for f in (obj/'train/images').iterdir() if f.is_file()}
 report={'object':{},'behavior':{}}
 for split,origin in [('val',obj/'valid/images'),('test',ROOT/'work/row_normalized_object_test/test/images')]:
  images=DATA/'object'/split/'images';labels=DATA/'object'/split/'labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
  excluded=[];instances=background=polygons=boxes=0;manifest=[]
  # Validation removes decoded-pixel overlap as well as exact bytes.
  def pixel(f):
   im=cv2.imread(str(f));assert im is not None,str(f)
   return hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
  train_pixels={pixel(f) for f in (obj/'train/images').iterdir() if f.is_file()} if split=='val' else set()
  for f in sorted(origin.iterdir()):
   if not f.is_file() or f.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
   if split=='val' and (sha(f) in train_hashes or pixel(f) in train_pixels):excluded.append(f.name);continue
   lp=(f.parent.parent/'labels'/f.name).with_suffix('.txt');new=[]
   for line in lp.read_text(encoding='utf-8-sig').splitlines():
    if not line.strip():continue
    values=list(map(float,line.split()))
    if len(values)==5:new.append(line);boxes+=1
    else:
     points=np.array(values[1:]).reshape(-1,2);lo=points.min(0);hi=points.max(0);b=[*((lo+hi)/2),*(hi-lo)]
     new.append(str(int(values[0]))+' '+' '.join(f'{x:.12g}' for x in b));polygons+=1
   instances+=len(new);background+=not bool(new)
   shutil.copy2(f,images/f.name);(labels/lp.name).write_text('\n'.join(new)+('\n' if new else ''),encoding='utf-8')
   manifest.append({'name':f.name,'sha256':sha(f),'instances':len(new)})
  report['object'][split]={'images':len(manifest),'instances':instances,'backgrounds':background,'excluded':excluded,'box_rows_preserved':boxes,'polygon_rows_converted':polygons,'manifest':manifest}
 names=['baseball bat','cricket bat','crow bar','hammer','ice pick','long knife','pistol','pocket-knife','rifle','stick']
 for task in ['object','behavior']:
  if task=='object':cfg={'path':(DATA/'object').as_posix(),'train':(obj/'train/images').as_posix(),'val':'val/images','test':'test/images','names':dict(enumerate(names))}
  else:
   beh=ROOT/'work/recovered_behavior_data';cfg={'path':beh.as_posix(),'train':'train/images','val':'valid/images','test':'test/images','names':{0:'violent'}}
   for split,folder in [('val','valid'),('test','test')]:
    fs=sorted((beh/folder/'images').glob('*.jpg'));label=(beh/folder/'labels')
    report['behavior'][split]={'images':len(fs),'instances':sum(len((label/(f.stem+'.txt')).read_text().splitlines()) for f in fs),'backgrounds':sum(not (label/(f.stem+'.txt')).read_text().strip() for f in fs),'manifest':[{'name':f.name,'sha256':sha(f)} for f in fs]}
  (DATA/f'{task}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
 assert report['object']['test']['images']==425 and report['behavior']['test']['images']==115
 atomic(DATA/'complete.json',report)

class StudyValidator(DetectionValidator):
 def build_dataset(self,img_path,mode='val',batch=None):
  dataset=super().build_dataset(img_path,mode,batch)
  if self.args.rect:
   assert self.args.batch==1
   shapes=[]
   for f in dataset.im_files:
    with Image.open(f) as im:w,h=im.size
    # Predictor-aligned minimum stride padding; do not retain validator pad=.5.
    shapes.append(np.ceil(np.array([h,w])*self.args.imgsz/max(h,w)/dataset.stride).astype(int)*dataset.stride)
   dataset.batch_shapes=np.array(shapes);dataset.pad=0.
  return dataset
 def init_metrics(self,model):
  super().init_metrics(model);self.image_records=[];self.study_stats=[]
 def update_metrics(self,preds,batch):
  for si,pred in enumerate(preds):
   gt=self._prepare_batch(si,batch);predn=self._prepare_pred(pred,gt)
   tp=self._process_batch(predn,gt['bbox'],gt['cls']) if len(predn) and len(gt['cls']) else torch.zeros((len(predn),self.niou),dtype=torch.bool,device=self.device)
   image_id=len(self.image_records)
   record={'image':Path(batch['im_file'][si]).name,'targets':len(gt['cls']),'predictions_at_05':int((predn[:,4]>=.5).sum().item()),'background':not bool(len(gt['cls'])),'input_shape':list(batch['img'].shape[-2:])}
   self.image_records.append(record)
   self.study_stats.append({'tp':tp.cpu().numpy(),'conf':predn[:,4].cpu().numpy(),'pred_cls':predn[:,5].cpu().numpy(),'target_cls':gt['cls'].cpu().numpy(),'pred_image':np.full(len(predn),image_id),'target_image':np.full(len(gt['cls']),image_id)})
  super().update_metrics(preds,batch)
 def get_stats(self):
  results=super().get_stats()
  out=self.save_dir
  np.savez_compressed(out/'matching_arrays.npz',**{k:np.concatenate([r[k] for r in self.study_stats],axis=0) for k in self.study_stats[0]})
  atomic(out/'per_image.json',self.image_records)
  return results

def evaluate(task,split,config):
 out=RESULT/'accuracy'/task/split/config['id']
 if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
 if split=='test':assert (RESULT/'selection.json').exists(),'Policy must be frozen before any new test evaluation'
 state('accuracy',f"{task} {split} {config['id']}")
 if out.exists():shutil.rmtree(out) # Verified fixed descendant under experiment results; partial trial only.
 model=YOLO(ROOT/f'{task}_detection/best.pt')
 start=time.perf_counter()
 m=model.val(validator=StudyValidator,data=str(DATA/f'{task}.yaml'),split=split,imgsz=config['size'],half=config['half'],batch=1,workers=0,device=0,conf=.001,iou=.7,max_det=300,rect=config['rect'],augment=False,plots=False,save_txt=False,project=str(out.parent),name=out.name,exist_ok=True,verbose=False)
 records=json.loads((out/'per_image.json').read_text());arrays=np.load(out/'matching_arrays.npz');mask=arrays['conf']>=.5
 tp=int(arrays['tp'][mask,0].sum());fp=int(mask.sum())-tp;fn=len(arrays['target_cls'])-tp
 bg=[r for r in records if r['background']]
 result={'task':task,'split':split,'config':config,'completed_utc':now(),'wall_seconds':time.perf_counter()-start,'checkpoint_sha256':sha(ROOT/f'{task}_detection/best.pt'),'metrics':{k:float(v) for k,v in m.results_dict.items()},'per_class':[{'id':int(c),'ap50':float(m.box.ap50[i]),'ap95':float(m.box.ap[i]),'precision':float(m.box.p[i]),'recall':float(m.box.r[i])} for i,c in enumerate(m.box.ap_class_index)],'images':len(records),'instances':len(arrays['target_cls']),'fixed_confidence_05_iou_05':{'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'background_images':len(bg),'background_images_with_detection':sum(r['predictions_at_05']>0 for r in bg),'background_detection_count':sum(r['predictions_at_05'] for r in bg)},'settings':{'rect':config['rect'],'rect_pad':0. if config['rect'] else None,'batch':1,'confidence_floor':.001,'nms_iou':.7,'max_det':300,'no_test_augmentation':True}}
 atomic(out/'complete.json',result);print('COMPLETE',task,split,config['id'],result['metrics'],flush=True)
 del model,m,arrays;gc.collect();torch.cuda.empty_cache()
 return result

def timing_frames():
 frames=[]
 for f in sorted((ROOT/'Integrated Code/test_videos').glob('*.mp4')):
  cap=cv2.VideoCapture(str(f))
  for _ in range(6):
   ok,frame=cap.read()
   if not ok:break
   frames.append(frame)
  cap.release()
 assert len(frames)==30
 return frames

def branch_cost(task,config):
 out=RESULT/'branch_cost'/task/config['id'];marker=out/'complete.json'
 if marker.exists():return json.loads(marker.read_text())
 state('branch_cost',task+' '+config['id']);out.mkdir(parents=True,exist_ok=True)
 frames=timing_frames();model=YOLO(ROOT/f'{task}_detection/best.pt')
 predict=lambda frame:model.predict(frame,imgsz=config['size'],half=config['half'],device=0,conf=.25,iou=.7,rect=config['rect'],max_det=300,verbose=False)[0]
 for _ in range(5):predict(frames[0])
 repetitions=[]
 for rep in range(3):
  values=[];sync();start=time.perf_counter()
  for frame in frames:
   sync();t=time.perf_counter();r=predict(frame);_=r.boxes.data.cpu().numpy();sync();values.append(1000*(time.perf_counter()-t))
  repetitions.append({'repeat':rep,'total_ms':1000*(time.perf_counter()-start),'per_frame_ms':values})
 allvalues=[v for r in repetitions for v in r['per_frame_ms']]
 result={'task':task,'config':config,'completed_utc':now(),'frames_per_repeat':30,'repeats':3,'mean_ms':float(np.mean(allvalues)),'median_ms':float(np.median(allvalues)),'p95_ms':float(np.percentile(allvalues,95)),'repetitions':repetitions,'boundary':'In-memory BGR input, aspect-preserving square letterbox inside predictor, prediction/NMS and CPU box transfer, GPU synchronization. Excludes decoding, outputs and loading. Dedicated timing inputs are not accuracy ground truth.'}
 atomic(marker,result);print('COST',task,config['id'],result['mean_ms'],flush=True)
 del model;gc.collect();torch.cuda.empty_cache();return result

def selection():
 marker=RESULT/'selection.json'
 if marker.exists():return json.loads(marker.read_text())
 rows={}
 for task in ['object','behavior']:
  rows[task]=[]
  for c in CONFIGS:
   a=json.loads((RESULT/'accuracy'/task/'val'/c['id']/'complete.json').read_text());cost=json.loads((RESULT/'branch_cost'/task/c['id']/'complete.json').read_text())
   rows[task].append({'id':c['id'],'ap50':a['metrics']['metrics/mAP50(B)'],'ap95':a['metrics']['metrics/mAP50-95(B)'],'ms':cost['mean_ms']})
 policies={str(t):choose(rows,{'object':'640_fp32','behavior':'640_fp32'},t) for t in [.90,.95,.975,1.0]}
 result={'frozen_utc':now(),'selection_data':'Validation only; dedicated unlabeled timing inputs; no new test outcomes accessed','primary_retention':.95,'candidates':rows,'pareto':{k:pareto(v) for k,v in rows.items()},'policies':policies,'primary':policies['0.95'],'criterion':'Minimize sum of independently measured branch mean latency subject to AP50 and AP50:95 in EACH branch >= retention*same-branch 640 FP32 validation metric. This retention ratio is an engineering constraint, not a safety guarantee. Joint timing is checked separately.'}
 atomic(marker,result);state('selection_frozen',json.dumps(result['primary']));print('FROZEN',result['primary'],flush=True);return result

def full_profile(selected,policy_overrides=None,block='primary'):
 policies={'original_640_fp32':{'object':'640_fp32','behavior':'640_fp32','geometry':'stretch'},'uniform_640_fp32':{'object':'640_fp32','behavior':'640_fp32','geometry':'letterbox'},'uniform_640_fp16':{'object':'640_fp16','behavior':'640_fp16','geometry':'letterbox'},'uniform_480_fp16':{'object':'480_fp16','behavior':'480_fp16','geometry':'letterbox'},'uniform_640_rect_fp32':{'object':'640_rect_fp32','behavior':'640_rect_fp32','geometry':'letterbox'},'uniform_640_rect_fp16':{'object':'640_rect_fp16','behavior':'640_rect_fp16','geometry':'letterbox'},'selected_095':{'object':selected['primary']['object']['id'],'behavior':selected['primary']['behavior']['id'],'geometry':'letterbox'}}
 if policy_overrides is not None:policies=policy_overrides
 order_seed=20261003 if block=='primary' else 20261004
 trials=[(name,rep) for rep in range(3) for name in policies];random.Random(order_seed).shuffle(trials)
 videos=sorted((ROOT/'Integrated Code/test_videos').glob('*.mp4'));lookup={c['id']:c for c in CONFIGS}
 for name,rep in trials:
  out=RESULT/'full_profile'/name/f'repeat_{rep}';marker=out/'complete.json'
  if marker.exists():continue
  state('full_profile',name+' repeat '+str(rep));out.mkdir(parents=True,exist_ok=True)
  cfg=policies[name];models={task:YOLO(ROOT/f'{task}_detection/best.pt') for task in ['object','behavior']}
  def detect(task,frame):
   c=lookup[cfg[task]];im=cv2.resize(frame,(c['size'],c['size'])) if cfg['geometry']=='stretch' else frame
   r=models[task].predict(im,imgsz=c['size'],half=c['half'],device=0,conf=.25,iou=.7,rect=c['rect'],max_det=300,verbose=False)[0]
   boxes=r.boxes.data.cpu().numpy();boxes=boxes[boxes[:,4]>=.5]
   if cfg['geometry']=='stretch' and len(boxes):boxes[:,[0,2]]*=frame.shape[1]/c['size'];boxes[:,[1,3]]*=frame.shape[0]/c['size']
   if task=='object' and len(boxes):
    keep=cv2.dnn.NMSBoxes([[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])] for b in boxes],boxes[:,4].tolist(),.499999,.4)
    boxes=boxes[np.array(keep).reshape(-1)] if len(keep) else boxes[:0]
   return boxes
  clip_results=[];torch.cuda.reset_peak_memory_stats()
  for video in videos:
   cap=cv2.VideoCapture(str(video));ok,first=cap.read();cap.release();assert ok
   for _ in range(5):detect('object',first);detect('behavior',first)
   cap=cv2.VideoCapture(str(video));writer=cv2.VideoWriter(str(out/(video.stem+'.avi')),cv2.VideoWriter_fourcc(*'MJPG'),25,(first.shape[1],first.shape[0]));assert writer.isOpened()
   logs=(out/(video.stem+'.txt')).open('w',encoding='utf-8');values=[];screenshots=0;count=0
   sync();start=time.perf_counter()
   while count<50:
    sync();t=time.perf_counter();ok,frame=cap.read()
    if not ok:break
    ob=detect('object',frame);be=detect('behavior',frame);state_id=2*int(bool(len(ob)))+int(bool(len(be)))
    logs.write(f'{count},{state_id},{len(ob)},{len(be)}\n')
    if len(be):
     assert cv2.imwrite(str(out/f'{video.stem}_{count:04d}.jpg'),frame);screenshots+=1
    for boxes,color in [(ob,(0,0,255)),(be,(255,0,0))]:
     for b in boxes:
      x1,y1,x2,y2=map(int,b[:4]);cv2.rectangle(frame,(x1,y1),(x2,y2),color,2);cv2.putText(frame,f'{int(b[5])}: {b[4]:.2f}',(x1,max(y1-5,10)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    cv2.putText(frame,f'Cue state: {state_id}',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2);writer.write(frame)
    sync();values.append(1000*(time.perf_counter()-t));count+=1
   logs.flush();logs.close();writer.release();cap.release();sync();elapsed=time.perf_counter()-start
   clip_results.append({'video':video.name,'frames':count,'seconds':elapsed,'fps':count/elapsed,'per_frame_ms':values,'snapshots':screenshots})
   atomic(out/'progress.json',clip_results)
  total_frames=sum(c['frames'] for c in clip_results);total_sec=sum(c['seconds'] for c in clip_results);values=[v for c in clip_results for v in c['per_frame_ms']]
  result={'name':name,'repeat':rep,'config':cfg,'completed_utc':now(),'frames':total_frames,'seconds':total_sec,'aggregate_fps':total_frames/total_sec,'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'clips':clip_results,'boundary':'Decode, aspect handling, two sequential predictions/NMS, retain conf>=.5, extra object class-agnostic NMS .4, CPU transfer, text log, behavior-triggered raw JPEG snapshot, box/text overlays, MJPG video write, log flush and writer finalization, GPU synchronization. Excludes model load, capture/writer opening, five untimed warm-ups per clip, GUI display. Offline full-output path, not live deployment.'}
  result['measurement_block']=block;result['trial_order_seed']=order_seed
  atomic(marker,result);print('FULL PROFILE',name,rep,result['aggregate_fps'],flush=True)
  del models;gc.collect();torch.cuda.empty_cache()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','prepare','validation','cost','select','test','profile'],default='all');args=parser.parse_args()
 protocol={'created_utc':now(),'configs':CONFIGS,'primary_retention':.95,'retention_sensitivity':[.90,.95,.975,1.0],'validation_selection_only':True,'object_validation_exact_overlap_exclusion':True,'accuracy_preprocessing':'Aspect-preserving square letterbox, batch1, rect=False, 0.001conf, 0.7NMS, 300maxdet, noTTA','fixed_threshold_analysis':{'confidence':.5,'matching_iou':.5},'accuracy_targets':'Bounding boxes in each branch; no incident intent labels and no pooled joint-event score','profile_repeats':3,'profile_trial_order_seed':20261003,'script_sha256':sha(__file__),'policy_sha256':sha(HERE/'policy.py'),'runtime':{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'opencv':cv2.__version__,'os':platform.platform()}}
 if not (HERE/'protocol.json').exists():atomic(HERE/'protocol.json',protocol)
 prepare()
 if args.phase in ['all','validation']:
  for c in CONFIGS:
   for task in ['object','behavior']:evaluate(task,'val',c)
 if args.phase in ['all','cost']:
  candidates=[(task,c) for task in ['object','behavior'] for c in CONFIGS];random.Random(20261003).shuffle(candidates)
  for task,c in candidates:branch_cost(task,c)
 if args.phase in ['all','select']:selected=selection()
 if args.phase in ['all','test']:
  assert (RESULT/'selection.json').exists()
  for c in CONFIGS:
   for task in ['object','behavior']:evaluate(task,'test',c)
 if args.phase in ['all','profile']:full_profile(json.loads((RESULT/'selection.json').read_text()))
 state('phase_complete',args.phase)
if __name__=='__main__':main()
