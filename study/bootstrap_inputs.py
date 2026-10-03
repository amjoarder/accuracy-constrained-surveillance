"""Prepare the supplied archives for rerunning the deployment study, without modifying originals."""
from pathlib import Path
import argparse,hashlib,json,zipfile,shutil
import cv2,numpy as np

def extract(archive,dest):
 dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   rel=Path(info.filename)
   assert not rel.is_absolute() and '..' not in rel.parts
   target=dest/rel
   assert target.resolve().is_relative_to(dest.resolve())
   if info.is_dir():continue
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():target.write_bytes(z.read(info))

def pixels(path):
 im=cv2.imread(str(path));assert im is not None
 return hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workspace-root',type=Path,required=True);args=parser.parse_args();root=args.workspace_root.resolve()
 assert (root/'object_detection/best.pt').exists() and (root/'behavior_detection/best.pt').exists()
 extract(root/'merged_object_detection.zip',root/'work/recovered_data')
 extract(root/'Violent Behavior Detection.v1i.yolov11.zip',root/'work/recovered_behavior_data')
 obj=root/'work/recovered_data/merged_object_detection';forbidden=set()
 for split in ['train','valid']:
  for f in (obj/split/'images').iterdir():
   if f.is_file():forbidden.add(pixels(f))
 dest=root/'work/row_normalized_object_test/test';(dest/'images').mkdir(parents=True,exist_ok=True);(dest/'labels').mkdir(parents=True,exist_ok=True)
 retained=[];excluded=[]
 for f in sorted((obj/'test/images').iterdir()):
  if not f.is_file():continue
  if pixels(f) in forbidden:excluded.append(f.name);continue
  rows=[];lp=(f.parent.parent/'labels'/f.name).with_suffix('.txt')
  for line in lp.read_text().splitlines():
   if not line.strip():continue
   v=list(map(float,line.split()))
   if len(v)==5:rows.append(line)
   else:
    points=np.array(v[1:]).reshape(-1,2);lo=points.min(0);hi=points.max(0)
    rows.append(str(int(v[0]))+' '+' '.join(f'{x:.12g}' for x in [*((lo+hi)/2),*(hi-lo)]))
  shutil.copy2(f,dest/'images'/f.name);(dest/'labels'/lp.name).write_text('\n'.join(rows)+('\n' if rows else ''))
  retained.append(f.name)
 assert len(retained)==425 and len(excluded)==5
 print(json.dumps({'object_test_retained':len(retained),'object_test_excluded':excluded,'workspace':str(root)},indent=2))
if __name__=='__main__':main()
