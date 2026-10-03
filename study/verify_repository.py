"""Verify every tracked release file, JSON, numeric archive and Python source."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 argparse.ArgumentParser(description=__doc__).parse_args()
 manifest=json.loads((ROOT/'SHA256_MANIFEST.json').read_text());counts={'files':0,'json':0,'npz':0,'python':0,'pdf':0}
 for name,expected in manifest.items():
  path=ROOT/name;assert path.resolve().is_relative_to(ROOT.resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,('Checksum mismatch',name);counts['files']+=1
  if path.suffix=='.json':json.loads(path.read_text(encoding='utf-8'));counts['json']+=1
  elif path.suffix=='.py':compile(path.read_text(encoding='utf-8'),name,'exec');counts['python']+=1
  elif path.suffix=='.npz':
   a=dict(np.load(path,allow_pickle=False));assert len(a['conf'])==len(a['tp'])==len(a['pred_cls']) and a['tp'].shape[1]==10
   assert np.array_equal(np.sort(a['evaluation_order']),np.arange(len(a['conf'])));assert np.all(np.diff(a['conf'][a['evaluation_order']])<=0);counts['npz']+=1
  elif path.suffix=='.pdf':assert path.read_bytes().startswith(b'%PDF-');counts['pdf']+=1
 print(json.dumps({'verified':counts,'passed':True},indent=2))
if __name__=='__main__':main()
