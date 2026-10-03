from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,time
HERE=Path(__file__).resolve().parent;DEST=HERE/'external_airtlab'
REPO='airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos'
revision=json.loads((HERE/'external_manifest.json').read_text())['revision']

def get(row):
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 target=DEST/rel;assert target.resolve().is_relative_to(DEST.resolve());target.parent.mkdir(parents=True,exist_ok=True)
 def valid():return target.exists() and target.stat().st_size==row['size'] and hashlib.sha1(f"blob {row['size']}\0".encode()+target.read_bytes()).hexdigest()==row['sha']
 if not valid():
  for attempt in range(3):
   try:
    import requests
    r=requests.get(f'https://raw.githubusercontent.com/{REPO}/{revision}/{row["path"]}',stream=True,timeout=45);r.raise_for_status()
    partial=target.with_suffix('.part')
    with partial.open('wb') as f:
     for chunk in r.iter_content(1024**2):f.write(chunk)
     f.flush();os.fsync(f.fileno())
    os.replace(partial,target);assert valid();break
   except Exception:
    if attempt==2:raise
    time.sleep(1)
 return {'path':row['path'],'bytes':row['size'],'git_blob_sha1':row['sha'],'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
def main():
 global DEST
 parser=argparse.ArgumentParser(description='Acquire or verify the frozen AIRTLab release; no work occurs on import.')
 parser.add_argument('--destination',type=Path,default=DEST)
 parser.add_argument('--verify-only',action='store_true',help='Verify existing media without downloads or rewriting the manifest')
 args=parser.parse_args();DEST=args.destination.resolve()
 tree=json.loads((HERE/'external_tree.json').read_text());files=[r for r in tree['tree'] if r['path'].endswith('.mp4')]
 assert len(files)==350
 if args.verify_only:
  for row in files:
   path=DEST/row['path'];content=path.read_bytes()
   assert len(content)==row['size'] and hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()==row['sha'],row['path']
  print(json.dumps({'verified_assets':len(files),'revision':revision,'no_downloads':True},indent=2));return
 manifest=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for row in pool.map(get,files):
   manifest.append(row)
   if len(manifest)%25==0:print('Downloaded',len(manifest),'of',len(files),flush=True)
 report={'repository':'https://github.com/'+REPO,'revision':revision,'files':manifest,'release_terms':'Freely released for research and educational purposes; cite Bianculli et al., Data in Brief33(2020)106587, DOI10.1016/j.dib.2020.106587. No raw media redistribution in submission package.'}
 (HERE/'external_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('External acquisition complete',revision,flush=True)
if __name__=='__main__':main()
