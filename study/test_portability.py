"""Regression checks for the standalone published utilities."""
from pathlib import Path
import ast,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
class PortabilityTests(unittest.TestCase):
 def test_every_python_file_compiles(self):
  for path in HERE.glob('*.py'):
   with self.subTest(file=path.name):compile(path.read_text(encoding='utf-8'),str(path),'exec')
 def test_geometry_runs_without_source_images(self):
  run=subprocess.run([sys.executable,str(HERE/'check_geometry.py')],capture_output=True,text=True)
  self.assertEqual(run.returncode,0,run.stderr)
  report=json.loads(run.stdout);self.assertEqual(report['checked'],1627);self.assertEqual(report['mismatches'],0)
 def test_long_running_utilities_have_guarded_entrypoints(self):
  for name in ['download_external.py','profile_strict.py','check_geometry.py']:
   tree=ast.parse((HERE/name).read_text())
   self.assertTrue(any(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test) for n in tree.body),name)
 def test_downloader_import_does_not_fetch_or_create_media(self):
  spec=importlib.util.spec_from_file_location('download_under_test',HERE/'download_external.py');module=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory() as tmp:
   module.DEST=Path(tmp);content=b'verified test fixture';path=Path(tmp)/'clips/example.mp4';path.parent.mkdir();path.write_bytes(content)
   row={'path':'clips/example.mp4','size':len(content),'sha':hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()}
   # An already verified asset must not need an HTTP request.
   report=module.get(row);self.assertEqual(report['sha256'],hashlib.sha256(content).hexdigest())
if __name__=='__main__':unittest.main()
