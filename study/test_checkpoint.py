import ctypes,json,os,threading,time,unittest
from pathlib import Path
from run_study import HERE,atomic

class CheckpointTests(unittest.TestCase):
 @unittest.skipUnless(os.name=='nt','Windows delete-sharing contention')
 def test_transient_reader_lock_does_not_lose_checkpoint(self):
  p=HERE/'atomic_checkpoint_test.json';assert not p.exists();atomic(p,{'old':True})
  dll=ctypes.WinDLL('kernel32',use_last_error=True);dll.CreateFileW.restype=ctypes.c_void_p;dll.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p];dll.CloseHandle.argtypes=[ctypes.c_void_p]
  handle=dll.CreateFileW(str(p),0x80000000,3,None,3,128,None);assert handle not in [None,ctypes.c_void_p(-1).value]
  def release():time.sleep(.4);dll.CloseHandle(handle)
  thread=threading.Thread(target=release);thread.start()
  try:
   atomic(p,{'new':True});self.assertEqual(json.loads(p.read_text()),{'new':True})
  finally:
   thread.join();p.unlink(missing_ok=True);p.with_suffix('.json.tmp').unlink(missing_ok=True)
if __name__=='__main__':unittest.main()
