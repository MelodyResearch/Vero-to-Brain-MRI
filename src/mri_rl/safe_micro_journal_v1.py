"""Private, single-writer, manifest-last operation journal. No native imports.
A reservation without a validated commit is indeterminate and NEVER replayed.
Reservations keep their full declared cost; a process crash cannot refund work.
"""
from pathlib import Path
import fcntl,json,os,re
from mri_rl.shared_rollout_gspo_core_v1 import publish_once,sync_dir,file_sha256,identity

def durable_mkdir(path):
 path=Path(path)
 if path.exists():
  if not path.is_dir():raise ValueError('Directory path occupied')
  return
 durable_mkdir(path.parent);path.mkdir();sync_dir(path.parent);sync_dir(path)

class Indeterminate(RuntimeError):pass

class Journal:
 def __init__(self,path,binding,before=None):
  self.path=Path(path).resolve();durable_mkdir(self.path)
  self.fd=os.open(self.path/'writer.lock',os.O_CREAT|os.O_RDWR,0o600)
  try:
   fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.binding=json.loads(json.dumps(binding));self.before=before or (lambda key:None)
   p=self.path/'binding_PRIVATE.json'
   if p.exists():
    if json.loads(p.read_text())!=self.binding:raise ValueError('Journal scope/input/source binding changed')
   else:publish_once(p,self.binding)
  except BaseException:os.close(self.fd);self.fd=None;raise
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def close(self):
  if self.fd is not None:os.close(self.fd);self.fd=None
 def _dir(self,key):
  if not re.fullmatch('[A-Za-z0-9_:-]+',key):raise ValueError('Unsafe operation key')
  return self.path/key
 def get(self,key):
  p=self._dir(key);claim=p/'reservation_PRIVATE.json';commit=p/'commit_PRIVATE.json'
  if not claim.exists():
   if p.exists() and any(p.iterdir()):raise Indeterminate('Unreserved operation artifacts: '+key)
   return None
  reserved=json.loads(claim.read_text())
  if reserved['binding_sha256']!=identity(self.binding) or reserved['key']!=key:raise ValueError('Reservation binding differs')
  if not commit.exists():raise Indeterminate('INDETERMINATE_NO_REPLAY: '+key+'; full reservation remains charged')
  rec=json.loads(commit.read_text())
  if rec['reservation_sha256']!=file_sha256(claim):raise ValueError('Commit reservation changed')
  if rec['record_sha256']!=identity({k:v for k,v in rec.items() if k!='record_sha256'}):raise ValueError('Commit digest changed')
  for filename,h in rec['files'].items():
   target=Path(filename)
   if not target.is_file() or file_sha256(target)!=h:raise ValueError('Committed artifact missing/changed: '+filename)
  return rec['value']
 def assert_recoverable(self):
  for p in self.path.iterdir():
   if p.is_dir():self.get(p.name)
  return True
 def run(self,key,operation,reservation=None):
  existing=self.get(key)
  if existing is not None:return existing
  self.before(key)                 # a pause HERE has not claimed/started work
  directory=self._dir(key);durable_mkdir(directory);sync_dir(self.path)
  claim=directory/'reservation_PRIVATE.json'
  publish_once(claim,{'key':key,'binding_sha256':identity(self.binding),'reserved_maximum_cost':reservation or {},'refund_on_crash':False})
  result=operation(directory)
  if not isinstance(result,dict) or set(result)!={'value','files'} or result['value'] is None:raise ValueError('Complete JSON value and immutable file manifest required')
  for filename,h in result['files'].items():
   target=Path(filename).resolve()
   if not target.is_relative_to(self.path.parent) or not target.is_file() or file_sha256(target)!=h:raise ValueError('Unbound/outside/missing committed file')
   with target.open('rb') as f:os.fsync(f.fileno())
   sync_dir(target.parent)
  rec={**result,'reservation_sha256':file_sha256(claim)};rec['record_sha256']=identity(rec)
  publish_once(directory/'commit_PRIVATE.json',rec)
  return self.get(key)
