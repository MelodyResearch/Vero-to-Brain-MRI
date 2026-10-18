"""Pure finite generation-control records and immutable journals; strict MRI reward/scorer remain unchanged."""
import copy,hashlib,json,os
from pathlib import Path
SHORT_INSTRUCTION='Keep the reasoning brief (at most two short sentences) and finish with the required boxed answer.'
COUNTS={'original_512_greedy':40,'original_512_null':1,'short_reason_256_greedy':40,'short_reason_256_candidates':25,'short_reason_256_null':1}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def control_records(evaluation):
 require(len(evaluation['greedy_records_PRIVATE'])==40 and len(evaluation['candidate_records_PRIVATE'])==25,'Frozen40/25 records required');out={}
 for name,cap in (('original_512',512),('short_reason_256',256)):
  def transform(record):
   r=copy.deepcopy(record);prompt=r['prompt'];require(len(prompt)==2 and prompt[0]['role']=='system' and prompt[1]['role']=='user' and isinstance(prompt[1]['content'],str),'Exact system/user question required')
   r['original_prompt_sha256']=canon(prompt)
   if name=='short_reason_256':prompt[1]['content']+='\n\n'+SHORT_INSTRUCTION
   r.update(recovery_condition=name,max_new_tokens=cap);return r
  out[name]={'max_tokens':cap,'greedy':[transform(r) for r in evaluation['greedy_records_PRIVATE']],'candidates':[transform(r) for r in evaluation['candidate_records_PRIVATE']],'candidate_mode':'baseline_reference' if name=='original_512' else 'native_short_prompt'}
 return out

def validate_greedy(row,cap):
 ids=row.get('token_ids');require(isinstance(row.get('response'),str) and isinstance(ids,list) and len(ids)<=cap and all(type(t) is int and t>=0 for t in ids),'Missing bounded generation evidence')
 require(type(row.get('generation_tokens')) is int and row['generation_tokens']==len(ids) and row.get('finish_reason') in {'stop','length'} and row.get('hit_max_tokens')==(len(ids)>=cap),'Generation finish/count evidence differs')
 require(row['finish_reason']!='length' or len(ids)==cap,'Length finish before the frozen cap')

def durable_prefix(path,records,identity,validator):
 path=Path(path)
 if not path.exists():return []
 raw=path.read_bytes();require(not raw or raw.endswith(b'\n'),'Torn control journal');rows=[json.loads(v) for v in raw.splitlines()];require(len(rows)<=len(records),'Duplicate/oversized control journal')
 for row,expected in zip(rows,records):
  try:json.dumps(row,allow_nan=False)
  except (ValueError,TypeError):raise ValueError('Nonfinite control output') from None
  require(row.get('run_identity')==identity and row.get('weight_updates')==0 and all(row.get(k)==v for k,v in expected.items()),'Control record/prompt/cap/identity differs');validator(row)
 return rows

def append(path,row):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def output_paths(output):
 p=Path(output);return {'original_greedy':p/'original_512/greedy_PRIVATE.jsonl','original_null':p/'original_512/null_greedy_PRIVATE.jsonl','short_greedy':p/'short_reason_256/greedy_PRIVATE.jsonl','short_candidates':p/'short_reason_256/candidates_PRIVATE.jsonl','short_null':p/'short_reason_256/null_greedy_PRIVATE.jsonl'}

def validate_completion(receipt,scope,paths):
 require(receipt['status']=='ACTUAL_GENERATION_RECOVERY_CONTROLS_COMPLETE' and receipt['run_identity']==scope['run_identity'] and receipt['adapter_sha256']==scope['adapter']['sha256'] and receipt['baseline_scope_sha256']==scope['baseline_evidence']['scope']['sha256'] and receipt['evaluation_contract_sha256']==scope['evaluation_contract_sha256'],'Completed control identity differs')
 require(receipt['native_release_verified'] is True and receipt['weight_updates']==0 and receipt['counts']==COUNTS,'Incomplete controls or native release')
 require(receipt['outputs_sha256']=={k:sha(v) for k,v in paths.items()},'Completed control files changed')
 expected={'original_greedy':40,'original_null':1,'short_greedy':40,'short_candidates':25,'short_null':1}
 require(set(paths)==set(expected) and all(len(Path(paths[k]).read_bytes().splitlines())==n for k,n in expected.items()),'Completed control file coverage differs')
