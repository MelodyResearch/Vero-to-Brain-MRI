"""Freeze existing internal-dev evaluation records only; never launches a model.

No final100 payload, predictions or labels are opened. The prospective external
patient plan is not represented as an established or loaded holdout.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from mri_rl.paired_z045_evaluation_v1 import protocol,require,LABELS
from mri_rl.v2_cpu_representation import _validate_rows
from mri_rl.answer_likelihood import PREFIX

FROZEN_DEV_SHA='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab'
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def swap_aliases(z):
    index={(r['extra_info']['patient_key'],r['reward_model']['ground_truth']):r for r in z}
    require(len(index)==len(z),'Duplicate z045 patient/class')
    swaps=[]
    for row in z:
        y=row['reward_model']['ground_truth'];labels=LABELS['mri_sequence'];other=labels[1-labels.index(y)]
        opposite=index[(row['extra_info']['patient_key'],other)]
        require(row['extra_info']['question']==opposite['extra_info']['question'],'Swap changed task question')
        swapped=copy.deepcopy(opposite);swapped['id']='paired_z045_swap_'+row['id']
        swaps.append({'id':swapped['id'],'condition':'same_patient_same_level_opposite_modality',
            'original_real_id_PRIVATE':row['id'],'displayed_source_id_PRIVATE':opposite['id'],
            'reuse_real_id_PRIVATE':opposite['id'],'row_PRIVATE':swapped})
    return swaps

def prepare(primary,output):
    frozen=primary/'.worktrees/vero-mri-v2-controls/local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json'
    require(sha(frozen)==FROZEN_DEV_SHA,'Existing frozen internal-dev source changed')
    dev=json.loads(frozen.read_text())['dev_PRIVATE'];_validate_rows(dev,'internal_dev')
    require(len(dev)==40,'Full old40 required')
    sequence=[r for r in dev if r['ability']=='mri_sequence'];z=[r for r in sequence if r['extra_info']['level']==45]
    require(len(sequence)==24 and len(z)==8,'Exact4people z045 pair coverage required')
    real=[];swaps=swap_aliases(z)
    for row in dev:
        image=Path(row['images'][0]);real.append({'id':row['id'],'condition':'real_image','row_PRIVATE':row,'image_sha256':sha(image)})
    real_hash={r['id']:r['image_sha256'] for r in real}
    for alias in swaps:alias['image_sha256']=real_hash[alias['reuse_real_id_PRIVATE']]
    output=output.resolve();require(output.is_relative_to(ROOT/'local_data'),'Private output required')
    output.mkdir(parents=True,exist_ok=False)
    d=protocol();d.update({'frozen_dev_source_sha256':FROZEN_DEV_SHA,'records_PRIVATE':real+swaps,
        'candidate_prefix':PREFIX,'label_order':LABELS,'metrics_helper_sha256':sha(ROOT/'src/mri_rl/paired_z045_evaluation_v1.py'),
        'historical_statistics_sha256':sha(ROOT/'src/mri_rl/v2_cpu_representation.py'),
        'scope_ready_for_native':False,'parent_and_child_adapter':'Bind actual Adam60 and exact finite terminal after numerical prerequisite; never select best-dev.',
        'new_training_or_inference_this_command':0})
    write(output/'evaluation_PRIVATE.json',d)
    public={k:v for k,v in d.items() if k not in {'records_PRIVATE','label_order'}}
    public.update({'evaluation_file_sha256':sha(output/'evaluation_PRIVATE.json'),'private_real_records':len(real),
        'private_swap_records':len(swaps),'source_file_sha256':sha(__file__),
        'source_images_read_for_hash_only':len(real),'MRI_images_decoded':0,
        'model_loads':0,'new_optimizer_updates':0})
    write(ROOT/'results/PAIRED_Z045_EVALUATION_PREPARATION_V1.json',public)
    return public

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=prepare(a.primary_root.resolve(),a.output)
    print(json.dumps({'status':r['status'],'greedy_calls_per_checkpoint':r['development']['greedy_calls_per_checkpoint'],'native_execution':False}))
if __name__=='__main__':main()
