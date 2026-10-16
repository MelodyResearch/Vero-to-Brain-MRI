"""Freeze all28 existing training pairs; no model imports or test-patient access."""
import argparse
import json
from pathlib import Path
from mri_rl.answer_likelihood import LABELS, PREFIX, SUFFIX, training_pairs
from mri_rl.frozen_eval import sha256
from mri_rl.paced_frozen_eval import SYSTEM_PROMPT

def build(root, level):
    train=root/'vero_training_v1/train.verl.jsonl'
    schedule=root/'balanced_sft_resume_step028_v1/schedule_PRIVATE.json'
    b=json.loads(schedule.read_text())['rows']; patients={x['patient_key'] for x in b}
    if len(patients)!=28:raise ValueError('Exactly28 previously trained patients required')
    rows=[json.loads(x) for x in train.read_text().splitlines()]
    pairs=training_pairs(rows,patients,level)
    selected_b={x['record_id'] for x in b if x['category']=='mri_sequence'}
    question=pairs[0]['extra_info']['question']
    def prompt(image):
        content=[] if image is None else [{'type':'image','image':image}]
        content.append({'type':'text','text':question})
        return [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':content}]
    records=[]
    for row in pairs:
        image=row['images'][0]
        records.append({'record_id':row['id'],'patient_key':row['extra_info']['patient_key'],
          'level':level,'visible_label':row['reward_model']['ground_truth'],'image':image,'image_sha256':sha256(Path(image)),
          'prompt':prompt(image),'direct_A_sequence_exposure':level==45 and row['reward_model']['ground_truth']=='FLAIR',
          'direct_B_sequence_exposure':row['id'] in selected_b})
    records.insert(2,{'record_id':'no_image_unique','patient_key':None,'level':None,'visible_label':None,
       'image':None,'image_sha256':None,'prompt':prompt(None),'direct_A_sequence_exposure':False,'direct_B_sequence_exposure':False})
    adapters={'A0':root/'sft_warmstart_28step_v1/before_adapters.safetensors',
              'A28':root/'sft_warmstart_28step_v1/after_adapters.safetensors',
              'B28':root/'balanced_sft_resume_step028_v1/after_adapters.safetensors'}
    expected=['502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088',
              '244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73',
              'b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572']
    for path,digest in zip(adapters.values(),expected):
        if sha256(path)!=digest:raise ValueError('Historical checkpoint mismatch')
    model=root.parent/'models/qwen3vl8b-base'
    artifacts={str(model/name):sha256(model/name) for name in ('config.json','preprocessor_config.json','tokenizer_config.json','chat_template.json','tokenizer.json','model.safetensors.index.json')}
    return {'version':'training_answer_likelihood_v1','level':level,'patients':28,'records':records,
      'train_path':str(train),'train_sha256':sha256(train),'schedule_path':str(schedule),'schedule_sha256':sha256(schedule),
      'model':str(model),'model_artifacts':artifacts,
      'checkpoints':{name:{'path':str(p),'sha256':sha256(p),'adapter_config_sha256':sha256(p.parent/'adapter_config.json')} for name,p in adapters.items()},
      'labels':LABELS,'fixed_prefix':PREFIX,'fixed_suffix':SUFFIX,'primary_metric':'mean_label_logp_T1_minus_FLAIR',
      'common_prefix_tolerance':0.002,'generation':{'max_tokens':256,'temperature':0},
      'units':'28patient_slice_pairs; one unique no-image prompt; swaps reuse exact same visible inputs',
      'other_patient_mapping':'next patient in frozen pair order, opposite modality; reuse identical visible input score',
      'heldout_used':False,'weight_updates':0,'clinical_claim':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--level',type=int,choices=[45,91,136],default=45);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Never replace a frozen contract')
    value=build(a.root.resolve(),a.level);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'records':len(value['records']),'patients':value['patients'],'sha256':sha256(a.output)}))
