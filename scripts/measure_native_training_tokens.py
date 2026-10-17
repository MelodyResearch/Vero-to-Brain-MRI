#!/usr/bin/env python3
"""Exact processor-only native token measurement; no model/weight construction.

Large runtime imports are inside run(), requiring a root-coordinated safe CPU
window. Importing this module for accounting tests uses the standard library.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from mri_rl.two_task_pool import canonical_sha, read_rows, require, sha, validate_model_boundary, write_exact


def inspect_input_tokens(input_ids,image_grid_thw,image_pad_id,merge_size):
    require(len(input_ids)==1,'Expected one single input sequence')
    ids=list(input_ids[0]);grid=[list(map(int,x)) for x in image_grid_thw]
    require(len(grid)==1 and len(grid[0])==3 and all(x>0 for x in grid[0]),'Expected one positive image grid')
    visual=sum(int(x)==image_pad_id for x in ids)
    product=grid[0][0]*grid[0][1]*grid[0][2]
    require(merge_size>0 and product%(merge_size**2)==0 and product//(merge_size**2)==visual,'Expanded image ID count differs from processor grid')
    require(0<visual<len(ids)<=2048,'Native prompt cost outside frozen profile')
    return {'input_tokens':len(ids),'visual_tokens':visual,'text_and_boundary_tokens':len(ids)-visual,'image_grid_thw':grid}


def run(args):
    # All validation before any native runtime import. No model class/load() call exists.
    require(bool(args.cpu_window_reference),'Root-coordinated processor-only CPU window reference required')
    require(args.destination.resolve().is_relative_to(REPO/'local_data'),'Private token outputs must remain under local_data')
    contract=json.loads(args.contract.read_text());pool=json.loads(args.pool_receipt.read_text())
    require(sha(args.manifest)==pool['manifest_sha256']['train'],'Frozen C training manifest differs')
    entries=read_rows(args.manifest)
    require(len(entries)==76 and Counter(x['row']['ability'] for x in entries)=={'mri_sequence':28,'mri_visit_correspondence':48},'Expected complete frozen 76-row training pool')
    for x in entries:validate_model_boundary(x['row'],'train')
    names={'model_config':'config.json','processor_config':'preprocessor_config.json','tokenizer_config':'tokenizer_config.json','chat_template':'chat_template.json'}
    for key,name in names.items():require(sha(args.model/name)==contract[key+'_sha256'],'Pinned processor/config artifact mismatch')
    processor_files={p.name:sha(p) for p in args.model.iterdir() if p.is_file() and p.name in {'config.json','preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','chat_template.json','chat_template.jinja','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json','video_preprocessor_config.json','generation_config.json'}}
    config=json.loads((args.model/'config.json').read_text())
    expected={'manifest_sha256':sha(args.manifest),'pool_receipt_sha256':sha(args.pool_receipt),'prompt_contract_sha256':sha(args.contract),
              'script_sha256':sha(__file__),'processor_artifact_sha256':processor_files,'cpu_window_reference':args.cpu_window_reference}
    args.destination.mkdir(parents=True,exist_ok=True)
    binding=args.destination/'binding_PRIVATE.json';write_exact(binding,(json.dumps(expected,indent=2,sort_keys=True)+'\n').encode())
    # The coordinator has confirmed no resident 8B job conflicts with this import window.
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false');os.environ.setdefault('OMP_NUM_THREADS','1')
    import mlx.core as mx
    from mlx_vlm.utils import load_processor,load_image_processor,load_config,get_model_and_args,prepare_inputs,should_add_special_tokens
    from mlx_vlm.prompt_utils import apply_chat_template
    from mri_rl.resource_supervisor import sample_macos
    from dataclasses import asdict
    mx.set_default_device(mx.cpu)
    write_exact(args.destination/'start_resource_SAFE.json',(json.dumps(asdict(sample_macos()),sort_keys=True)+'\n').encode())
    # Native load() imports the model package before AutoProcessor; that import
    # installs MLX-VLM's existing torch-free Qwen processor patch. Resolve only
    # its configuration classes, never instantiate Model or load model weights.
    config=load_config(args.model,local_files_only=True)
    architecture,_=get_model_and_args(config)
    native_config=architecture.ModelConfig.from_dict(config)
    image_processor=load_image_processor(args.model,local_files_only=True)
    processor=load_processor(args.model,True,eos_token_ids=getattr(native_config,'eos_token_id',None),local_files_only=True)
    if image_processor is not None:processor.image_processor=image_processor
    tokenizer=processor.tokenizer
    image_pad_id=int(tokenizer.convert_tokens_to_ids('<|image_pad|>'))
    require(tokenizer.decode([image_pad_id])=='<|image_pad|>','Tokenizer image pad identity mismatch')
    merge_size=int(processor.image_processor.merge_size)
    rows=[];private=[];record_dir=args.destination/'records_PRIVATE';record_dir.mkdir(exist_ok=True)
    for index,item in enumerate(entries):
        row=item['row'];record_sha=canonical_sha(row);record_file=record_dir/(record_sha+'.json')
        image_sha=sha(row['images'][0])
        inventory=json.loads((args.pool_receipt.parent/'row_inventory_PRIVATE.json').read_text())['image_sha256_by_record']
        require(image_sha==inventory[row['id']],'Frozen image bytes differ')
        if record_file.exists():
            saved=json.loads(record_file.read_text())
            require(saved['binding_sha256']==sha(binding) and saved['record_sha256']==record_sha and saved['image_sha256']==image_sha,'Completed token record differs')
            measurement=saved['measurement'];details=saved['details']
        else:
            # Mirrors NativeGSPOBackend.rollout exactly: fixed system, question-only user,
            # installed native formatter, add_generation_prompt, and identical prepare_inputs.
            messages=[{'role':'system','content':contract['system_instruction']},{'role':'user','content':row['extra_info']['question']}]
            prompt=apply_chat_template(processor,native_config,messages,add_generation_prompt=True,num_images=1,enable_thinking=contract['enable_thinking'])
            require(prompt.count('<|image_pad|>')==1 and prompt.endswith(contract['assistant_prefix']),'Native placeholder/prefix mismatch')
            inputs=prepare_inputs(processor,images=[row['images'][0]],prompts=prompt,
                image_token_index=getattr(native_config,'image_token_index',None),
                add_special_tokens=should_add_special_tokens(native_config.model_type,processor))
            mx.eval(inputs['input_ids'],inputs['image_grid_thw'])
            ids=inputs['input_ids'].tolist();grid=inputs['image_grid_thw'].tolist()
            costs=inspect_input_tokens(ids,grid,image_pad_id,merge_size)
            measurement={'record_sha256':record_sha,'source':row['data_source'],'route':row['ability'],
                         'input_tokens':costs['input_tokens'],'visual_tokens':costs['visual_tokens']}
            details={'record_id':row['id'],'image_sha256':image_sha,'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                     'input_token_ids':ids,'input_token_ids_sha256':canonical_sha(ids),'image_grid_thw':grid,
                     'text_and_boundary_tokens':costs['text_and_boundary_tokens']}
            saved={'binding_sha256':sha(binding),'record_sha256':record_sha,'image_sha256':image_sha,'measurement':measurement,'details':details}
            write_exact(record_file,(json.dumps(saved,sort_keys=True)+'\n').encode())
            del inputs;mx.clear_cache();gc.collect()
        rows.append(measurement);private.append(details)
        print(json.dumps({'state':'NATIVE_PROCESSOR_RECORD_MEASURED','completed':index+1,'of':76,'route':row['ability'],
                          'input_tokens':measurement['input_tokens'],'visual_tokens':measurement['visual_tokens']}),flush=True)
    processor_binding={**processor_files,'native_processor_patch_sha256':sha(Path(importlib.metadata.distribution('mlx-vlm').locate_file('mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py'))),'runtime_transformers':importlib.metadata.version('transformers'),'runtime_mlx_vlm':importlib.metadata.version('mlx-vlm')}
    receipt={'schema':'frozen-training-token-measurements-v1','split':'train','manifest_sha256':sha(args.manifest),
             'processor_sha256':canonical_sha(processor_binding),'rows':rows,
             'method':'Installed mlx_vlm load_processor/apply_chat_template/prepare_inputs; CPU tensor device, no model weights or model class constructed.',
             'prompt_contract_sha256':sha(args.contract),'pool_receipt_sha256':sha(args.pool_receipt),'source_script_sha256':sha(__file__),
             'processor_artifacts':processor_binding,'binding_sha256':sha(binding),'model_weights_loaded':False,
             'input_tokens_include_visual':True,'visual_tokens_not_added_twice':True,'records_measured':len(rows),
             'peak_process_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
             'cpu_window_reference':args.cpu_window_reference,
             'native_equivalence_scope':'Same installed processor, formatter, image preparation, special-token rule and fixed system/prefix as native backend; token IDs and grid measured directly. No model predictions or pixel-tensor numerical equivalence claim.'}
    write_exact(args.destination/'measurements_PRIVATE.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode())
    write_exact(args.destination/'native_inputs_PRIVATE.json',(json.dumps(private,indent=2,sort_keys=True)+'\n').encode())
    write_exact(args.destination/'end_resource_SAFE.json',(json.dumps(asdict(sample_macos()),sort_keys=True)+'\n').encode())
    summary={'status':'ALL76_NATIVE_PROCESSOR_INPUTS_MEASURED_NO_MODEL_WEIGHTS','records':len(rows),
             'manifest_sha256':receipt['manifest_sha256'],'processor_sha256':receipt['processor_sha256'],
             'measurement_receipt_sha256':sha(args.destination/'measurements_PRIVATE.json'),
             'input_tokens_min':min(r['input_tokens'] for r in rows),'input_tokens_max':max(r['input_tokens'] for r in rows),
             'per_route':{t:{'records':sum(r['route']==t for r in rows),'input_token_counts':sorted({r['input_tokens'] for r in rows if r['route']==t}),
                              'visual_token_counts':sorted({r['visual_tokens'] for r in rows if r['route']==t})} for t in ('mri_sequence','mri_visit_correspondence')},
             'peak_process_rss_bytes':receipt['peak_process_rss_bytes'],'model_weights_loaded':False,'model_or_rl_run':False}
    write_exact(args.destination/'measurement_summary_SAFE.json',(json.dumps(summary,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','manifest','pool-receipt','contract','destination'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cpu-window-reference',required=True)
    run(p.parse_args())


if __name__=='__main__':main()
