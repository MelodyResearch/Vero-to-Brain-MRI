"""Thin adapter for REAL continuous_rl_dev_eval JSONL; never infers missing bindings.

The caller supplies independently authenticated launch-scope/projection/output
hashes. Current CLI rows omit contract/cache/precision scope, so their exact
approved scope and completed output file are authenticated separately. This
module does not create approvals or substitute the caller's own chosen digest
for an independent review. Legacy predictions require their separate bridge.
"""
from __future__ import annotations
import json
from pathlib import Path
from mri_rl.two_task_pool import canonical_sha,read_rows,require,sha
from mri_rl.two_task_evaluation import project_records

CONFIG_KEYS={'model_path','model_config_sha256','model_template_sha256','template_override_sha256',
             'manifest_sha256','max_tokens','temperature','adapter_checkpoint_sha256','adapter_config_sha256'}


def adapt_cli_output(*,source_manifest,eval_manifest,projection_receipt,contract_path,output_path,
                     expected_config,approved_scope,expected_scope_sha256,expected_projection_sha256,
                     expected_output_sha256,expected_cache_binding):
    """Normalize one complete approved native-eval component, including subsets.

    No defaults for missing scope/cache/checkpoint evidence. The hashes must be
    captured independently from the approved launch and completed worker receipt,
    not synthesized from an untrusted file inside this function.
    """
    require(canonical_sha(approved_scope)==expected_scope_sha256,'Approved scope hash differs')
    require(sha(projection_receipt)==expected_projection_sha256,'Projection receipt hash differs')
    require(sha(output_path)==expected_output_sha256,'Completed output hash differs')
    projection=json.loads(Path(projection_receipt).read_text());contract=json.loads(Path(contract_path).read_text())
    require(projection['source_verl_manifest_sha256']==sha(source_manifest) and projection['eval_manifest_sha256']==sha(eval_manifest)
            and projection['prompt_contract_sha256']==sha(contract_path),'Frozen projection/source/contract binding differs')
    require(set(expected_config)==CONFIG_KEYS,'Exact expected native configuration fields required')
    require(expected_config['manifest_sha256']==sha(eval_manifest) and expected_config['temperature']==0
            and type(expected_config['max_tokens']) is int and expected_config['max_tokens']>0
            and expected_config['template_override_sha256'] is None
            and expected_config['max_tokens']==contract['generation']['max_tokens']
            and expected_config['model_config_sha256']==contract['model_config_sha256']
            and expected_config['model_template_sha256']==contract['chat_template_sha256'],'Native manifest/decoding/template contract differs')
    required_scope={'execution_tier','kind','label','contract_id','manifest_sha256','subset_record_ids','max_tokens','temperature',
        'adapter_checkpoint_sha256','adapter_config_sha256','output','weight_updates','derived_cache','image_sha256','model_inventory','source_sha256'}
    require(required_scope<=set(approved_scope),'Required approved launch scope is absent')
    require(approved_scope['execution_tier']=='B' and approved_scope['kind']=='continuous_rl_dev_eval_v1'
            and approved_scope['weight_updates']==0 and approved_scope['contract_id']==canonical_sha(contract)
            and approved_scope['output']==str(Path(output_path).resolve()),'Wrong launch contract/kind/output or weight-changing scope')
    for field in ('manifest_sha256','max_tokens','temperature','adapter_checkpoint_sha256','adapter_config_sha256'):
        require(approved_scope[field]==expected_config[field],'Approved scope/configuration differs: '+field)
    require(isinstance(expected_cache_binding,dict) and expected_cache_binding
            and approved_scope['derived_cache']==expected_cache_binding,'Exact approved cache/precision binding differs')
    require(expected_cache_binding.get('tensors')==1438 and expected_cache_binding.get('config_source_sha256')==expected_config['model_config_sha256']
            and expected_cache_binding.get('certificate_sha256'),'Certified full base/precision source binding absent')
    inventory=approved_scope['model_inventory']
    require(inventory['model_dir']==expected_config['model_path'],'Model directory differs')
    configs=[x for x in inventory['files'] if x['path']=='config.json']
    require(len(configs)==1 and configs[0]['sha256']==expected_config['model_config_sha256'],'Approved model inventory/config differs')
    require({'continuous_rl_dev_eval.py','mlx_continuous_gspo.py','likelihood_cache.py'}<=set(approved_scope['source_sha256']),'Required actual native source bindings absent')
    originals={x['row']['id']:x['row'] for x in read_rows(source_manifest)}
    projected={r['record_id']:r for r in project_records(list(originals.values()),contract)}
    records=[json.loads(line) for line in Path(eval_manifest).read_text().splitlines() if line.strip()]
    require(len(records)==len({r['record_id'] for r in records}) and all(r==projected.get(r['record_id']) for r in records),'Eval rows differ from native system-matched projection')
    by_id={r['record_id']:r for r in records};selected=approved_scope['subset_record_ids']
    require(bool(selected) and len(selected)==len(set(selected)) and set(selected)<=set(by_id),'Invalid approved subset inventory')
    require(set(selected)<=set(approved_scope['image_sha256']),'Approved image inventory absent')
    normalized={}
    for line in Path(output_path).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);key=row.get('record_id')
        require(key in selected and key not in normalized,'Unknown/duplicate native output ID')
        target=by_id[key]
        require(all(field in row and row[field]==value for field,value in expected_config.items()),'Native model/checkpoint/manifest/precision/decoding fields differ or are missing')
        require(row.get('category')==target['category'] and row.get('prompt')==target['prompt'],'Native task or system/question prompt differs')
        image_sha=sha(target['image'])
        require(row.get('image_sha256')==approved_scope['image_sha256'][key]==image_sha and row.get('image_placeholder_count')==1,'Native image or placeholder binding differs')
        require(row.get('engine')=='continuous_rl_dev_eval_native_serial_v1','Unreviewed generation engine')
        require(isinstance(row.get('response'),str) and isinstance(row.get('token_ids'),list)
                and all(type(t) is int for t in row['token_ids']) and len(row['token_ids'])<=expected_config['max_tokens'],'Malformed/incomplete native prediction')
        normalized[key]={'response':row['response']}
    require(set(normalized)==set(selected),'Scoped native evaluation remains incomplete')
    require(sha(output_path)==expected_output_sha256 and sha(eval_manifest)==expected_config['manifest_sha256'],'Evidence changed during adaptation')
    proof={'status':'REAL_NATIVE_CLI_OUTPUT_BOUND_AND_NORMALIZED','rows':len(normalized),'output_sha256':expected_output_sha256,
        'projection_receipt_sha256':expected_projection_sha256,'approved_scope_sha256':expected_scope_sha256,
        'eval_manifest_sha256':sha(eval_manifest),'source_verl_manifest_sha256':sha(source_manifest),
        'contract_sha256':sha(contract_path),'adapter_checkpoint_sha256':expected_config['adapter_checkpoint_sha256'],
        'cache_certificate_sha256':expected_cache_binding['certificate_sha256'],
        'scope_supplies_omitted_contract_cache_precision_binding':True,
        'precision_scope':'Exact certified cache object/source config plus adapter_config SHA and approved native source scope; no invented per-row cache or precision field.',
        'source_sha256':sha(__file__),'legacy_rows_relabelled':False,'review_authorization_created':False}
    return normalized,proof
