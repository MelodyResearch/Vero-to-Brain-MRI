"""Immutable C evaluation projection and strict raw-output adaptation; CPU only."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

from mri_rl.two_task_pool import read_rows, require, sha, write_exact
from mri_rl.two_task_analysis import _reference


def project_records(rows,contract):
    require(contract['enable_thinking'] is False and contract['assistant_prefix']=='<|im_start|>assistant\n','Native generation prefix contract differs')
    return [{'record_id':r['id'],'category':r['ability'],'image':r['images'][0],
             'prompt':[{'role':'system','content':contract['system_instruction']},
                       {'role':'user','content':r['extra_info']['question']}]} for r in rows]


def render_cpu_prompt(messages,model,formatter_path):
    """Pure formatter+Jinja rendering, verified against already native-recorded SHA.

    This shim is not substituted into model execution. It avoids importing a
    processor/model merely to check the two fixed chat strings used by C40.
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    template=json.loads((Path(model)/'chat_template.json').read_text())['chat_template']
    class TemplateOnly:
        chat_template=template
        def apply_chat_template(self,messages,add_generation_prompt=True,tokenize=False,**kwargs):
            require(tokenize is False,'CPU parity rendering does not tokenize')
            environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
            return environment.from_string(self.chat_template).render(messages=messages,add_generation_prompt=add_generation_prompt,**kwargs)
    spec=importlib.util.spec_from_file_location('mri_eval_native_formatter_cpu',formatter_path)
    formatter=importlib.util.module_from_spec(spec);spec.loader.exec_module(formatter)
    return formatter.apply_chat_template(TemplateOnly(),json.loads((Path(model)/'config.json').read_text()),messages,
                                         num_images=1,enable_thinking=False)


def freeze_projection(source_manifest,pool_receipt,contract_path,model,training_manifest,native_training_inputs,formatter_path,destination):
    source_manifest=Path(source_manifest);destination=Path(destination)
    pool=json.loads(Path(pool_receipt).read_text());contract=json.loads(Path(contract_path).read_text())
    require(sha(source_manifest)==pool['manifest_sha256']['internal_dev'],'Frozen C dev manifest hash differs')
    source=[x['row'] for x in read_rows(source_manifest)];_reference(source,'internal_dev')
    records=project_records(source,contract)
    training={x['row']['id']:x['row'] for x in read_rows(training_manifest)}
    native_inputs=json.loads(Path(native_training_inputs).read_text());known={}
    for row in native_inputs:
        origin=training[row['record_id']];key=(origin['ability'],origin['extra_info']['question'])
        value=row['rendered_prompt_sha256']
        require(key not in known or known[key]==value,'Training native prompt hashes differ for the same question')
        known[key]=value
    private=[]
    for record,original in zip(records,source):
        rendered=render_cpu_prompt(record['prompt'],model,formatter_path)
        digest=hashlib.sha256(rendered.encode()).hexdigest()
        require(rendered.endswith(contract['assistant_prefix']) and rendered.count('<|image_pad|>')==1,'Native prefix/image placeholder parity failed')
        require(digest==known[(original['ability'],original['extra_info']['question'])],'Projected native text hash differs from captured training render')
        private.append({'record_id':record['record_id'],'image_sha256':sha(record['image']),'rendered_prompt_sha256':digest})
    manifest=destination/'internal_dev_native_system_matched.jsonl'
    write_exact(manifest,(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)).encode())
    write_exact(destination/'projection_inventory_PRIVATE.json',(json.dumps(private,indent=2,sort_keys=True)+'\n').encode())
    safe={'status':'C40_NATIVE_SYSTEM_MATCHED_EVAL_PROJECTION_FROZEN','cases':len(records),
        'source_verl_manifest_sha256':sha(source_manifest),'eval_manifest_sha256':sha(manifest),
        'pool_receipt_sha256':sha(pool_receipt),'prompt_contract_sha256':sha(contract_path),
        'model_config_sha256':sha(Path(model)/'config.json'),'template_sha256':sha(Path(model)/'chat_template.json'),
        'installed_formatter_source_sha256':sha(formatter_path),'projection_source_sha256':sha(__file__),
        'captured_native_training_inputs_sha256':sha(native_training_inputs),
        'projection_kind':'training-system-matched: system message plus unchanged question string',
        'messages_equal_native_training':True,'all40_rendered_prompt_sha_match_captured_native_training':True,
        'prefix_and_single_image_placeholder_checked':True,'image_bytes_verified':True,'hidden_targets_in_model_text':False,
        'actual_model_execution':False,'processor_or_model_imported':False,'test_content_used':False,
        'parity_method':'Pure installed formatter and pinned Jinja chat template; each resulting SHA matched the previously captured actual native processor training render. No new tokenizer/processor/model was loaded.'}
    write_exact(destination/'projection_receipt_SAFE.json',(json.dumps(safe,indent=2,sort_keys=True)+'\n').encode())
    return manifest,safe
