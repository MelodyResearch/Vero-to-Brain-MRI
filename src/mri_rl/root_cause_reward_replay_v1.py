"""Train-only historical reward/gate replay. No model or tensor-state loading.

Native decoding functions are selected as AST definitions from the previously
certified independent verifier, whose complete top-level program is never run.
The historical parser, reward callback and gate remain unchanged. An explicitly
named single-box diagnostic does not contribute to historical grades or gates.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from mri_rl.vero_mri_reward import _answer_and_format, _boxed_contents

TASKS=('mri_sequence','mri_visit_correspondence')
ARMS=('S_sequence','S_correspondence','P_raw','U_raw','P_calibrated','U_calibrated')
SEEDS=(7,11,13)
NATIVE_VERIFIER_SHA='0c3408940bc36a646dfc97eb37c27c5964c0eba331f8435b636dce3e714df048'
DECODER_SOURCES={'tokenizer_utils.py':'7fb64819e0577e7e960c8b8eb247644b4a70046c117fa5eb4a00d30c0f9c645f',
 'generate/dispatch.py':'f7dcec326a4d2aff1286b1abd4419f8ec533189d1d26b0ac69002513ba45dab8',
 'utils.py':'d38eabab88d5d6b835d4ab703cdac2e30924fc04463be8134d12ae5406e50b0c'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def hashobj(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def sample_advantages(values,epsilon=1e-6):
    """Historical sample SD; equality guard gives the same exact-tie result."""
    require(len(values)>=2 and all(math.isfinite(x) for x in values),'Finite group required')
    require(math.isfinite(epsilon) and epsilon>=0,'Nonnegative finite epsilon required')
    if min(values)==max(values):return [0.]*len(values)
    mu=sum(values)/len(values)
    sd=math.sqrt(sum((x-mu)**2 for x in values)/(len(values)-1))
    return [(x-mu)/(sd+epsilon) for x in values]


def logistic(value,stats):
    require(stats['iqr']>0,'Positive frozen IQR required')
    z=3.*(value-stats['p50'])/stats['iqr']
    if z>=0:return 1./(1.+math.exp(-z))
    e=math.exp(z);return e/(1.+e)


def analyze_group(texts,lengths,target,task,calibration=None):
    require(task in TASKS and len(texts)==len(lengths)==4,'Historical four-response task group required')
    require(all(type(n) is int and 0<n<=256 for n in lengths),'Historical response length out of range')
    parsed=[_answer_and_format(t) for t in texts]
    raw=[float(f>0 and a.lower()==target.lower()) for a,f in parsed]
    fmt=[f for a,f in parsed]
    calibrated=[logistic(a,calibration) if calibration else a for a in raw]
    overlong=[min(-(n-16384)/2048.,0.) for n in lengths]
    total=[.8*a+.2*f+o for a,f,o in zip(calibrated,fmt,overlong)]
    raw_total=[.8*a+.2*f+o for a,f,o in zip(raw,fmt,overlong)]
    # Preserve order of Python's historical short-circuit gate.
    predicates=[('format_not_all_full',all(f==1 for f in fmt)),
                ('task_accuracy_constant',len(set(calibrated))>1),
                ('overlong_not_equal',len(set(overlong))==1)]
    failures=[name for name,passed in predicates if not passed]
    diagnostics=[]
    allowed={'mri_sequence':{'flair','contrast-enhanced t1'},'mri_visit_correspondence':{'same','different'}}[task]
    # Strict parser only: an accuracy zero from an invalid wrapper does not
    # identify an incorrect categorical answer. Unknown/nonclass answers never
    # enter either valid-label subset, even when their native accuracy is zero.
    class_status=[('format_invalid_unknown' if f==0 else
                   'format_valid_nonclass_unknown' if a.lower() not in allowed else
                   'correct' if a.lower()==target.lower() else 'wrong') for a,f in parsed]
    positive_status={status for status in class_status if status in {'correct','wrong'}}
    full_status={status for status,(_,f) in zip(class_status,parsed)
                 if f==1 and status in {'correct','wrong'}}
    positive_mix=positive_status=={'correct','wrong'}
    full_mix=full_status=={'correct','wrong'}
    if len(set(raw))==1:variation_source='constant_native_accuracy'
    elif positive_mix:variation_source='valid_class_correct_and_wrong'
    elif all(f==0 for a,f in zip(raw,fmt) if a==0):
        variation_source='only_format_invalid_zero_vs_valid_correct'
    else:variation_source='unknown_or_nonclass_zero_vs_valid_correct'
    for text in texts:
        boxes=_boxed_contents(text)
        answer=boxes[0].lower() if len(boxes)==1 and text.count(r'\boxed')==1 else ''
        diagnostics.append(float(answer==target.lower()) if answer in allowed else None)
    adv=sample_advantages(total);raw_adv=sample_advantages(raw_total)
    epsfree=sample_advantages(total,0);raw_epsfree=sample_advantages(raw_total,0)
    raw_varies=len(set(raw))>1;fmt_varies=len(set(fmt))>1;total_varies=len(set(total))>1
    return {'accepted':not failures,'failed_gates':failures,'first_gate':failures[0] if failures else 'accepted',
        'raw_accuracy':raw,'format':fmt,'calibrated_accuracy':calibrated,'overlong':overlong,
        'strict_categorical_response_status':class_status,
        'positive_format_valid_class_mix':positive_mix,'full_format_valid_class_mix':full_mix,
        'native_accuracy_variation_source':variation_source,
        'total':total,'advantages':adv,'raw_total':raw_total,'raw_advantages':raw_adv,
        'epsilon_free_advantages':epsfree,'raw_epsilon_free_advantages':raw_epsfree,
        'raw_calibrated_max_difference':max(abs(a-b) for a,b in zip(adv,raw_adv)),
        'raw_calibrated_epsilon_free_max_difference':max(abs(a-b) for a,b in zip(epsfree,raw_epsfree)),
        'task_accuracy_varies':raw_varies,'format_varies':fmt_varies,'total_reward_varies':total_varies,
        'informative_advantages':raw_varies and not failures and max(adv)-min(adv)>1e-12,
        'nonzero_advantages':max(adv)-min(adv)>1e-12,
        'format_only_total_variation':total_varies and not raw_varies and fmt_varies,
        'correctness_varies_but_format_rejected':raw_varies and bool(failures) and failures[0]=='format_not_all_full',
        'single_box_diagnostic_accuracy':diagnostics,
        'single_box_diagnostic_accuracy_varies':all(x is not None for x in diagnostics) and len(set(diagnostics))>1,
        'lengths':lengths,'at_generation_cap':sum(n==256 for n in lengths),
        'native_correct_responses':int(sum(raw)),'full_format_responses':sum(f==1 for f in fmt)}


class ReadAudit:
    """Record all explicit evidence reads; refuse tensors and heldout paths."""
    def __init__(self):self.files={}
    def path(self,path):
        p=Path(path).resolve()
        require(p.suffix not in {'.safetensors','.npz','.npy','.png','.nii','.gz'},'Tensor/image reads forbidden')
        require(not any(bad in part.lower() for part in p.parts for bad in ('final100','c40','test','internal_dev')),'Heldout evidence read forbidden')
        self.files[str(p)]=sha(p)
        return p
    def json(self,path):return json.loads(self.path(path).read_text())
    def text(self,path):return self.path(path).read_text()


def certified_decoder(root,audit,certificate_sha):
    """Reuse exact certified verifier function bodies without native imports."""
    path=root/'local_data/c_two_task_pool_v1/tier_c_v1/independent_remaining_RL_v1/verify_completed_RL_PRIVATE.py'
    source=audit.text(path)
    require(sha(path)==NATIVE_VERIFIER_SHA,'Certified verifier source changed')
    wanted={'resolve_effective_eos_ids','authenticated_eos_ids','decode_tokens'}
    selected=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    require({x.name for x in selected}==wanted,'Missing certified decoder functions')
    tokenizer_path=root/'local_data/models/qwen3vl8b-base/tokenizer.json'
    tok=audit.json(tokenizer_path)
    require(tok['decoder']['type']=='ByteLevel','Certified ByteLevel decoder required')
    symbols=list(range(ord('!'),ord('~')+1))+list(range(0xa1,0xad))+list(range(0xae,0x100))
    codes=symbols[:];offset=0
    for b in range(256):
        if b not in symbols:symbols.append(b);codes.append(256+offset);offset+=1
    byte_decoder={chr(c):b for b,c in zip(symbols,codes)}
    pieces={v:k for k,v in tok['model']['vocab'].items()}
    pieces.update({v['id']:v['content'] for v in tok['added_tokens']})
    certificate=root/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'
    for p in [certificate,tokenizer_path.with_name('config.json'),tokenizer_path.with_name('generation_config.json'),tokenizer_path.with_name('tokenizer_config.json')]:audit.path(p)
    import pathlib
    namespace={'sha':sha,'json':json,'pathlib':pathlib,'byte_decoder':byte_decoder,'id_to_piece':pieces}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    eos=namespace['authenticated_eos_ids'](tokenizer_path.with_name('config.json'),tokenizer_path.with_name('generation_config.json'),tokenizer_path.with_name('tokenizer_config.json'),certificate,certificate_sha,{v:k for k,v in pieces.items()})
    require(eos=={151645,151643},'Certified native EOS union changed')
    namespace['eos_ids']=eos
    native=(root/'../../work/mlx-venv/lib/python3.13/site-packages/mlx_vlm').resolve()
    for name,h in DECODER_SOURCES.items():require(sha(audit.path(native/name))==h,'Certified installed decoder changed')
    return namespace['decode_tokens'],eos


def run_replay(root,destination,public_path):
    """Replay the fixed 18 arms. No evaluation or optimizer state is opened."""
    from mri_rl.vero_category_sampler import build_category_schedule
    from mri_rl.two_task_training import make_grader
    from mri_rl.mlx_gspo_smoke import task_group_is_informative, advantages_from_rewards
    root=Path(root).resolve();destination=Path(destination).resolve();public_path=Path(public_path).resolve()
    require('local_data' in destination.parts and 'root_cause_cpu_v1' in destination.parts,'Private replay destination required')
    require(not destination.exists(),'Refuse overwrite of prior replay')
    require(not public_path.exists(),'Refuse overwrite of prior public replay')
    audit=ReadAudit();pool=root/'local_data/c_two_task_pool_v1';tier=pool/'tier_c_v1'
    train=pool/'train.verl.jsonl';rows=[json.loads(l) for l in audit.text(train).splitlines() if l.strip()]
    require(all(r['extra_info']['internal_split']=='train' for r in rows),'Only frozen train rows admitted')
    source_by_id={r['id']:r for r in rows}
    measurements=audit.json(pool/'native_tokens_v2/measurements_PRIVATE.json')
    require(measurements['split']=='train' and measurements['input_tokens_include_visual'],'Train native input measurements required')
    require(sha(audit.path(root/'scripts/measure_native_training_tokens.py'))==measurements['source_script_sha256'],'Recorded imagepad measurement source changed')
    measures={r['record_sha256']:r for r in measurements['rows']}
    native={r['record_id']:r for r in audit.json(pool/'native_tokens_v2/native_inputs_PRIVATE.json')}
    calibrated_manifest=audit.json(tier/'calibrated_bundle_v1/manifest_PRIVATE.json')
    calibration=audit.json(calibrated_manifest['calibration']['path'])
    require(sha(calibrated_manifest['calibration']['path'])==calibrated_manifest['calibration']['file_sha256'],'Frozen calibration changed')
    require(calibration==calibrated_manifest['calibration']['value'],'Frozen calibration differs')
    # Source hashes are checked against native binding before reading any group.
    inventory=[]
    for arm in ARMS:
        for seed in SEEDS:
            path=(tier/'calibrated_bundle_v1'/f'{arm}_seed{seed}'/'run' if 'calibrated' in arm else tier/f'{arm}_seed{seed}_run_v1')
            initial=audit.json(path/'boundary_0000/receipt.json');binding=initial['binding'];exp=binding['experiment']
            require(exp['arm']['name']==arm and exp['seed']==seed,'Arm/seed binding mismatch')
            require(exp['common']['manifest_sha256']==sha(train),'Training manifest changed')
            require(exp['calibration_sha256']==(hashobj(calibration) if 'calibrated' in arm else None),'Arm frozen calibration binding changed')
            reward_hashes={name:sha(audit.path(root/'src/mri_rl'/name)) for name in ('vero_mri_reward.py','visit_correspondence.py','two_task_training.py','reward_replay.py')}
            require(hashobj(reward_hashes)==exp['common']['reward_source_sha256'],'Frozen composite reward source changed')
            for name,h in reward_hashes.items():require(sha(Path(__file__).with_name(name))==h,'Imported replay callback differs from frozen source')
            for name,expected in [('continuous_rl.py',binding['driver_source_sha256']),('mlx_continuous_gspo.py',exp['common']['native_backend_sha256'])]:
                require(sha(audit.path(root/'src/mri_rl'/name))==expected,'Historical critical source changed: '+name)
            require(binding['configuration']['group_size']==4 and binding['configuration']['max_tokens']==256,'Historical runtime changed')
            sampled=build_category_schedule(rows,mode=exp['arm']['sampling'],seed=seed,max_groups=198,specialist_category=exp['arm'].get('task'))
            require(hashobj(sampled['schedule'])==binding['schedule_sha256'],'Frozen full schedule mismatch')
            require([r['id'] for r in sampled['schedule']]==binding['schedule_record_ids'],'Frozen occurrence inventory mismatch')
            inventory.append((arm,seed,path,binding,sampled['schedule']))
    decode,eos=certified_decoder(root,audit,inventory[0][3]['experiment']['common']['cache_certificate_sha256'])
    image_pad_id=next(tid for tid,piece in decode.__globals__['id_to_piece'].items() if piece=='<|image_pad|>')
    groups=[];partials=[];summary_rows=[];tasks=defaultdict(Counter)
    for arm,seed,path,binding,schedule in inventory:
        boundaries=sorted(path.glob('boundary_[0-9][0-9][0-9][0-9]'))
        previous=None;previous_updates=0;completed_cost=0;charged_partials=set();debt=0;imputed=0;per_task=defaultdict(Counter)
        for index,boundary in enumerate(boundaries):
            receipt=audit.json(boundary/'receipt.json')
            require(boundary.name==f'boundary_{index:04d}' and receipt['completed_groups']==index,'Boundary cursor mismatch')
            require(receipt['binding']==binding and receipt['parent_receipt_sha256']==previous,'Boundary parent/binding mismatch')
            previous=sha(boundary/'receipt.json')
            if index==0:continue
            row=schedule[index-1];task=row['ability'];origin=source_by_id[row['source_record_id']]
            require(receipt['record_id']==receipt['schedule_record_id']==row['id'],'Schedule receipt identity mismatch')
            raw_path=boundary/'rollout_PRIVATE.json';raw=audit.json(raw_path)
            require(sha(raw_path)==receipt['files_sha256']['rollout_PRIVATE.json'],'Raw rollout SHA mismatch')
            require(raw['record_id']==row['id'] and raw['group_index']==index-1,'Raw rollout index mismatch')
            require(raw['image_sha256']==binding['image_sha256'][row['id']],'Raw image binding mismatch')
            cap=native[origin['id']];m=measures[hashobj(origin)]
            require(raw['prompt_token_ids']==cap['input_token_ids'],'Native input tokens changed')
            require(hashlib.sha256(raw['rendered_prompt'].encode()).hexdigest()==raw['prompt_sha256']==cap['rendered_prompt_sha256'],'Native prompt mismatch')
            inp=len(raw['prompt_token_ids'][0]);require(inp==m['input_tokens'],'Measured input token mismatch')
            require(sum(t==image_pad_id for t in raw['prompt_token_ids'][0])==m['visual_tokens'],'Recorded expanded imagepad count mismatch')
            responses=raw['responses'];require(len(responses)==4,'Incomplete historical group')
            lengths=[]
            for response in responses:
                ids=response['token_ids'];require(all(type(t) is int and t>=0 for t in ids),'Invalid native token ID')
                require(response['generation_tokens']==len(ids) and decode(ids)==response['text'],'Certified native decoding mismatch')
                lengths.append(len(ids))
            stats=calibration[task] if 'calibrated' in arm else None
            g=analyze_group([r['text'] for r in responses],lengths,row['reward_model']['ground_truth'],task,stats)
            frozen_comparison=analyze_group([r['text'] for r in responses],lengths,row['reward_model']['ground_truth'],task,calibration[task])
            g['frozen_calibration_comparison']={k:frozen_comparison[k] for k in ('calibrated_accuracy','total','advantages','epsilon_free_advantages','raw_calibrated_max_difference','raw_calibrated_epsilon_free_max_difference')}
            g['eos_terminated_responses']=sum(r['token_ids'][-1] in eos for r in responses)
            g['length_terminated_responses']=sum(r['token_ids'][-1] not in eos and len(r['token_ids'])==256 for r in responses)
            g['unresolved_termination_responses']=sum(r['token_ids'][-1] not in eos and len(r['token_ids'])<256 for r in responses)
            g['capped_missing_answer_close_responses']=sum(n==256 and '</answer>' not in r['text'] for n,r in zip(lengths,responses))
            grader=make_grader(calibration if stats else None)
            grades=[]
            for response in responses:
                grade=grader(row,response['text'],binding['configuration']);grade['overlong']=0.;grade['total']=grade['score'];grades.append(grade)
            require(grades==receipt['grades'],'Historical certified callback grade mismatch')
            for i,grade in enumerate(grades):
                require(grade=={'accuracy':g['calibrated_accuracy'][i],'format':g['format'][i],'raw_accuracy':g['raw_accuracy'][i],'overlong':g['overlong'][i],'score':g['total'][i],'total':g['total'][i]},'Independent reward reconstruction mismatch')
            require(max(abs(a-b) for a,b in zip(g['advantages'],receipt['advantages']))<1e-13,'Recorded advantage mismatch')
            require(max(abs(a-b) for a,b in zip(g['advantages'],advantages_from_rewards(g['total'])))<1e-13,'Native sample SD mismatch')
            require(g['accepted']==receipt['updated']==task_group_is_informative(grades),'Historical acceptance mismatch')
            require(receipt['optimizer_updates']==previous_updates+int(g['accepted']),'Optimizer update counter mismatch')
            previous_updates=receipt['optimizer_updates']
            cost=4*inp+sum(lengths);completed_cost+=cost
            c=per_task[task];c['attempted_groups']+=1;c['accepted_updates']+=g['accepted'];c['completed_rollout_tokens']+=cost
            c['input_tokens_including_visual']+=4*inp;c['generated_tokens']+=sum(lengths);c['measured_visual_tokens']+=4*m['visual_tokens']
            c['recorded_expanded_imagepad_positions_once_per_completed_group']+=m['visual_tokens']
            c['expanded_prompt_imagepad_positions_times_completed_rollouts']+=4*m['visual_tokens']
            c['source_derived_generation_invocations_completed']+=4
            for kind in ('old_logp','current_logp','gradient_logp'):c['source_derived_'+kind+'_model_calls']+=4*int(g['accepted'])
            c['source_derived_teacher_forced_model_calls']+=12*int(g['accepted'])
            c['source_derived_teacher_forced_imagepad_positions']+=12*int(g['accepted'])*m['visual_tokens']
            c['first_gate:'+g['first_gate']]+=1
            for reason in g['failed_gates']:c['overlapping:'+reason]+=1
            c['positive_format_valid_class_mix']+=g['positive_format_valid_class_mix']
            c['full_format_valid_class_mix']+=g['full_format_valid_class_mix']
            if g['accepted']:c['accepted_native_variation_source:'+g['native_accuracy_variation_source']]+=1
            if g['correctness_varies_but_format_rejected']:
                c['format_rejected_native_variation_source:'+g['native_accuracy_variation_source']]+=1
                c['format_rejected_positive_format_valid_class_mix']+=g['positive_format_valid_class_mix']
                c['format_rejected_full_format_valid_class_mix']+=g['full_format_valid_class_mix']
            for key in ('task_accuracy_varies','format_varies','total_reward_varies','informative_advantages','nonzero_advantages','format_only_total_variation','correctness_varies_but_format_rejected','single_box_diagnostic_accuracy_varies','at_generation_cap','native_correct_responses','full_format_responses','eos_terminated_responses','length_terminated_responses','unresolved_termination_responses','capped_missing_answer_close_responses'):c[key]+=g[key]
            included=receipt['ledger']['partial_dirs_included']
            require(included==sorted(set(included)) and charged_partials<=set(included),'Duplicate/disappeared partial ledger debt')
            for name in set(included)-charged_partials:
                require(name.startswith('.pending-') and '/' not in name,'Invalid partial path')
                pp=path/name;reservation=audit.json(pp/'attempt_reservation.json');j=reservation['index'];pr=schedule[j];pt=pr['ability'];pi=measures[hashobj(source_by_id[pr['source_record_id']])]['input_tokens']
                require(0<=j<index and reservation=={'schema':'attempt-reservation-v1','run_binding_sha256':hashobj(binding),'parent_receipt_sha256':sha(boundaries[j]/'receipt.json'),'index':j,'record_id':pr['id'],'image_sha256':binding['image_sha256'][pr['id']],'input_tokens':pi,'group_size':4,'max_tokens':256},'Partial reservation mismatch')
                require(reservation['run_binding_sha256']==hashobj(binding) and reservation['parent_receipt_sha256']==sha(boundaries[j]/'receipt.json'),'Partial reservation parent mismatch')
                full=audit.json(pp/'rollout_PRIVATE.json') if (pp/'rollout_PRIVATE.json').is_file() else None
                progress=audit.json(pp/'rollout_progress_PRIVATE.json') if (pp/'rollout_progress_PRIVATE.json').is_file() else None
                for evidence in (full,progress):
                    if evidence is not None:require(evidence['record_id']==pr['id'] and evidence['group_index']==j and evidence['image_sha256']==binding['image_sha256'][pr['id']],'Partial rollout identity mismatch')
                if full and progress:require(len(progress['responses'])<=len(full['responses']) and all(a['token_ids']==b['token_ids'] for a,b in zip(progress['responses'],full['responses'])),'Partial progress mismatch')
                rr=(full or progress or {}).get('responses',[])
                require(len(rr)<=4 and (not full or len(rr)==4),'Invalid partial response count')
                for response in rr:
                    require(0<len(response['token_ids'])<=256 and all(type(t) is int and t>=0 for t in response['token_ids']),'Partial token inventory mismatch')
                    require(decode(response['token_ids'])==response['text'],'Partial native decoding mismatch')
                measured=len(rr)*pi+sum(len(r['token_ids']) for r in rr);unknown=pi+256 if full is None and len(rr)<4 else 0
                debt+=measured;imputed+=unknown;cpart=per_task[pt];cpart['charged_partial_attempts']+=1;cpart['measured_partial_tokens']+=measured;cpart['imputed_unknown_tokens']+=unknown;cpart['measured_partial_visual_tokens']+=len(rr)*measures[hashobj(source_by_id[pr['source_record_id']])]['visual_tokens']
                partials.append({'arm':arm,'seed':seed,'task':pt,'partial_dir':name,'schedule_index':j,'known_responses':len(rr),'measured_tokens':measured,'imputed_unknown_tokens':unknown,'reservation_sha256':sha(pp/'attempt_reservation.json')})
            charged_partials=set(included)
            require(receipt['ledger']['used_tokens']==completed_cost+debt+imputed and receipt['ledger']['imputed_unknown_tokens']==imputed,'Token accounting mismatch')
            groups.append({'arm':arm,'seed':seed,'task':task,'group_index':index-1,'record_id':row['id'],'source_record_id':origin['id'],'receipt_sha256':previous,'rollout_sha256':sha(raw_path),'input_tokens_per_response':inp,'visual_tokens_per_response':m['visual_tokens'],**g})
        for task in TASKS:
            c=per_task[task]
            for key in ('attempted_groups','accepted_updates','completed_rollout_tokens','input_tokens_including_visual','generated_tokens','measured_visual_tokens','measured_partial_visual_tokens','charged_partial_attempts','measured_partial_tokens','imputed_unknown_tokens','first_gate:accepted','first_gate:format_not_all_full','first_gate:task_accuracy_constant','first_gate:overlong_not_equal','overlapping:format_not_all_full','overlapping:task_accuracy_constant','overlapping:overlong_not_equal','format_rejected_native_variation_source:valid_class_correct_and_wrong','format_rejected_native_variation_source:only_format_invalid_zero_vs_valid_correct','format_rejected_native_variation_source:unknown_or_nonclass_zero_vs_valid_correct','format_rejected_positive_format_valid_class_mix','format_rejected_full_format_valid_class_mix','accepted_native_variation_source:valid_class_correct_and_wrong','accepted_native_variation_source:only_format_invalid_zero_vs_valid_correct','accepted_native_variation_source:unknown_or_nonclass_zero_vs_valid_correct'):c[key]+=0
            c['charged_tokens']=c['completed_rollout_tokens']+c['measured_partial_tokens']+c['imputed_unknown_tokens']
            c['measured_visual_tokens_including_partial']=c['measured_visual_tokens']+c['measured_partial_visual_tokens'];tasks[task].update(c)
        summary_rows.append({'arm':arm,'seed':seed,'completed_groups':len(boundaries)-1,'accepted_updates':previous_updates,'completed_rollout_tokens':completed_cost,'measured_partial_tokens':debt,'imputed_unknown_tokens':imputed,'charged_tokens':completed_cost+debt+imputed,'charged_partial_attempts':len(charged_partials),'terminal_boundary_receipt_sha256':previous,'tasks':{t:dict(per_task[t]) for t in TASKS}})
    require(len(groups)==2286 and sum(g['accepted'] for g in groups)==362,'Original 18-arm inventory mismatch')
    destination.mkdir(parents=True)
    ledger=destination/'group_causes_PRIVATE.jsonl'
    ledger.write_text(''.join(canonical(g)+'\n' for g in groups))
    (destination/'partial_debt_PRIVATE.json').write_text(canonical(partials)+'\n')
    (destination/'read_inventory_PRIVATE.json').write_text(canonical(audit.files)+'\n')
    active=[g for g in groups if g['accepted']]
    report={'schema':'root-cause-reward-replay-v1','analysis_revision':'strict-valid-class-variation-v2','status':'TESTED_CPU','evidence_tier':'REPRODUCED','research_tag':'ENGINEERING_UNBLOCK','groups':len(groups),'responses':4*len(groups),'accepted_updates':len(active),'arm_seed_task_rows':summary_rows,'task_totals':{t:dict(tasks[t]) for t in TASKS},
      'totals':{k:sum(r[k] for r in summary_rows) for k in ('completed_groups','accepted_updates','completed_rollout_tokens','measured_partial_tokens','imputed_unknown_tokens','charged_tokens','charged_partial_attempts')},
      'strict_class_variation_audit':{
        'legacy_392_count_interpretation':'The legacy correctness_varies_but_format_rejected field counts native parser-gated accuracy variation, not necessarily valid correct/wrong categorical labels or recoverable training groups.',
        'format_rejected_native_accuracy_varies_groups':sum(g['correctness_varies_but_format_rejected'] for g in groups),
        'partition':{source:sum(g['correctness_varies_but_format_rejected'] and g['native_accuracy_variation_source']==source for g in groups) for source in ('valid_class_correct_and_wrong','only_format_invalid_zero_vs_valid_correct','unknown_or_nonclass_zero_vs_valid_correct')},
        'accepted_groups_with_mixed_valid_classes':sum(g['accepted'] and g['positive_format_valid_class_mix'] for g in groups),
        'accepted_groups_without_mixed_valid_classes':sum(g['accepted'] and not g['positive_format_valid_class_mix'] for g in groups),
        'accepted_partition':{source:sum(g['accepted'] and g['native_accuracy_variation_source']==source for g in groups) for source in ('valid_class_correct_and_wrong','only_format_invalid_zero_vs_valid_correct','unknown_or_nonclass_zero_vs_valid_correct')},
        'format_rejected_mixed_valid_classes_format_gt_zero':sum(g['correctness_varies_but_format_rejected'] and g['positive_format_valid_class_mix'] for g in groups),
        'format_rejected_mixed_valid_classes_format_equals_one':sum(g['correctness_varies_but_format_rejected'] and g['full_format_valid_class_mix'] for g in groups),
        'definition':'Valid class means existing strict parser gives format>0 and an answer exactly in the declared task class inventory. Correct and wrong are relative to that train record target. The full-format subset additionally requires format==1. Invalid wrappers and nonclass answers remain unknown, without salvage extraction.',
        'partition_rule':'Category a has at least one valid correct and one valid wrong class. Otherwise category b requires every native accuracy-zero response to have format==0. Remaining native-varying groups are category c with positive-format nonclass/malformed answers. Other unknown responses may coexist with category a.',
        'scope':'These are label correctness contrasts in saved responses; they do not establish image dependence, gradients, recoverability under a changed gate, or clinical accuracy.'},
      'acceptance_rule':{'ordered':['format_not_all_full','task_accuracy_constant','overlong_not_equal'],'order_source':'historical Python short-circuit; overlapping reasons independently counted','unchanged':True},
      'advantage_math':{'std':'sample, denominator n-1','epsilon':1e-6,'raw_total':'0.8*raw_accuracy+0.2*format+overlong','calibrated_total':'0.8*logistic(3*(raw_accuracy-p50)/IQR)+0.2*format+overlong','frozen_calibration':calibration,'comparison_scope':'Frozen calibration applied as a counterfactual to the same saved groups from all 18 arms; actual grades use the arm reward binding.','max_active_raw_calibrated_difference':max(g['frozen_calibration_comparison']['raw_calibrated_max_difference'] for g in active),'max_active_epsilon_free_difference':max(g['frozen_calibration_comparison']['raw_calibrated_epsilon_free_max_difference'] for g in active),'binary_support_explanation':'For accepted groups format=1 and overlong is equal. A strictly monotone transform on binary accuracy has exactly two levels and is a positive affine transform on this support. Standardization cancels it without epsilon; epsilon leaves only numerical scale dependence.'},
      'length_contract':{'generation_cap':256,'penalty_starts_after':16384,'maximum_response_length':18432,'buffer':2048,'nonzero_actual_overlong_responses':sum(o!=0 for g in groups for o in g['overlong']),'unequal_actual_overlong_groups':sum('overlong_not_equal' in g['failed_gates'] for g in groups),'at_cap_responses':sum(g['at_generation_cap'] for g in groups),'eos_terminated_responses':sum(g['eos_terminated_responses'] for g in groups),'length_terminated_responses':sum(g['length_terminated_responses'] for g in groups),'unresolved_termination_responses':sum(g['unresolved_termination_responses'] for g in groups),'at_cap_missing_answer_close_responses':sum(g['capped_missing_answer_close_responses'] for g in groups),'causal_claim':'No actual length penalty influenced rewards, advantages or acceptance. Cap-terminated responses and their missing answer closures are counted; a longer-generation counterfactual was not run.'},
      'visual_token_accounting':{
        'status':'RECORDED_EXPANDED_INPUT_IMAGEPAD_POSITIONS_AND_SOURCE_DERIVED_INVOCATIONS',
        'measurement_definition':'The frozen processor-only measurement counts image_pad_id positions in expanded input_token_ids and verifies image_grid product / merge_size squared. This replay independently counts the same positions in every saved completed-group prompt.',
        'completed_group_prompt_imagepad_positions_once':sum(g['visual_tokens_per_response'] for g in groups),
        'completed_rollout_prompt_imagepad_positions_K4':sum(4*g['visual_tokens_per_response'] for g in groups),
        'known_partial_rollout_prompt_imagepad_positions':sum(t['measured_partial_visual_tokens'] for t in tasks.values()),
        'legacy_measured_visual_tokens_field':'K4 times the recorded per-prompt imagepad positions, not once per group, not measured vision-encoder compute.',
        'charged_input_tokens_include_imagepad_positions':True,'imagepad_positions_not_added_again_to_token_ledger':True,
        'source_derived_invocation_inventory':{
          'completed_generation_invocations':4*len(groups),
          'old_logp_model_calls':4*len(active),'current_logp_model_calls':4*len(active),
          'gradient_logp_model_calls':4*len(active),'total_teacher_forced_model_calls':12*len(active),
          'scope':'SOURCE_DERIVED from the exact bound NativeGSPOBackend: four generate invocations per completed group; accepted updates call _logps four times for old, four for current, four inside value_and_grad. Each _logps source call passes saved pixel_values. Counts describe Python call structure, not runtime image-encoder operation telemetry, caching, FLOPs, or time.',
          'known_partial_generation_invocations_with_saved_response':sum(p['known_responses'] for p in partials)},
        'actual_total_visual_encoder_tokens_or_FLOPs':'NOT_VERIFIED',
        'partial_unknown_visual_work':'NOT_VERIFIED'},
      'execution':{'models_loaded':0,'inference_calls':0,'optimizer_updates_performed':0,'weights_or_optimizer_tensors_read':0,'images_read':0,'dev_or_test_content_reads':0,'native_imports':0},
      'source_sha256':{'replay_module':sha(__file__),'training_manifest':sha(train),'native_measurements':sha(pool/'native_tokens_v2/measurements_PRIVATE.json'),'certified_verifier':sha(root/'local_data/c_two_task_pool_v1/tier_c_v1/independent_remaining_RL_v1/verify_completed_RL_PRIVATE.py'),'private_group_ledger':sha(ledger),'private_read_inventory':sha(destination/'read_inventory_PRIVATE.json'),'private_partial_debt':sha(destination/'partial_debt_PRIVATE.json'),'read_inventory_canonical':hashobj(audit.files)},
      'native_decoding':{'method':'AST reuse of certified independent verifier functions; exact installed loader/streaming decoder/dispatch sources checked','effective_eos_ids':sorted(eos),'eos_not_assumed_final':True,'all_recorded_token_ids_decoded':True},
      'correction_decision':{'definite_historical_implementation_bug_reproduced':False,'training_reward_gate_change':'NOT_APPLIED','separate_task_advantage_diagnostic':'A format-only total reward can produce nonzero total advantages while task accuracy is constant; its historical optimizer gate rejects it. This is a mathematical diagnostic and does not establish a training defect or accuracy fix.'},
      'mission_alignment_check':{'main_question_link':'Reconstruct effective task exposure and reward information in existing Vero-to-MRI transfer arms.','claim_and_falsifier':'Every saved train group must reproduce exact grade, advantage, gate, cursor and charged ledger; any mismatch aborts.','change_isolation':'New offline analyzer only; historical source, rewards, groups, weights, optimizer and RNG remain frozen.','evidence_provenance':'2286 native train rollout groups with SHA-bound receipts and certified decoding.','shortcut_and_baseline':'Accuracy variation is native parser-gated task reward, not visual dependence or generalization.','resources_and_decision':'CPU only, one process, no model loading; diagnostic explanation only.'},
      'limitations':['No optimizer numerical verification is repeated; saved update counters and historical receipt bindings are replayed.','No internal-development, final100, heldout response or test content is read. No model or checkpoint selection occurs.','No clinical accuracy improvement, visual dependence, model generalization, official Vero parity, or causal reward-calibration benefit is established.','Single-box correctness is an explicit format-agnostic diagnostic limited to unambiguous categorical boxes; absent/ambiguous boxes are unknown, never inferred from prose and never admitted to training.','Partial recovery debt is charged once per recorded pending directory; imputed unknown work is a conservative charge, not measured consumption.','CROSS_MODEL_REVIEW=NOT_VERIFIED for this replay.']}
    public_path.parent.mkdir(parents=True,exist_ok=True)
    require(not public_path.exists(),'Refuse overwrite of prior public replay')
    public_path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report
