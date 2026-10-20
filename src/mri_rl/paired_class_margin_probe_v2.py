"""Fresh read-only native class-margin VJP probe;CPU implementation,NOT_RUN.

No inherited human permission. Shared VJP computes two F32 canonical teacher
forces and one correct-minus-wrong MEAN-margin gradient per image. All native
imports remain inside explicitly authorized execution functions. Never call an
optimizer update or perturb parameters;Adam previews are pure array arithmetic.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from mri_rl.paired_z045_f32_backend_v1 import f32_response_logps_observed
from mri_rl.paired_z045_matched_f32_eval_v1 import score_input_positions,label_scores,common_prefix_alignment
from mri_rl import paired_interaction_signal_v1 as v1
from mri_rl import paired_classification_signal_v2 as v2


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REQUIRED_DIRECT_PROBE_SOURCES=(
    'src/mri_rl/paired_class_margin_probe_v2.py','scripts/run_paired_class_margin_probe_v2.py',
    'tests/test_paired_class_margin_probe_v2.py','src/mri_rl/paired_z045_matched_f32_eval_v1.py',
    'src/mri_rl/paired_z045_evaluation_v1.py','src/mri_rl/answer_likelihood.py',
    'src/mri_rl/paired_signal_candidate_probe_v1.py','src/mri_rl/paired_classification_signal_v2.py',
    'src/mri_rl/paired_interaction_signal_v1.py')


def validate_source_closure(sources,*,root):
    require(all(str(Path(root)/relative) in sources for relative in REQUIRED_DIRECT_PROBE_SOURCES),'Direct probe source missing from closure')
    require(all(hash_file(path)==expected for path,expected in sources.items()),'Source closure byte hash changed')
    return True


def budget():
    return {'contexts':4,'full_TF_forwards':11,'teacherforced_positions':4675,'expected_teacherforced_positions':2868,
        'backward_calls':4,'vision_TF_forwards':11,'model_loads':1,'maximum_seconds':1200,
        'generation_calls':0,'optimizer_updates':0,'parameter_perturbations':0,'simultaneous_image_contexts':1}


class Counter:
    def __init__(self):self.values={'full_TF_forwards':0,'teacherforced_positions':0,'backward_calls':0}
    def forward(self,positions):
        require(type(positions) is int and 0<positions<=425 and self.values['full_TF_forwards']<11 and self.values['teacherforced_positions']+positions<=4675,'Frozen11TF/4675position cap exceeded')
        self.values['full_TF_forwards']+=1;self.values['teacherforced_positions']+=positions
    def backward(self):
        require(self.values['backward_calls']<4,'Frozen4margin backward cap exceeded');self.values['backward_calls']+=1


def readonly_adam_leaf(gradient,moment1,moment2,*,learning_rate,betas,eps,bias_correction,step,ops):
    """Installed MLX Adam.apply_single formula,without state assignment/step().

    delta is the proposed parameter change INCLUDING lr and existing momentum.
    incremental_delta subtracts the same Adam61 preview with new g=0. Neither is
    a finite evaluated parameter move or an observed change in model accuracy.
    """
    require(len(betas)==2 and all(math.isfinite(float(b)) and 0<=float(b)<1 for b in betas) and math.isfinite(float(learning_rate)) and learning_rate>0 and math.isfinite(float(eps)) and eps>=0 and bias_correction is False,'Only explicit current uncorrected Adam profile is supported')
    require(type(step) is int and step>=0 and gradient.shape==moment1.shape==moment2.shape,'Adam preview inventory/counter mismatch')
    def finite(x):return bool(ops.all(ops.isfinite(x)).item())
    require(all(finite(x) for x in (gradient,moment1,moment2)) and bool(ops.all(moment2>=0).item()),'Adam preview nonfinite input or negative second moment')
    lr=ops.array(learning_rate,dtype=gradient.dtype);b1,b2=betas
    m=b1*moment1+(1-b1)*gradient;v=b2*moment2+(1-b2)*ops.square(gradient)
    require(finite(lr) and finite(m) and finite(v),'Adam prospective moment overflow/nonfinite')
    baseline=-lr*(b1*moment1)/(ops.sqrt(b2*moment2)+eps)
    delta=-lr*m/(ops.sqrt(v)+eps)
    increment=delta-baseline
    require(all(finite(x) for x in (baseline,delta,increment)),'Adam preview nonfinite delta/baseline/increment')
    return {'delta':delta,'baseline_delta':baseline,'incremental_delta':increment,'preview_step':step+1}


def auxiliary_gradient_coefficients(scores):
    """Coefficients of G_mF and G_mT;alpha is applied once by the caller."""
    out={}
    for name,module in [('v1',v1),('v2',v2)]:
        d=module.score_derivatives(scores)
        require(all(abs(row[0]+row[1])<=1e-12 for row in d),'Not a two-class margin-only derivative')
        out[name]=[d[0][0],d[1][1]]
    return out


def gradient_aliases(gradients):
    def neg(g):return {k:neg(v) for k,v in g.items()} if isinstance(g,dict) else -g
    return {'SWAPPED_F':neg(gradients['T']),'SWAPPED_T':neg(gradients['F']),
        'BLANK_F':gradients['BLANK'],'BLANK_T':neg(gradients['BLANK'])}


def authorize(scope,human,review):
    digest=canonical(scope)
    require(human.get('authorized') is True and human.get('action')=='readonly_class_margin_v2_probe' and human.get('scope_sha256')==digest and human.get('human_instruction_sha256'),'Fresh explicit human margin-probe authorization required;priorone-shot permission consumed')
    require(review.get('scope_sha256')==digest and review.get('verdict') in ['PASS','PASS_WITH_LIMITATIONS'] and review.get('reviewer_id') and review.get('reviewer_kind') in ['isolated_codex','claude_code'],'Fresh independent exact-scope review required')
    return True


def serial_margin_vjp(backend,inputs,token_plan,truth_index,*,before,charge_forward,charge_backward):
    """Two canonical F32 TF/image→one native mean-margin VJP,504keys at runtime.

    Prefix/suffix/EOS are excluded. Mean/sum/first values share these forwards;
    only mean has a measured gradient. No tensors are detached from the scalar
    objective. Detached copies are retained solely for score evidence.
    """
    require(truth_index in [0,1] and len(token_plan['labels'])==2,'Two canonical candidates and explicit correct class required')
    mx=backend.mx;P=int(inputs['input_ids'].shape[-1]);prefix=token_plan['prefix'];captured=[];evidence=[]
    def objective(model):
        means=[]
        for ids in token_plan['labels']:
            before('margin_candidate_F32_TF');charge_forward(score_input_positions(P,len(prefix),len(ids)))
            lp=f32_response_logps_observed(backend,prefix+ids,inputs,evidence=evidence)
            captured.append(mx.stop_gradient(lp));means.append(mx.mean(lp[len(prefix):]))
        return means[truth_index]-means[1-truth_index]
    before('mean_margin_VJP');charge_backward()
    vg=backend.nn.value_and_grad(backend.model,objective);value,gradient=vg(backend.model);mx.eval(value,gradient,captured)
    require(len(captured)==2 and math.isfinite(float(value.item())),'Exactly two finite margin-candidate forwards required')
    flat=dict(backend.tree_flatten(gradient));weights=dict(backend.tree_flatten(backend.model.trainable_parameters()))
    require(set(flat)==set(weights) and all(v.shape==weights[k].shape and bool(mx.all(mx.isfinite(v)).item()) for k,v in flat.items()),'Complete finite trainable native margin gradient required')
    raw=[v.tolist() for v in captured];alignment=common_prefix_alignment([v[:len(prefix)] for v in raw])
    scores=[label_scores(v,len(ids)) for v,ids in zip(raw,token_plan['labels'])]
    sign=1 if truth_index==0 else -1
    margins={k:sign*(scores[0][k]-scores[1][k]) for k in ['mean','sum','first']}
    require(abs(float(value.item())-margins['mean'])<=.002,'VJP scalar and detached mean score differ')
    result={'gradient':backend.tree_map(lambda g:g.astype(mx.float32),gradient),'mean_scores':[s['mean'] for s in scores],
        'per_label':scores,'margins':margins,'common_prefix_alignment':alignment,'dtype_context_evidence':evidence,
        'sum_first_directional_gradients':'UNKNOWN_NOT_MEASURED','truth_index':truth_index}
    del vg,value,captured,gradient,objective;return result


ROOT=Path(__file__).resolve().parents[2]
REFERENCE_SCOPE_SHA='834be758cb6fc660d498b0e1a9f909dbf622647956accfeb0d7d7aab7dc651b2'
REFERENCE_COMPLETE_SHA='252d44fb879ef8926b4dda301168265be6702641d37c44554fe5f306a6f1a911'
POWER_EXCEPTION={'scope':'paired_class_margin_probe_v2','minimum_battery_percent':60,'battery_threshold_applies_on_AC_too':True,
    'global_training_gate_changed':False,'status':'PROPOSED_REQUIRES_FRESH_EXACT_HUMAN_AUTHORIZATION','prior_permission_consumed':True}


def prepare(spec):
    """Small TRAIN metadata/source inspection only;no weights/Adam arrays loaded."""
    import copy
    from mri_rl import paired_z045_continuation_entry_v1 as entry
    from mri_rl import paired_signal_candidate_probe_v1 as old
    for key,expected in spec['file_sha256'].items():require(entry.digest(spec[key])==expected,'Frozen probe input changed: '+key)
    require(entry.digest(spec['reference_scope'])==REFERENCE_SCOPE_SHA and entry.digest(spec['reference_complete'])==REFERENCE_COMPLETE_SHA,'Only preserved actual four-context reference accepted')
    prior=json.loads(Path(spec['reference_scope']).read_text());complete=json.loads(Path(spec['reference_complete']).read_text());runtime=json.loads(Path(spec['runtime_spec']).read_text())
    require(complete['actual_calls']=={'full_TF_forwards':12,'teacherforced_positions':2811} and complete['backward_calls']==complete['optimizer_updates']==0,'Reference is not the completed actual readonly probe')
    runtime_inputs=old.runtime_artifact_bindings(runtime);source=prior['contexts_PRIVATE'];contexts=[];refs={}
    for name,label in [('F_REAL','FLAIR'),('T_REAL','contrast-enhanced T1')]:
        found=[c for c in source if c['row_PRIVATE']['ability']=='mri_sequence' and c['source_truth']==label and not c['no_image']]
        require(len(found)==1,'Original fixed modality context ambiguous');context=copy.deepcopy(found[0]);oldname=context['name'];context['name']=name
        old.verify_context(context);contexts.append(context);refs[name]=next(r for r in complete['contexts'] if r['context']==oldname)
    context=copy.deepcopy(next(c for c in source if c['name']=='DALLAS_REAL'));old.verify_context(context);contexts.append(context)
    refs['DALLAS_REAL']=next(r for r in complete['contexts'] if r['context']=='DALLAS_REAL')
    a,b=[c['row_PRIVATE'] for c in contexts[:2]]
    require(a['extra_info']['patient_key']==b['extra_info']['patient_key'] and a['extra_info']['question']==b['extra_info']['question'] and a['images']!=b['images'],'Same original TRAIN patient/level opposite-modality pair required')
    require(all(c['row_PRIVATE']['extra_info']['internal_split']=='train' for c in contexts),'Only original TRAIN inputs permitted')
    blank=copy.deepcopy(contexts[0]);blank['name']='BLANK_ZERO_NORMALIZED_PIXELS';blank['source_truth']=None;blank['zero_normalized_pixels']=True;contexts.append(blank)
    derived=entry.cache_metadata(runtime)['expected_binding'];require(derived==prior['derived_cache'],'Certified Adam60 cache metadata changed')
    parent=entry.parent_metadata(runtime['parent_checkpoint'],runtime['parent_receipt_sha256']);require(parent['updates']==60,'Exact preserved Adam60 required')
    manifest=json.loads((Path(runtime['parent_checkpoint'])/'optimizer/manifest.json').read_text())
    moment_inventory=manifest_moment_inventory(manifest['layout'])
    require(moment_inventory==manifest['bindings']['trainable_signature'] and len(moment_inventory)==504,'Actual Adam60 manifest moment key/shape/dtype inventory differs')
    sources=entry.source_closure(runtime)
    for path in [Path(__file__),ROOT/'scripts/run_paired_class_margin_probe_v2.py',ROOT/'tests/test_paired_class_margin_probe_v2.py',
        ROOT/'src/mri_rl/paired_classification_signal_v2.py',ROOT/'src/mri_rl/paired_interaction_signal_v1.py',
        ROOT/'src/mri_rl/paired_signal_candidate_probe_v1.py',ROOT/'src/mri_rl/paired_z045_matched_f32_eval_v1.py',
        ROOT/'src/mri_rl/paired_z045_evaluation_v1.py',ROOT/'src/mri_rl/answer_likelihood.py',Path(spec['optimizer_source'])]:sources[str(path)]=entry.digest(path)
    validate_source_closure(sources,root=ROOT)
    require(spec['power_exception']==POWER_EXCEPTION,'Fresh local power proposal changed')
    output=Path(spec['output']).resolve();require(output.is_relative_to(ROOT/'local_data/paired_class_margin_probe_v2'),'Isolated new private probe output required')
    return {'schema':'paired-class-margin-probe-v2','status':'CPU_PREPARED_NOT_APPROVED_NOT_RUN','human_permission_present':False,'native_approved':False,'training_authorized':False,
        'contexts_PRIVATE':contexts,'reference_scores_PRIVATE':refs,'token_plans_PRIVATE':prior['token_plans_PRIVATE'],'parent':parent,
        'parent_moment_inventory_sha256':canonical(moment_inventory),'parent_moment_count':len(moment_inventory),'parent_moment_metadata_matches_actual_trainable_signature':True,
        'source_sha256':sources,'bound_artifact_sha256':{**{str(spec[k]):v for k,v in spec['file_sha256'].items()},**runtime_inputs},
        'derived_cache':derived,'budget':budget(),'output':str(output),'power_exception':dict(POWER_EXCEPTION),
        'numeric_contract':'Three real contexts one mean-marginVJP each(2TF),align everylabel logp against preserved canonical reference<=.002;newblank2candidate+oneprefixoracle3TF,thenoneblankmarginVJP2TF. ExistingfunctionalF32scorer only;no re-run probability audit/greedy.',
        'gradient_comparison_contract':'F64streaming CPUcosine/normdifference:v2REAL-vs-BLANK,v2REAL-vs-SWAPPED,v1REAL-vs-v2REAL,g_mF-vs-negative_g_mT;zero-normcosineUNDEFINED;noadditionalmodelcalls.',
        'gradient_contract':'Only correct-minus-wrong labelMEAN native margin gradients;canonicalprefix/suffix/EOS excluded.504actualLoRAkeys;one image context at a time;two canonical candidate graphs in that context. Sum/first values only,directionsUNKNOWN.',
        'aliases':'SWAPPED_FIXED_LABEL:swap stored F/T score rows and negate/reorder stored correct-margin grads;blanksamepixel pair uses oneG and-G;zeroadditionalforwards/backwards.',
        'directional_contract':'PurechainruleL1/L2 aux gradients;report-.1gEuclidean and actualuncorrectedAdam61(.1g)parameterchange using preservedAdam60moments;alsoAdam61(g0)baselineanddifference;dotwithmF/mT/mDallas. No optimizer.step/parameterperturbation.',
        'state_contract':'All1942parameters+fullAdam+Python/MLXRNG fingerprints unchanged BEFORE any restore;restore same savedstate/RNG thenverify;release<128MiB.',
        'resources':'One model;1200soperation-startbefore/afterholds;originalglobalflock/memory/swap/thermal/otherheavyprotections;freshscopedbattery>=60onACorBATTERYproposal;noautomaticretry;globaltraininggatesunchanged.',
        'missing_authorization':'Prior one-shot consumed;requires NEW explicit human permission and independent exact-scope review before model load.',
        'mission_alignment':{'tag':'ENGINEERING_UNBLOCK','evidence_tier':'PROPOSED_NOT_RUN','link':'Test whether proposed supervisedauxiliary pushes both native class margins and assess Dallas interference before any future training.',
            'falsifier':'A negativeclassdirection,blankprior-onlygradient orDallasinterference can rejectanauxiliary;CPUscore-space encouragement doesnotpredictparameter-space benefit.'},
        'model_loads_this_preparation':0,'backward_calls_this_preparation':0,'generation_calls':0,'optimizer_updates':0,'holdout_or_final_reads':0,
        'limitations':['NativeVJP/Adam-direction numbers are NOT_RUN;CPUfixturesare notnativequalification.','Onlymean gradientsmeasured;sum/first directional effects UNKNOWN.','LocalEuclidean/Adam previews are not finite evaluated steps,accuracy gains,GSPO+auxjointupdate qualification or independentgeneralization.','Blank normalizedpixels are OOD with retainedimagegrid,not no-image/null;never assign them visual truth.']}


def _moments(state):
    out={}
    def visit(node,path=''):
        if isinstance(node,dict):
            if 'm' in node and 'v' in node:out[path]=(node['m'],node['v'])
            else:
                for k,v in node.items():visit(v,(path+'.'+k).lstrip('.'))
        elif isinstance(node,(list,tuple)):
            for k,v in enumerate(node):visit(v,(path+'.'+str(k)).lstrip('.'))
    visit(state);return out


def _score_alignment(actual,reference):
    for a,b in zip(actual['per_label'],reference['per_label']):
        require(a['token_count']==b['token_count'],'Canonical token lengths changed')
        require(max(abs(x-y) for x,y in zip(a['label_logps_PRIVATE'],b['label_logps_PRIVATE']))<=.002,'Preserved realcontext canonical logp alignment exceeds.002')
    return {'tolerance':.002,'per_label_logps_aligned':True,'gradient_parity_not_established_by_scores':True}


def _analyze_directions(backend,results,initial_adam):
    """Array-only post-VJP analysis;never assign optimizer or model state."""
    import numpy as np
    mx=backend.mx;flat={name:dict(backend.tree_flatten(r['gradient'])) for name,r in results.items()};keys=set(flat['F_REAL'])
    require(len(keys)==504 and all(set(g)==keys for g in flat.values()),'Four complete504 native gradients required')
    moments=_moments(initial_adam);require(set(moments)==keys,'Actual Adam moment paths differ from trainable gradient names')
    opt=backend.optimizer;profile={'learning_rate':float(opt.learning_rate.item()),'betas':tuple(opt.betas),'eps':opt.eps,'bias_correction':opt.bias_correction,'step':int(initial_adam['step'].item())}
    require(profile=={'learning_rate':profile['learning_rate'],'betas':(.9,.999),'eps':1e-8,'bias_correction':False,'step':60} and math.isclose(profile['learning_rate'],1e-5,rel_tol=1e-7),'Actual optimizer profile differs from reviewed Adam60')
    scores=[results['F_REAL']['mean_scores'],results['T_REAL']['mean_scores']];blank=results['BLANK_ZERO_NORMALIZED_PIXELS']['mean_scores'];coeff_real=auxiliary_gradient_coefficients(scores)
    conditions={'REAL_PAIR':(scores,'F_REAL','T_REAL',1,1),'SWAPPED_FIXED_LABEL_PAIR':([scores[1],scores[0]],'T_REAL','F_REAL',-1,-1),
        'BLANK_OPPOSITE_LABEL_PAIR':([blank,blank],'BLANK_ZERO_NORMALIZED_PIXELS','BLANK_ZERO_NORMALIZED_PIXELS',1,-1)}
    targets={'mF':'F_REAL','mT':'T_REAL','mDallas':'DALLAS_REAL'};table={}
    def cpu(x):return np.asarray(x,dtype=np.float64).ravel()
    for condition,(matrix,fname,tname,fsign,tsign) in conditions.items():
        table[condition]={'scores_PRIVATE':matrix,'loss_values':{'v1':v1.pair_loss(matrix),'v2':v2.classification_loss(matrix)},'versions':{}}
        for version,(cf,ct) in auxiliary_gradient_coefficients(matrix).items():
            dots={kind:{t:0. for t in targets} for kind in ['Euclidean_minus_alpha_g','Adam61_total_delta','Adam61_zero_new_gradient_baseline','Adam61_increment_over_zero_gradient']};norms={kind:0. for kind in dots};g2=0.
            for key in sorted(keys):
                g=cf*fsign*flat[fname][key]+ct*tsign*flat[tname][key];scaled=v2.ALPHA*g;m,v=moments[key]
                preview=readonly_adam_leaf(scaled,m,v,ops=mx,**profile);vectors={'Euclidean_minus_alpha_g':-scaled,'Adam61_total_delta':preview['delta'],
                    'Adam61_zero_new_gradient_baseline':preview['baseline_delta'],'Adam61_increment_over_zero_gradient':preview['incremental_delta']}
                mx.eval(g,vectors);gg=cpu(g);g2+=float(gg@gg)
                for kind,vec in vectors.items():
                    x=cpu(vec);norms[kind]+=float(x@x)
                    for target,name in targets.items():dots[kind][target]+=float(x@cpu(flat[name][key]))
                require(math.isfinite(g2) and all(math.isfinite(n) for n in norms.values()) and all(math.isfinite(n) for row in dots.values() for n in row.values()),'Directional scalar dot/norm overflow/nonfinite')
            for kind in dots:
                dots[kind]['L1_real_pair']=coeff_real['v1'][0]*dots[kind]['mF']+coeff_real['v1'][1]*dots[kind]['mT']
                dots[kind]['L2_real_pair']=coeff_real['v2'][0]*dots[kind]['mF']+coeff_real['v2'][1]*dots[kind]['mT']
                require(all(math.isfinite(n) for n in dots[kind].values()),'Loss-direction scalar overflow/nonfinite')
            table[condition]['versions'][version]={'gradient_coefficients':[cf,ct],'unscaled_aux_gradient_l2':math.sqrt(g2),
                'alpha':v2.ALPHA,'directions':dots,'parameter_direction_l2':{k:math.sqrt(n) for k,n in norms.items()},
                'sum_first_margin_directions':'UNKNOWN_NOT_MEASURED','Adam_preview_is_aux_only_not_joint_GSPO_step':True}
    matrices={name:values[0] for name,values in conditions.items()}
    coefficients={name:auxiliary_gradient_coefficients(matrix) for name,matrix in matrices.items()}
    def combo(condition,version,key):
        _,f,t,fs,ts=conditions[condition];cf,ct=coefficients[condition][version]
        return cf*fs*cpu(flat[f][key])+ct*ts*cpu(flat[t][key])
    comparisons={}
    for label,left,right in [
        ('v2_REAL_vs_BLANK',('REAL_PAIR','v2'),('BLANK_OPPOSITE_LABEL_PAIR','v2')),
        ('v2_REAL_vs_SWAPPED',('REAL_PAIR','v2'),('SWAPPED_FIXED_LABEL_PAIR','v2')),
        ('v1_REAL_vs_v2_REAL',('REAL_PAIR','v1'),('REAL_PAIR','v2'))]:
        comparisons[label]=compare_gradient_vectors((key,combo(*left,key),combo(*right,key)) for key in sorted(keys))
    comparisons['g_mF_vs_negative_g_mT']=compare_gradient_vectors((key,cpu(flat['F_REAL'][key]),-cpu(flat['T_REAL'][key])) for key in sorted(keys))
    return {'conditions':table,'gradient_comparisons':comparisons,'actual_optimizer_profile':profile,'step_executed':False,'moments_written':False,'Adam_delta_is_update_vector_before_parameter_rounding':True,
        'interpretation':'Local directional dotproducts of true mean-margin gradients with Euclidean/Adam-preview directions;notfinite-steporaccuracyobservations. Adamzero-gradientbaseline retains oldmomentum;difference isolates newaux contribution under this preview.'}


def run_resident(backend,scope,out,counter,before):
    from mri_rl import paired_signal_candidate_probe_v1 as old
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_f32_qualification_v1 import _state
    mx=backend.mx;out=Path(out);params=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));state=_state(backend,utility)
    require(len(params)==1942,'All1942 parameters required');initial_adam=backend.optimizer_state();initial_rng=backend.rng_state();results={};blank_inputs=None;blank_fp=None;blank_prompt_sha=None
    language=backend.model.language_model;metadata=(language._rope_deltas,language._position_ids)
    def integrity():require(params==utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters()))) and state==_state(backend,utility),'Readonly VJP changed parameters/Adam/Python/MLXRNG before restore')
    try:
        for context in scope['contexts_PRIVATE'][:3]:
            before('context_prepare');old.verify_context(context);prompt,inputs,fp=old.prepare_inputs(backend,context,utility)
            reference=scope['reference_scores_PRIVATE'][context['name']]
            require(fp==reference['inputs_fingerprints_PRIVATE'] and hashlib.sha256(prompt.encode()).hexdigest()==reference['prompt_sha256'],'Original realcontext input/prompt identity differs')
            plan=scope['token_plans_PRIVATE'][context['row_PRIVATE']['ability']];truth=plan['label_order'].index(context['source_truth'])
            result=serial_margin_vjp(backend,inputs,plan,truth,before=before,charge_forward=counter.forward,charge_backward=counter.backward)
            result['inputs_fingerprints_PRIVATE']=fp;result['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
            result['reference_alignment']=_score_alignment(result,reference);flat=dict(backend.tree_flatten(result['gradient']));require(len(flat)==504,'Actual504LoRAgradient required')
            backend.save_arrays(out/(context['name']+'_margin_gradient_PRIVATE.safetensors'),flat);utility._fsync(out/(context['name']+'_margin_gradient_PRIVATE.safetensors'))
            utility.durable(out/(context['name']+'_scores_PRIVATE.json'),{k:v for k,v in result.items() if k!='gradient'});results[context['name']]=result
            if context['name']=='F_REAL':
                blank_inputs={**inputs,'pixel_values':mx.zeros_like(inputs['pixel_values'])};mx.eval(blank_inputs)
                blank_fp=utility._fingerprints(backend,blank_inputs);blank_prompt_sha=result['prompt_sha256']
                utility.validate_condition_fingerprints(fp,blank_fp)
            inputs=None;gc.collect();mx.clear_cache();integrity()
        blank_context=scope['contexts_PRIVATE'][3];plan=scope['token_plans_PRIVATE']['mri_sequence']
        value_evidence=old.score_context(backend,blank_context,blank_inputs,plan,type('CounterAdapter',(),{'charge':staticmethod(counter.forward)})(),before)
        value_evidence['inputs_fingerprints_PRIVATE']=blank_fp;value_evidence['prompt_sha256']=blank_prompt_sha
        result=serial_margin_vjp(backend,blank_inputs,plan,0,before=before,charge_forward=counter.forward,charge_backward=counter.backward)
        result['inputs_fingerprints_PRIVATE']=blank_fp;result['prompt_sha256']=blank_prompt_sha
        result['blank_value_oracle_alignment']=_score_alignment(result,value_evidence);results[blank_context['name']]=result
        path=out/'BLANK_ZERO_NORMALIZED_PIXELS_margin_gradient_PRIVATE.safetensors';backend.save_arrays(path,dict(backend.tree_flatten(result['gradient'])));utility._fsync(path)
        utility.durable(out/'blank_oracle_scores_PRIVATE.json',value_evidence)
        utility.durable(out/'BLANK_ZERO_NORMALIZED_PIXELS_scores_PRIVATE.json',{k:v for k,v in result.items() if k!='gradient'})
        integrity();before('readonly_Adam61_direction_analysis');analysis=_analyze_directions(backend,results,initial_adam);utility.durable(out/'direction_analysis_PRIVATE.json',analysis)
        require(counter.values=={'full_TF_forwards':11,'teacherforced_positions':2868,'backward_calls':4},'Actual calls differ from frozen exact11TF/2868positions/4VJP plan')
        integrity()
        return {'status':'ACTUAL_MARGIN_VJP_READONLY_COMPLETE','actual_calls':counter.values,'analysis':analysis,'generation_calls':0,'optimizer_updates':0,'parameter_perturbations':0,'sum_first_gradient_directions':'UNKNOWN_NOT_MEASURED'}
    finally:
        after_params=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));after_state=_state(backend,utility)
        utility.durable(out/'before_restore_integrity_PRIVATE.json',{'all_parameters_before':params,'all_parameters_after':after_params,'state_before':state,'state_after':after_state,'unchanged':params==after_params and state==after_state})
        require(params==after_params and state==after_state,'Readonly exit integrity failed before restoration')
        backend.restore(initial_adam,initial_rng);restored=_state(backend,utility)
        utility.durable(out/'restored_state_PRIVATE.json',{'initial':state,'restored':restored,'unchanged':state==restored});require(state==restored,'Postrestore state differs')
        language._rope_deltas,language._position_ids=metadata
        results.clear();blank_inputs=None;gc.collect();mx.clear_cache()


def execute(spec,scope,*,human_path,human_sha,review_path,review_sha):
    """Dormant entry. Fresh human+review evidence is checked before native imports."""
    import fcntl
    import os
    import importlib.metadata
    from dataclasses import asdict
    from types import SimpleNamespace
    from mri_rl import paired_z045_continuation_entry_v1 as entry
    from mri_rl import paired_signal_candidate_probe_v1 as old
    from mri_rl.paced_resources import PacedGate,PACED_POLICY
    require(entry.digest(human_path)==human_sha and entry.digest(review_path)==review_sha,'Fresh permission/review file SHA mismatch')
    authorize(scope,json.loads(Path(human_path).read_text()),json.loads(Path(review_path).read_text()))
    require(prepare(spec)==scope,'Frozen scope source/input changed')
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'One-thread caps required')
    out=Path(scope['output']);require(not out.exists(),'One fresh run only;no retry')
    # Imports below this point are unreachable from CPU prepare without evidence.
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.experiment_contract import load_contract
    runtime=json.loads(Path(spec['runtime_spec']).read_text());started=time.monotonic();counter=Counter();resident=[None]
    class FreshScopeBatteryGate(PacedGate):
        def before_unit(self,name,loading=False):
            for _ in range(PACED_POLICY['wait_rechecks']):
                sample=self.sampler();level,reasons=self.policy.observe(sample)
                # Reuse tested POWER DECISION ARITHMETIC only,not old permission.
                action,reasons=old.probe_power_decisions(sample,loading=loading,supervisor_reasons=reasons)
                with self.path.open('a') as f:
                    f.write(json.dumps({'unit_id':name,'action':action,'reasons':reasons,'supervisor_level':level,'sample':asdict(sample),
                        'fresh_authorized_scope_sha256':canonical(scope),'fresh_human_file_sha256':human_sha,'local_power_proposal':POWER_EXCEPTION},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
                if action=='CONTINUE':return sample
                if action=='STOP':raise RuntimeError('Fresh margin probe stopped: '+','.join(reasons))
                self.sleep(PACED_POLICY['wait_seconds'])
            raise RuntimeError('Resource HOLD persists;no automatic restart')
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);require(prepare(spec)==scope,'Source/input changed at exclusive lock');out.mkdir(parents=True)
        gate=FreshScopeBatteryGate(out/'resources_PRIVATE.jsonl');cache={'dir':runtime['cache_dir']}
        def before(name):
            require(time.monotonic()-started<=1200,'1200s operation-start deadline exhausted');gate.before_unit(name,loading=is_loading_unit(name,cache));require(time.monotonic()-started<=1200,'Deadline exhausted during resource HOLD')
            if name=='native_model_load':
                old.runtime_artifact_bindings(runtime)
                for c in scope['contexts_PRIVATE']:old.verify_context(c)
        def run():
            require(importlib.metadata.version('mlx')=='0.32.4' and importlib.metadata.version('mlx-vlm')=='0.7.8','Installed native versions differ')
            derived=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),cache_certificate_sha256=runtime['cache_certificate_sha256'],model=Path(runtime['model'])))
            require(derived==scope['derived_cache'],'Actual certified base cache differs')
            anchor=parent_anchor(runtime['parent_checkpoint'],runtime['parent_receipt_sha256'])
            resident[0]=F32NativeGSPOBackend(Path(runtime['parent_checkpoint'])/'adapter.safetensors',{**entry.driver.PROFILE,'group_size':4,'max_tokens':256},model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=before,cache=cache)
            backend=resident[0]
            import inspect
            actual_optimizer_source=Path(inspect.getsourcefile(type(backend.optimizer))).resolve()
            require(actual_optimizer_source==Path(spec['optimizer_source']).resolve() and entry.digest(actual_optimizer_source)==scope['source_sha256'][str(Path(spec['optimizer_source']))],'Loaded Adam implementation differs from frozen preview formula source')
            adam,rng,_=load_optimizer_checkpoint(Path(runtime['parent_checkpoint'])/'optimizer',expected_bindings={'run':anchor['original_binding'],'adapter_sha256':anchor['adapter_sha256'],'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays)
            backend.restore(adam,rng);require(int(backend.optimizer_state()['step'].item())==60,'Restored exact Adam60 required');adam=rng=None
            return run_resident(backend,scope,out,counter,before)
        result=entry.run_with_release(run,resident,out)
        require(prepare(spec)==scope,'Source/input binding changed during readonly execution')
        result.update({'scope_sha256':canonical(scope),'source_sha256':scope['source_sha256'],'wall_seconds':time.monotonic()-started})
        entry.driver.write_json(out/'complete_PRIVATE.json',result)


def compare_gradient_vectors(chunks):
    """F64 streaming CPU innerproducts;zero norms have no defined cosine."""
    import numpy as np
    aa=bb=ab=dd=0.;count=0
    for name,left,right in chunks:
        a=np.asarray(left,dtype=np.float64).ravel();b=np.asarray(right,dtype=np.float64).ravel()
        require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Gradient comparison inventory/nonfinite failure')
        d=a-b;aa+=float(a@a);bb+=float(b@b);ab+=float(a@b);dd+=float(d@d);count+=1
        require(all(math.isfinite(n) for n in (aa,bb,ab,dd)),'Gradient comparison scalar overflow/nonfinite')
    require(count>0,'No saved-gradient chunks')
    cosine=(ab/math.sqrt(aa))/math.sqrt(bb) if aa>0 and bb>0 else 'UNDEFINED'
    relative=math.sqrt(dd)/math.sqrt(aa) if aa>0 else 'UNDEFINED'
    require(all(n=='UNDEFINED' or math.isfinite(n) for n in (cosine,relative)),'Gradient comparison derived scalar nonfinite')
    return {'tensors':count,'left_l2':math.sqrt(aa),'right_l2':math.sqrt(bb),'difference_l2':math.sqrt(dd),
        'cosine':cosine,
        'relative_difference_to_left':relative,
        'interpretation':'Gradient similarity/difference only;nonzero norm is not proof of image grounding.'}


def manifest_moment_inventory(layout):
    require(layout.get('kind')=='dict','Optimizer metadata root layout differs')
    root=dict(layout['items']);require('optimizer' in root,'Optimizer layout missing')
    out={}
    def visit(node,path=''):
        kind=node['kind']
        if kind=='dict':
            items=dict(node['items'])
            if 'm' in items or 'v' in items:
                require('m' in items and 'v' in items,'Incomplete Adam moment leaf')
                m,v=items['m'],items['v'];require(m['kind']==v['kind']=='tensor' and m['shape']==v['shape'] and m['dtype']==v['dtype'],'Adam moment metadata mismatch')
                out[path]={'shape':m['shape'],'dtype':m['dtype']}
            else:
                for k,v in node['items']:visit(v,(path+'.'+k).lstrip('.'))
        elif kind in ['list','tuple']:
            for i,v in enumerate(node['items']):visit(v,(path+'.'+str(i)).lstrip('.'))
    visit(root['optimizer']);return out
