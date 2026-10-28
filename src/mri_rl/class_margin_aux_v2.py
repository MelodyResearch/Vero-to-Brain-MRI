"""Native F32 GSPO + supervised class-margin v2 integration, CPU implemented.

No MLX import or launch on import. Existing loader, rollout, GSPO loss AST,
reward, admission, LoRA and persistent Adam are reused. Numerical training
qualification and fresh human/exact independent launch review remain required.
"""
from __future__ import annotations
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from mri_rl.continuous_rl import require
from mri_rl import paired_classification_signal_v2 as objective
from mri_rl import paired_z045_continuation_entry_v1 as entry
from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend,generation_alignment
from mri_rl.paired_z045_f32_qualification_v1 import frozen_loss_node,_state

STATUS='IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED'
POLICY='F32_GSPO_PLUS_SUPERVISED_CLASS_MARGIN_V2'


def branch_mode(active,auxiliary):return objective.update_kind(active,auxiliary)


def require_update_budget(updates,maximum,mode):
    require(type(updates) is int and type(maximum) is int and 0<=updates<=maximum<=2,'Invalid TOTAL optimizer step count')
    require(mode=='NO_UPDATE' or updates<maximum,'TOTAL optimizer update budget exhausted')
    return True


def group_reservation(prompt,auxiliary_positions):
    require(prompt in (213,405) and type(auxiliary_positions) is int and auxiliary_positions>=0,'Fixed prompt/auxiliary token reservation differs')
    return 4*(prompt+256)+auxiliary_positions


def combine_gradients(gspo,auxiliary,tree_map):
    result=gspo
    for gradient,coefficient in auxiliary:
        require(math.isfinite(float(coefficient)),'Nonfinite auxiliary coefficient')
        scaled=tree_map(lambda value:value*coefficient,gradient)
        result=scaled if result is None else tree_map(lambda left,right:left+right,result,scaled)
    return result


def authorize_launch(scope,certificate,human,review):
    require(scope.get('scope_kind')=='FUTURE_EXACT_CLASS_MARGIN_AUX_V2_TRAIN','Preparation is not native authorization')
    require(scope.get('budget',{}).get('maximum_attempted_groups')==4 and scope['budget'].get('maximum_optimizer_updates')==2,'Exact4groups/2TOTALupdates required')
    require(certificate.get('schema')=='independent-class-margin-v2-compatibility-admission-v1' and
            certificate.get('status')=='ADMITTED_WITH_EXPLICIT_NATIVE_LIMITATIONS' and certificate.get('policy')==POLICY and
            certificate.get('training_source_sha256')==scope.get('source_sha256') and
            certificate.get('training_budget')==scope.get('budget'),'Independent exact source/budget compatibility certificate required')
    require(certificate.get('native_joint_update_measured') is False and certificate.get('native_onpolicy_backward_measured') is False,
            'Readonly margin proof cannot claim native joint/onpolicy training qualification')
    require(certificate.get('independent_reviewer_id') and certificate.get('independent_reviewer_kind') in ('isolated_codex','claude_code'), 'Independent admission reviewer required')
    binding=entry.scope_digest(scope)
    require(human.get('authorized') is True and human.get('action')=='finite_class_margin_aux_v2_train' and human.get('scope_sha256')==binding and human.get('human_instruction_sha256') and
            human.get('accepts_unmeasured_native_joint_onpolicy') is True,'New exact human training authorization accepting explicit native limitations required')
    require(review.get('scope_sha256')==binding and review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('reviewer_id') and review.get('reviewer_kind') in ('isolated_codex','claude_code') and
            review.get('accepts_unmeasured_native_joint_onpolicy') is True,'New exact independent training review accepting explicit limitations required')
    return True


class ComputeBudget:
    """Actual scorer API calls; checkpoint internal recomputation is not counted."""
    def __init__(self,limits=None,used=None):
        self.limits=limits or {'full_TF_forwards':40,'backward_calls':20,'teacherforced_positions':19860,'auxiliary_positions':1812,'auxiliary_TF_forwards':8,'auxiliary_backward_calls':4}
        if used is None:used={key:0 for key in self.limits}
        require(isinstance(used,dict),'Restored compute counters must be a mapping')
        legacy={'full_TF_forwards':0,'backward_calls':0,'teacherforced_positions':0}
        if set(used)==set(legacy) and all(type(value) is int and value==0 for value in used.values()):
            used={key:0 for key in self.limits}
        require(set(used)==set(self.limits) and all(type(value) is int and 0<=value<=self.limits[key] for key,value in used.items()),
                'Restored compute counter inventory/count outside exact finite limits')
        self.used=dict(used)
    def _charge(self,amounts):
        require(all(type(value) is int and value>=0 for value in amounts.values()) and
                all(self.used[key]+value<=self.limits[key] for key,value in amounts.items()),'Native compute budget exhausted before operation')
        for key,value in amounts.items():self.used[key]+=value
        if hasattr(self,'persist'):self.persist()
    def forward(self,positions):self._charge({'full_TF_forwards':1,'teacherforced_positions':positions})
    def backward(self):self._charge({'backward_calls':1})
    def auxiliary_forward(self,positions):self._charge({'full_TF_forwards':1,'teacherforced_positions':positions,'auxiliary_positions':positions,'auxiliary_TF_forwards':1})
    def auxiliary_backward(self):self._charge({'backward_calls':1,'auxiliary_backward_calls':1})


class ClassMarginAuxBackend(F32NativeGSPOBackend):
    """Single optimizer; serial branch gradients combine before the sole Adam call."""
    def configure_auxiliary(self,companions,plan,expected_images,compute):
        self.companions=companions;self.aux_token_plan=plan;self.expected_image_sha256=expected_images;self.compute=compute

    def rollout(self,row,index,workdir):
        raw=super().rollout(row,index,workdir)
        returned=sum(len(response['token_ids']) for response in raw['responses'])
        calls=sum(len(response['generation_dtype_cache_evidence_PRIVATE']) for response in raw['responses'])
        require(returned<=1024 and calls<=returned+8,'Actual F32 stream LM API count exceeds K4*(tokens+2) bound')
        raw['runtime_API_counts']={'returned_response_tokens':returned,'generation_LM_calls':calls,
            'generation_vision_embedding_calls':4,'unreturned_AR_lookahead_draws':4,
            'counting':'Completed stream API endpoints, not checkpoint/kernel/hardware counts; installed one unreturned lookahead draw perresponse.'}
        self.active['raw_sha256']=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        return raw

    def _checked_gradient(self,gradient):
        mx=self.mx;flat=dict(self.tree_flatten(gradient));signature=self.trainable_signature()
        require(set(flat)==set(signature) and all(bool(mx.all(mx.isfinite(value)).item()) for value in flat.values()),'Complete finite trainable gradient required')
        return math.sqrt(sum(float(mx.sum(value.astype(mx.float32)**2).item()) for value in flat.values()))

    def _charged_logps(self,tokens):
        self.compute.forward(int(self.active['inputs']['input_ids'].shape[-1])+len(tokens)-1)
        return self._logps(tokens)

    def detached_ownold(self,row,raw):
        """Every actual update branch first qualifies its actual rollout selected logps."""
        mx=self.mx;self.model.eval()
        require(self.active is not None and self.active['record_id']==row['id'] and
                self.active['raw_sha256']==hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),'Update differs from exact current rollout')
        require(len(raw['responses'])==self.config['group_size']==4,'ExactK4 response inventory required')
        old=[];alignments=[]
        for response in raw['responses']:
            self.before_unit('v2_detached_ownold')
            values=mx.stop_gradient(self._charged_logps(response['token_ids']));mx.eval(values)
            alignments.append(generation_alignment(response['raw_selected_logps'],values.tolist()));old.append(values)
        return old,alignments

    def gspo_gradient(self,row,raw,advantages,old,alignments):
        """Original native loss compiled directly from unchanged source AST."""
        mx=self.mx
        require(len(advantages)==len(old)==4,'Exact original advantages required')
        result=None;losses=[];errors=[]
        for response,previous,advantage in zip(raw['responses'],old,advantages):
            fresh=[]
            def score(tokens):
                value=self._charged_logps(tokens);fresh.append(mx.stop_gradient(value));return value
            namespace={'mx':mx,'self':SimpleNamespace(_logps=score)};node=frozen_loss_node()
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged-native-GSPO-loss-AST>','exec'),namespace)
            self.before_unit('v2_GSPO_response_gradient');self.compute.backward()
            loss,gradient=self.nn.value_and_grad(self.model,namespace['loss_fn'])(self.model,response['token_ids'],previous,advantage)
            mx.eval(loss,gradient,fresh);require(len(fresh)==1,'One fresh scorer per original GSPO backward required')
            error=float(mx.max(mx.abs(fresh[0]-previous)).item());require(math.isfinite(error) and error<=1e-5,'Own-old/current1e-5 gate failed inside gradient')
            value=float(loss.item());require(math.isfinite(value),'Nonfinite unchanged GSPO loss');self._checked_gradient(gradient)
            scaled=self.tree_map(lambda v:v/4,gradient);result=scaled if result is None else self.tree_map(lambda x,y:x+y,result,scaled)
            mx.eval(result);errors.append(error);losses.append(value/4)
            del gradient,scaled,fresh,namespace;gc.collect();mx.clear_cache()
        return result,{'loss_terms':losses,'old_current_logp_max_abs_errors':errors,'rollout_full_alignment':alignments,
                       'native_loss_AST_sha256':hashlib.sha256(ast.dump(frozen_loss_node()).encode()).hexdigest()}

    def _companion_inputs(self,row):
        from mri_rl.paired_z045_continuation_entry_v1 import verify_images
        self.before_unit('v2_auxiliary_image_prepare');verify_images([row],self.expected_image_sha256)
        prompt=self.template(self.processor,self.model.config,[{'role':'system','content':self.contract.system_instruction},
            {'role':'user','content':row['extra_info']['question']}],num_images=1,add_generation_prompt=True,enable_thinking=self.contract.enable_thinking)
        require(prompt.count('<|image_pad|>')==1 and prompt.endswith(self.contract.assistant_prefix),'Companion native prompt differs')
        result=self.prepare_inputs(self.processor,images=row['images'],prompts=prompt,image_token_index=self.model.config.image_token_index,
            add_special_tokens=self.special_tokens(self.model.config.model_type,self.processor))
        require(int(result['input_ids'].shape[-1])==213 and bool(self.mx.all(result['attention_mask']==1).item()),'Companion fixed prompt213 differs')
        verify_images([row],self.expected_image_sha256);return result

    def auxiliary_gradient(self,row):
        from mri_rl.paired_class_margin_probe_v2 import serial_margin_vjp
        pair=self.companions[row['id']];scores=[];branches=[];evidence=[]
        for index,image_row in enumerate(pair):
            require(image_row['reward_model']['ground_truth']==('FLAIR','contrast-enhanced T1')[index],'F/T companion order changed')
            inputs=self._companion_inputs(image_row)
            result=serial_margin_vjp(self,inputs,self.aux_token_plan,index,before=self.before_unit,
                charge_forward=self.compute.auxiliary_forward,charge_backward=self.compute.auxiliary_backward)
            scores.append(result['mean_scores']);branches.append(result['gradient']);evidence.append({k:v for k,v in result.items() if k!='gradient'})
            del inputs,result;gc.collect();self.mx.clear_cache()
        coefficients=objective.score_derivatives(scores)
        scaled=[(gradient,objective.ALPHA*coefficients[index][index]) for index,gradient in enumerate(branches)]
        total=combine_gradients(None,scaled,self.tree_map);self.mx.eval(total);self._checked_gradient(total)
        return total,{'scores':scores,'auxiliary_loss':objective.classification_loss(scores),'alpha':objective.ALPHA,'margin_evidence':evidence,
                      'mean_margin_coefficients':[objective.ALPHA*coefficients[i][i] for i in range(2)]}

    def transition(self,row,raw,advantages,active,*,baseline=None):
        """Called even if original GSPO gate is inactive; advantages never replaced."""
        from mri_rl import early_native_conditioning_probe_v1 as utility
        mode=branch_mode(active,row['id'] in self.companions);state_after_rollout=_state(self,utility)
        gspo=None;aux=None;evidence={'mode':mode,'accepted_GSPO':active}
        old=None;alignments=[]
        if mode!='NO_UPDATE':old,alignments=self.detached_ownold(row,raw);evidence['rollout_full_selected_logp_alignment']=alignments
        if active:gspo,values=self.gspo_gradient(row,raw,advantages,old,alignments);evidence['GSPO']=values
        if row['id'] in self.companions:aux,values=self.auxiliary_gradient(row);evidence['auxiliary']=values
        evidence['objective_value']=sum(evidence.get('GSPO',{}).get('loss_terms',[]))+objective.ALPHA*evidence.get('auxiliary',{}).get('auxiliary_loss',0.)
        require(_state(self,utility)==state_after_rollout,'Deterministic scoring/backward changed Adam/RNG')
        gradient=combine_gradients(gspo,[] if aux is None else [(aux,1.)],self.tree_map)
        if gradient is None:self.active=None;return {**evidence,'updated':False,'gradient_l2':0.}
        norm=self._checked_gradient(gradient);require(math.isfinite(norm),'Nonfinite joint gradient')
        evidence['GSPO_gradient_l2']=self._checked_gradient(gspo) if gspo is not None else 0.
        evidence['auxiliary_gradient_l2']=self._checked_gradient(aux) if aux is not None else 0.
        if not active and evidence['auxiliary_gradient_l2']==0:
            self.active=None;return {**evidence,'mode':'NO_UPDATE','updated':False,'gradient_l2':norm,'reason':'ZERO_AUXILIARY_GRADIENT_NO_MOMENTUM_STEP'}
        require(not active or evidence['GSPO_gradient_l2']>0,'Original active GSPO has zero gradient')
        evidence['total_gradient_cancellation']=norm==0
        require(norm>0,'TOTAL_GRADIENT_CANCELLATION: rollback current group, no momentum-only step')
        if baseline is not None:evidence['pre_step_all_parameters']=self.transaction_validate(baseline,False)
        self.before_unit('v2_one_combined_optimizer_update')
        self.optimizer.update(self.model,gradient);self.mx.eval(self.model.trainable_parameters(),self.optimizer.state)
        self.active=None;self.mx.clear_cache()
        return {**evidence,'updated':True,'gradient_l2':norm,'optimizer_calls':1}

    def activate_saved_rollout(self,row,raw):
        """Rebuild deterministic exact inputs, never generate a replacement response."""
        from mri_rl.paired_z045_continuation_entry_v1 import verify_images
        from mri_rl.paired_z045_f32_backend_v1 import require_prompt_cost
        from mri_rl import early_native_conditioning_probe_v1 as utility
        state_before=_state(self,utility)
        verify_images([row],self.expected_image_sha256);self.model.eval()
        require(raw['record_id']==row['id'] and raw['image_sha256']==self.expected_image_sha256[row['id']],'Pending rollout identity changed')
        prompt=self.template(self.processor,self.model.config,[{'role':'system','content':self.contract.system_instruction},
            {'role':'user','content':row['extra_info']['question']}],num_images=1,add_generation_prompt=True,enable_thinking=self.contract.enable_thinking)
        require(prompt==raw['rendered_prompt'],'Pending prompt changed')
        inputs=self.prepare_inputs(self.processor,images=row['images'],prompts=prompt,image_token_index=self.model.config.image_token_index,
            add_special_tokens=self.special_tokens(self.model.config.model_type,self.processor))
        require(inputs['input_ids'].tolist()==raw['prompt_token_ids'],'Pending prompt IDs changed')
        length=int(inputs['input_ids'].shape[-1]);require_prompt_cost(row['ability'],length)
        self.active={'record_id':row['id'],'inputs':inputs,'prompt_length':length,
                     'raw_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()}
        require(_state(self,utility)==state_before,'Pending input rebuild consumed RNG or changed Adam')

    def transaction_snapshot(self):
        """MLX arrays immutable: retain base references, clone trainable/state trees only."""
        from mri_rl import early_native_conditioning_probe_v1 as utility
        parameters=self.model.parameters()
        # Tree containers are detached; immutable arrays/base payload are not copied.
        references=self.tree_map(lambda value:value,parameters)
        snapshot={'parameters':references,'optimizer':self.tree_map(lambda v:self.mx.array(v) if hasattr(v,'dtype') else v,self.optimizer_state()),
                  'rng':self.rng_state(),'parameter_fingerprints':utility._fingerprints(self,dict(self.tree_flatten(parameters))),'state':_state(self,utility)}
        self.mx.eval(snapshot['optimizer'],snapshot['rng']);return snapshot

    def transaction_validate(self,snapshot,updated):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        after=utility._fingerprints(self,dict(self.tree_flatten(self.model.parameters())))
        before=snapshot['parameter_fingerprints'];trainable=set(self.trainable_signature())
        require(len(before)==len(after)==1942 and len(trainable)==504,'Full1942/504 inventory changed')
        require(set(before)==set(after) and all(before[key]==after[key] for key in before if key not in trainable),'Frozen base parameter mutation')
        if not updated:require(before==after,'NO_UPDATE changed model parameters')
        return {'before_PRIVATE':before,'after_PRIVATE':after,'base1438_unchanged':True,'parameters1942_checked':True}

    def transaction_restore(self,snapshot):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        self.active=None;self.model.update(snapshot['parameters']);self.mx.eval(self.model.parameters())
        self.restore(snapshot['optimizer'],snapshot['rng'])
        require(utility._fingerprints(self,dict(self.tree_flatten(self.model.parameters())))==snapshot['parameter_fingerprints'] and
                _state(self,utility)==snapshot['state'],'Rollback failed full model/Adam/RNG integrity')
        self.mx.clear_cache()
