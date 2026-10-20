"""Dormant native mixture candidate with current-Adam pre-step protection.

No native import/model construction here. Original GSPO gate/loss, single Adam
and atomic transaction driver are preserved. No launch is authorized. Fresh
guard VJPs consume the same hard compute counter and can stop a future stage
before an update. Norm envelope is NEW, not an old clipping algorithm.
"""
import gc
import hashlib
import math
import numpy as np
from mri_rl import paired_classification_mix_v3 as objective
from mri_rl.class_margin_aux_v2 import ClassMarginAuxBackend,combine_gradients,branch_mode
from mri_rl.class_margin_aux_v2_driver import run as run_micro_driver
from mri_rl.paired_class_margin_probe_v2 import serial_margin_vjp,readonly_adam_leaf,_moments
from mri_rl.paired_z045_f32_qualification_v1 import _state
from mri_rl.continuous_rl import require

STATUS='IMPLEMENTED_CPU_NATIVE_NOT_RUN_NOT_AUTHORIZED'
DELTA_CAP=.024884930546339723
AUX_GRADIENT_CAP=.34244086341414565


def target_scope(current_pair,references,expected_images):
    """Sequence slots protect their actual pair; Dallas slots use fixed F/T.

    Scope follows the computed VJPs explicitly. Never relabel current pair
    gradients as the old reference patient's gradients.
    """
    result={}
    for name,label,index in [('F_REAL','mF',0),('T_REAL','mT',1),('DALLAS_REAL','mDallas',None)]:
        current=current_pair is not None and index is not None
        row=current_pair[index] if current else references[name]['row_PRIVATE']
        result[label]={'mode':'CURRENT_AUX_PAIR' if current else 'FIXED_REFERENCE_CASE',
            'record_id_PRIVATE':row['id'],'image_sha256':expected_images[row['id']] if current else references[name]['image_sha256'],
            'question_sha256':hashlib.sha256(row['extra_info']['question'].encode()).hexdigest()}
    return result


def check_direction_report(total,delta_norm,*,guard=1e-6,delta_cap=DELTA_CAP):
    require(all(math.isfinite(total[k]) and total[k]>=guard for k in ('mT','mF','mDallas')),
            'CURRENT_JOINT_ADAM_CLASS_PROTECTION_FAILED: stop, do not alter gradients/advantages')
    require(math.isfinite(delta_norm) and delta_norm<=delta_cap*(1+1e-6),'NEW_PARAMETER_DELTA_NORM_ENVELOPE_EXCEEDED')
    return True


def preview_tree(gradient,targets,moments,profile,*,ops):
    """Pure native/CPU array Adam arithmetic; never updates optimizer or parameters."""
    require(set(targets)=={'mF','mT','mDallas'},'Need three exact protected target gradients')
    keys=set(gradient)
    require(set(moments)==keys and all(set(t)==keys for t in targets.values()),'Joint direction inventory differs')
    kinds=('Adam_total','Adam_g0_baseline','Adam_increment');dots={kind:{name:0. for name in targets} for kind in kinds};norm=0.
    for key in sorted(keys):
        g=gradient[key];m,v=moments[key]
        p=readonly_adam_leaf(g,m,v,ops=ops,**profile)
        for kind,field in zip(kinds,('delta','baseline_delta','incremental_delta')):
            vec=np.asarray(p[field],np.float64).ravel()
            for name,t in targets.items(): dots[kind][name]+=float(vec@np.asarray(t[key],np.float64).ravel())
            if kind=='Adam_total':norm+=float(vec@vec)
    require(math.isfinite(norm) and all(math.isfinite(v) for row in dots.values() for v in row.values()),'Nonfinite current-Adam direction')
    return {'directions':dots,'Adam_total_delta_l2':math.sqrt(norm),'finite_step_evaluated':False}


class MixtureClassMarginBackend(ClassMarginAuxBackend):
    def configure_mixture_guards(self,reference_contexts,token_plans,*,qualification_receipt):
        require(qualification_receipt.get('status')=='FRESH_V3_MULTI_TRAIN_READONLY_REVIEWED'
                and qualification_receipt.get('weight')==objective.WEIGHT
                and qualification_receipt.get('native_joint_updates_measured') is False,
                'Fresh V3 multi-TRAIN VJP review is absent; it must not claim joint training measured')
        require(set(reference_contexts)=={'F_REAL','T_REAL','DALLAS_REAL'},'Fixed TRAIN guard contexts required')
        require(all(c['row_PRIVATE']['extra_info']['internal_split']=='train' and not c['no_image'] for c in reference_contexts.values()),'TRAIN-only real guard inputs required')
        self.reference_contexts=reference_contexts
        self.guard_token_plans=token_plans

    def auxiliary_gradient(self,row):
        scores=[];gradients=[];evidence=[]
        for index,image_row in enumerate(self.companions[row['id']]):
            require(image_row['reward_model']['ground_truth']==('FLAIR','contrast-enhanced T1')[index],'Auxiliary F/T order differs')
            inputs=self._companion_inputs(image_row)
            r=serial_margin_vjp(self,inputs,self.aux_token_plan,index,before=self.before_unit,
                charge_forward=self.compute.auxiliary_forward,charge_backward=self.compute.auxiliary_backward)
            scores.append(r['mean_scores']);gradients.append(r['gradient']);evidence.append({k:v for k,v in r.items() if k!='gradient'})
            del inputs,r;gc.collect();self.mx.clear_cache()
        derivatives=objective.score_derivatives(scores)
        combined=combine_gradients(None,[(g,objective.ALPHA*derivatives[i][i]) for i,g in enumerate(gradients)],self.tree_map)
        self.mx.eval(combined);norm=self._checked_gradient(combined)
        require(norm<=AUX_GRADIENT_CAP*(1+1e-6),'NEW_AUX_GRADIENT_NORM_ENVELOPE_EXCEEDED')
        self._fresh_aux_targets=gradients
        self._fresh_aux_rows=self.companions[row['id']]
        return combined,{'scores':scores,'auxiliary_loss':objective.loss(scores),'weight':objective.WEIGHT,'alpha':objective.ALPHA,
                         'mean_margin_coefficients':[objective.ALPHA*derivatives[i][i] for i in range(2)],'margin_evidence':evidence}

    def require_current_targets(self,current,snapshot):
        require(snapshot.get('step')==current['step'] and set(snapshot.get('targets',{}))=={'mF','mT','mDallas'},
                'Need fresh current model margin VJPs; saved Adam60 targets are stale after an update')
        return snapshot['targets']

    def fresh_targets(self):
        from mri_rl import paired_signal_candidate_probe_v1 as old
        from mri_rl import early_native_conditioning_probe_v1 as utility
        targets={}
        if self._fresh_aux_targets is not None:
            targets.update(zip(('mF','mT'),self._fresh_aux_targets))
        for name,label in [('F_REAL','mF'),('T_REAL','mT'),('DALLAS_REAL','mDallas')]:
            if label in targets:continue
            context=self.reference_contexts[name];old.verify_context(context)
            _,inputs,_=old.prepare_inputs(self,context,utility)
            plan=self.guard_token_plans[context['row_PRIVATE']['ability']]
            truth=plan['label_order'].index(context['source_truth'])
            r=serial_margin_vjp(self,inputs,plan,truth,before=self.before_unit,
                charge_forward=self.compute.forward,charge_backward=self.compute.backward)
            targets[label]=r['gradient'];del inputs,r;gc.collect();self.mx.clear_cache()
        return {'step':int(self.optimizer_state()['step'].item()),'targets':targets,
                'target_scope_PRIVATE':target_scope(self._fresh_aux_rows,self.reference_contexts,self.expected_image_sha256)}

    def transition(self,row,raw,advantages,active,*,baseline=None):
        """Same original gate; unprotected proposed joint step stops before Adam."""
        from mri_rl import early_native_conditioning_probe_v1 as utility
        require(hasattr(self,'reference_contexts') and hasattr(self,'guard_token_plans'),'V3 guard qualification/configuration missing')
        self._fresh_aux_targets=None
        self._fresh_aux_rows=None
        mode=branch_mode(active,row['id'] in self.companions);state_after_rollout=_state(self,utility)
        evidence={'mode':mode,'accepted_GSPO':active,'candidate':'v3','weight':objective.WEIGHT};gspo=aux=None
        old=[];alignments=[]
        if mode!='NO_UPDATE':old,alignments=self.detached_ownold(row,raw);evidence['rollout_full_selected_logp_alignment']=alignments
        if active:gspo,evidence['GSPO']=self.gspo_gradient(row,raw,advantages,old,alignments)
        if row['id'] in self.companions:aux,evidence['auxiliary']=self.auxiliary_gradient(row)
        gradient=combine_gradients(gspo,[] if aux is None else [(aux,1.)],self.tree_map)
        if gradient is None:
            self.active=None;self._fresh_aux_targets=None
            return {**evidence,'updated':False,'gradient_l2':0.}
        norm=self._checked_gradient(gradient)
        if not active and self._checked_gradient(aux)==0:
            self.active=None;self._fresh_aux_targets=None
            return {**evidence,'mode':'NO_UPDATE','updated':False,'gradient_l2':0.}
        require(norm>0 and math.isfinite(norm),'TOTAL_GRADIENT_CANCELLATION or nonfinite')
        require(not active or self._checked_gradient(gspo)>0,'Original active GSPO has zero gradient')
        snapshot=self.fresh_targets();step=int(self.optimizer_state()['step'].item())
        targets=self.require_current_targets({'step':step},snapshot)
        opt=self.optimizer
        profile={'learning_rate':float(opt.learning_rate.item()),'betas':tuple(opt.betas),'eps':opt.eps,'bias_correction':opt.bias_correction,'step':step}
        report=preview_tree(dict(self.tree_flatten(gradient)),{k:dict(self.tree_flatten(v)) for k,v in targets.items()},_moments(self.optimizer_state()),profile,ops=self.mx)
        require(snapshot.get('target_scope_PRIVATE')==target_scope(self._fresh_aux_rows,self.reference_contexts,self.expected_image_sha256),
                'Current protected target identities changed')
        report['target_scope_PRIVATE']=snapshot['target_scope_PRIVATE']
        check_direction_report(report['directions']['Adam_total'],report['Adam_total_delta_l2'])
        require(_state(self,utility)==state_after_rollout,'Guard scoring/backward changed Adam/RNG')
        if baseline is not None:evidence['pre_step_all_parameters']=self.transaction_validate(baseline,False)
        evidence['current_joint_Adam_direction']=report
        evidence['objective_value']=sum(evidence.get('GSPO',{}).get('loss_terms',[]))+objective.ALPHA*evidence.get('auxiliary',{}).get('auxiliary_loss',0.)
        self.before_unit('v3_one_combined_optimizer_update')
        self.optimizer.update(self.model,gradient);self.mx.eval(self.model.trainable_parameters(),self.optimizer.state)
        self.active=None;self._fresh_aux_targets=None;self.mx.clear_cache()
        return {**evidence,'updated':True,'gradient_l2':norm,'optimizer_calls':1}
