"""CPU paired-prompt GSPO mechanism; PROPOSED VERO-STYLE ADAPTATION.

Only group membership changes. No CE/margin auxiliary, reward recomputation,
model import, generation, optimizer or native execution path. Typed identities
preserve per-response conditioning; callers must authenticate real receipts.
Synthetic fixtures and effective groups are never MRI accuracy evidence.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
from mri_rl.mlx_gspo_smoke import advantages_from_rewards,task_group_is_informative as historical18_strict_gate
from mri_rl.uniform_balance_continuation_gate_v1 import accuracy_group_is_informative as inherited_Adam60_gate

STATUS='PROPOSED_VERO_STYLE_ADAPTATION_CPU_TESTED_NOT_TRAINED'
CLIP_LOW=.0003
CLIP_HIGH=.0004
LOG_RATIO_MAX=10.


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def _sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _finite(values):return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class PromptContext:
    context_id:str
    pair_id:str
    modality:str
    prompt_sha256:str
    image_sha256:str
    checkpoint_sha256:str
    policy_sha256:str
    reward_policy_sha256:str
    def __post_init__(self):
        require(self.context_id and self.pair_id and self.modality in ('F','T'),'Explicit paired TRAIN context identity required')
        require(all(_sha(value) for value in (self.prompt_sha256,self.image_sha256,self.checkpoint_sha256,self.policy_sha256,self.reward_policy_sha256)),'Exact prompt/image/checkpoint/policy SHA required')
    @property
    def sha256(self):return identity(asdict(self))


@dataclass(frozen=True)
class Grade:
    accuracy:float
    format:float
    total:float
    overlong:float=0.
    raw_accuracy:float|None=None
    def __post_init__(self):
        require(_finite((self.accuracy,self.format,self.total,self.overlong)) and 0<=self.accuracy<=1 and self.format in (0,.5,1),'Finite unchanged calibrated accuracy and historical0/.5/1 format reward receipt required')
        raw=self.accuracy if self.raw_accuracy is None and self.accuracy in (0,1) else self.raw_accuracy
        require(raw in (0,1),'Raw binary accuracy required for calibrated receipt')
        object.__setattr__(self,'raw_accuracy',raw)


@dataclass(frozen=True)
class Response:
    response_id:str
    context:PromptContext
    token_ids:tuple[int,...]
    previous_logps:tuple[float,...]
    grade:Grade
    reward_receipt_sha256:str
    def __post_init__(self):
        require(type(self.token_ids) is tuple and type(self.previous_logps) is tuple and 0<len(self.token_ids)==len(self.previous_logps)<=256 and
                all(type(value) is int and value>=0 for value in self.token_ids) and _finite(self.previous_logps) and all(value<=0 for value in self.previous_logps),'Bounded own-response tokens/logps required')
        require(_sha(self.reward_receipt_sha256),'Original response-specific reward receipt binding required')
    @property
    def token_sha256(self):return identity(self.token_ids)


@dataclass(frozen=True)
class WeightedResponse:
    response:Response
    group_index:int
    advantage:float
    admitted:bool
    weight:float


@dataclass(frozen=True)
class Plan:
    grouping:str
    items:tuple[WeightedResponse,...]
    effective_groups:int
    reward_variable_groups:int
    response_count:int=8


@dataclass(frozen=True)
class ScoreRequest:
    """Route this exact input and response to the scorer; never group prompt alone."""
    response_id:str
    context:PromptContext
    token_ids:tuple[int,...]
    previous_logps:tuple[float,...]
    context_sha256:str
    token_sha256:str


@dataclass(frozen=True)
class ConditionedLogps:
    response_id:str
    context_sha256:str
    token_sha256:str
    logps:tuple[float,...]


def group_responses(block,grouping):
    require(grouping in ('same_prompt','paired_prompt') and len(block)==8,'Exact matched eight-response block and fixed grouping required')
    lookup={response.response_id:response for response in block}
    require(set(lookup)=={f'{task}{i}' for task in ('F','T') for i in range(4)},'Unique F0..F3/T0..T3 response inventory required')
    contexts={task:lookup[task+'0'].context for task in ('F','T')}
    require(all(lookup[f'{task}{i}'].context==contexts[task] and contexts[task].modality==task for task in ('F','T') for i in range(4)),'Each response must retain its original exact input context')
    f,t=contexts['F'],contexts['T']
    require(f.pair_id==t.pair_id and f.prompt_sha256==t.prompt_sha256 and f.image_sha256!=t.image_sha256,'Same paired person/level and generic prompt, distinct original images required')
    require(f.checkpoint_sha256==t.checkpoint_sha256 and f.policy_sha256==t.policy_sha256 and f.reward_policy_sha256==t.reward_policy_sha256,'Fixed-checkpoint/numerical-policy matched block required; no historical splicing')
    names=([['F0','F1','F2','F3'],['T0','T1','T2','T3']] if grouping=='same_prompt' else
           [['F0','F1','T0','T1'],['F2','F3','T2','T3']])
    return tuple(tuple(lookup[name] for name in group) for group in names)


def group_diagnostics(block,grouping):
    result=[]
    for group in group_responses(block,grouping):
        contexts={}
        for response in group:
            values=contexts.setdefault(response.context.modality,{'all_accuracy':set(),'format_valid_accuracy':set(),'raw_accuracy':set()})
            values['all_accuracy'].add(response.grade.accuracy);values['raw_accuracy'].add(response.grade.raw_accuracy)
            if response.grade.format==1:values['format_valid_accuracy'].add(response.grade.accuracy)
        result.append({'inherited_Adam60_gate_admitted':inherited_Adam60_gate([asdict(r.grade) for r in group]),
            'total_reward_varies':len({r.grade.total for r in group})>1,
            'accuracy_by_condition':{key:{'calibrated_accuracy_values':sorted(value['all_accuracy']),'format_valid_calibrated_accuracy_values':sorted(value['format_valid_accuracy']),
                                          'raw_accuracy_values':sorted(value['raw_accuracy']),'within_condition_raw_correctness_varies':len(value['raw_accuracy'])>1,
                                          'within_condition_calibrated_accuracy_varies':len(value['all_accuracy'])>1} for key,value in contexts.items()}})
    return result


def make_plan(block,grouping):
    items=[];effective=variable=0
    for number,group in enumerate(group_responses(block,grouping)):
        grades=[asdict(response.grade) for response in group]
        admitted=inherited_Adam60_gate(grades);effective+=int(admitted)
        rewards=[response.grade.total for response in group];variable+=int(max(rewards)-min(rewards)>1e-12)
        advantages=advantages_from_rewards(rewards)
        for response,advantage in zip(group,advantages):items.append(WeightedResponse(response,number,advantage,admitted,1/8 if admitted else 0.))
    return Plan(grouping,tuple(items),effective,variable)


def response_surrogate(current,previous,advantage):
    """Original sequence ratio/clip forward and its local token-logp derivative.

    Algebraically equivalent to the historical native stop-gradient construction
    away from clipping/cap ties. No claim of native autograd execution. Exact
    clipping boundary derivatives are deliberately undefined in this CPU API.
    """
    require(len(current)==len(previous)>0 and _finite([*current,*previous,advantage]),'Finite complete current/own-old response vectors required')
    delta=math.fsum(float(a)-float(b) for a,b in zip(current,previous))/len(current)
    ratio=math.exp(min(delta,LOG_RATIO_MAX));clipped=min(max(ratio,1-CLIP_LOW),1+CLIP_HIGH)
    loss=max(-advantage*ratio,-advantage*clipped)
    if advantage!=0:
        require(delta!=LOG_RATIO_MAX and ratio not in (1-CLIP_LOW,1+CLIP_HIGH),'Native clip/cap tie derivative requires explicit backend convention')
    active=delta<LOG_RATIO_MAX and ((advantage>0 and ratio<1+CLIP_HIGH) or (advantage<0 and ratio>1-CLIP_LOW))
    coefficient=-advantage*ratio/len(current) if active else 0.
    return {'loss':loss,'sequence_ratio':ratio,'d_loss_d_token_logps':[coefficient]*len(current),'clip_branch_active':active}


def score_plan(plan,scorer):
    """Scorer returns its conditioning receipt; weights apply exactly once at1/8.

    This is a CPU scalar/derivative interface, not an executable native trainer.
    It assumes genuine on-policy old logps and verifiable rewards were externally
    authenticated. Identity equality alone cannot prove that provenance.
    """
    require(isinstance(plan,Plan) and callable(scorer),'Typed plan and conditioned scorer required')
    output=[]
    for item in plan.items:
        response=item.response
        request=ScoreRequest(response.response_id,response.context,response.token_ids,response.previous_logps,response.context.sha256,response.token_sha256)
        result=scorer(request)
        require(isinstance(result,ConditionedLogps) and result.response_id==request.response_id and result.context_sha256==request.context_sha256 and
                result.token_sha256==request.token_sha256,'Response/prompt/image/token attribution mismatch')
        value=response_surrogate(result.logps,response.previous_logps,item.advantage)
        output.append({'response_id':response.response_id,'context_sha256':response.context.sha256,'group_index':item.group_index,'admitted':item.admitted,
                       'advantage':item.advantage,'weight':item.weight,'loss':item.weight*value['loss'],'sequence_ratio':value['sequence_ratio'],
                       'd_loss_d_token_logps':[item.weight*v for v in value['d_loss_d_token_logps']]})
    return {'loss':math.fsum(value['loss'] for value in output),'responses':output,'effective_groups':plan.effective_groups,'response_count':8}


def local_descent(plan,mean_logp_gradients):
    """At the own-old point r=1: −∇L=Σ admitted A_i ∇meanlogp_i/8.

    Gradients are explicit caller-supplied CPU vectors; never substitute a label
    Jacobian for the complete response mean-logp Jacobian. No optimizer step.
    """
    require(set(mean_logp_gradients)=={item.response.response_id for item in plan.items},'Exact per-response conditional Jacobian inventory required')
    vectors=list(mean_logp_gradients.values());dimension=len(vectors[0])
    require(dimension>0 and all(len(vector)==dimension and _finite(vector) for vector in vectors),'Finite common-dimensional Jacobians required')
    return [math.fsum(item.weight*item.advantage*mean_logp_gradients[item.response.response_id][j] for item in plan.items) for j in range(dimension)]


def synthetic_block(correct_F,correct_T,*,format_F=1,format_T=1,total_F=None,total_T=None,lengths_F=None,lengths_T=None):
    """Exactly labeled synthetic counterexamples, NEVER authenticated MRI output."""
    require(len(correct_F)==len(correct_T)==4,'Synthetic four-per-input outcomes required')
    rows=[]
    for task,correct,format,total,lengths in [('F',correct_F,format_F,total_F,lengths_F),('T',correct_T,format_T,total_T,lengths_T)]:
        context=PromptContext('SYNTHETIC_'+task,'SYNTHETIC_PAIR',task,identity('generic question'),identity('synthetic image'+task),identity('SYNTHETIC checkpoint'),identity('SYNTHETIC policy'),identity('SYNTHETIC shared reward calibration'))
        for i,value in enumerate(correct):
            n=(lengths or [1]*4)[i];grade=Grade(float(value),float(format),float(value if total is None else total))
            rows.append(Response(task+str(i),context,tuple([7]*n),tuple([-1.]*n),grade,identity({'SYNTHETIC':task+str(i),'grade':asdict(grade)})))
    return tuple(rows)


def enumerate_bernoulli(pF,pT,grouping,*,reward_mode='correctness'):
    """Exact256-state one-token on-policy example, no sampling/model/MLX.

    Each action y∈{0,1}; correctness reward y, dlogπ/dlogit=y−p.
    prompt_constant instead assigns F reward1/T reward0 independent of y.
    Return expected local loss-descent, not accuracy or an optimizer update.
    """
    require(0<pF<1 and 0<pT<1 and reward_mode in ('correctness','prompt_constant'),'Interior Bernoulli policies and declared reward mode required')
    expectation=[0.,0.];effective=0.;mass=0.
    for actions in itertools.product((0,1),repeat=8):
        probabilities=[pF]*4+[pT]*4
        probability=math.prod(p if y else 1-p for y,p in zip(actions,probabilities));mass+=probability
        rewards=actions if reward_mode=='correctness' else (1,1,1,1,0,0,0,0)
        plan=make_plan(synthetic_block(rewards[:4],rewards[4:]),grouping)
        gradients={task+str(i):([actions[i]-pF,0.] if task=='F' else [0.,actions[4+i]-pT]) for task in ('F','T') for i in range(4)}
        direction=local_descent(plan,gradients)
        for i in range(2):expectation[i]+=probability*direction[i]
        effective+=probability*plan.effective_groups/2
    require(abs(mass-1)<1e-12,'Enumeration probability mass differs')
    return {'probability_mass':mass,'expected_descent':expectation,'effective_group_probability':effective,'outcomes_enumerated':256,
            'reward_mode':reward_mode,'scope':'Synthetic independent one-token Bernoulli model only'}


def source_contract():
    root=Path(__file__).resolve().parents[2]
    names=('mlx_gspo_smoke.py','mlx_continuous_gspo.py','uniform_balance_continuation_gate_v1.py')
    tree=ast.parse((root/'src/mri_rl/mlx_continuous_gspo.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
    return {'native_loss_AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'native_derivative_convention':'fresh-stop_gradient(fresh)+stop_gradient(mean(fresh-ownold)); CPU dual-number AST oracle checks this exact source, no native import.',
            'reward_contract':'Grade.total/accuracy/format/overlong carried unchanged from authenticated task-routed calibrated reward; both prompts require identical reward-policy/calibration SHA. Synthetic rawbinary fixtures do not replace real calibration.',
            'unmodified_sources_sha256':{name:hashlib.sha256((root/'src/mri_rl'/name).read_bytes()).hexdigest() for name in names},
            'admission_gate':'inherited Adam60 uniform_balance_continuation_gate_v1.accuracy_group_is_informative; valid-format-subset correctness variation. Held identical in both future arms.',
            'historical18_gate_difference':'mlx_gspo_smoke.task_group_is_informative requires every format==1 and differs from inherited Adam60. Historical strict predicate is descriptive only; it is not substituted into this proposed comparison.',
            'advantage_std':'Sample standard deviation(ddof1)+epsilon1e-6; existing function reused',
            'ratio':'exp(min(mean(current-ownold),10))','clip_low':CLIP_LOW,'clip_high':CLIP_HIGH,
            'block_normalization':'Mean of2K4 losses=sum8 response losses/8, masked by unchanged inherited Adam60 valid-format-subset gate',
            'native_execution':False}


def variable_length_constant_reward_example():
    """Two complete response outcomes, probabilities p,1−p and lengths1,2.

    Rewards are constant conditional on each prompt. E[∇logπ]=0, whereas
    E[∇logπ/T]=p(1−p)/2. Cross-prompt nonzero constant advantage activates
    this mean-token weighting. This is a synthetic policy, not MRI data.
    """
    p=.5
    expected_joint=p*(1-p)+(1-p)*(-p)
    expected_mean=p*(1-p)/1+(1-p)*(-p)/2
    plan=make_plan(synthetic_block([1]*4,[0]*4),'paired_prompt')
    advantage=next(item.advantage for item in plan.items if item.response.context.modality=='F')
    return {'p':p,'complete_response_lengths':[1,2],'expected_joint_score':expected_joint,
            'expected_mean_logp_score':expected_mean,'mixed_descent_F_logit':.5*advantage*expected_mean,
            'true_reward_gradient':0.,'assumptions':'Constant per-prompt reward; F responses lengths1/2, T fixed-length independent policy; synthetic complete-response distribution.'}


def mechanism_report():
    """Deterministic public CPU counterexamples, with zero actual data claims."""
    block=synthetic_block([1]*4,[0]*4);same=make_plan(block,'same_prompt');mixed=make_plan(block,'paired_prompt')
    equal={r.response_id:[1.] for r in block};prior={r.response_id:[2. if r.context.modality=='F' else 1.] for r in block}
    prior_direction=local_descent(mixed,prior)[0]
    format_only=make_plan(synthetic_block([0]*4,[0]*4,format_F=1,format_T=0,total_F=1.,total_T=0.),'paired_prompt')
    half_format=make_plan(synthetic_block([1]*4,[0]*4,format_F=.5,format_T=1,total_F=1.5,total_T=.5),'paired_prompt')
    bern_same=enumerate_bernoulli(.3,.9,'same_prompt');bern_mixed=enumerate_bernoulli(.3,.9,'paired_prompt')
    true=[.3*.7/2,.9*.1/2];slope=-2.346
    shared=lambda vector:vector[0]+slope*vector[1]
    direction=shared(bern_mixed['expected_descent'])
    calibrated=tuple(replace(r,grade=Grade(.7 if r.context.modality=='F' else .3,1.,9.25 if r.context.modality=='F' else 8.75,raw_accuracy=1. if r.context.modality=='F' else 0.)) for r in block)
    subset_fixture=[{'accuracy':a,'format':f,'total':a,'overlong':0.} for a,f in [(1.,1.),(0.,1.),(1.,.5),(0.,1.)]]
    constant=enumerate_bernoulli(.8,.2,'paired_prompt',reward_mode='prompt_constant')
    return {'schema':'paired-prompt-CPU-mechanism-v1','status':STATUS,'evidence_tier':'CPU_SYNTHETIC_EXACT_ENUMERATION',
        'native_model_imports':0,'model_loads':0,'native_calls':0,'generations':0,'optimizer_updates':0,
        'adaptation':'Group-relative baseline/std scope changes from same-input K4 to fixed F2T2 K4; both blocks contain the same8 responses. Inherited Adam60 calibrated reward, sample-std advantages, ratio, clips and valid-format-subset informative predicate preserved identically across arms.',
        'source_contract':source_contract(),
        'cases':{
            'all_F_correct_all_T_wrong':{'same_effective_groups':same.effective_groups,'paired_effective_groups':mixed.effective_groups,
                'same_response_count':8,'paired_response_count':8,'T_correct_responses_created':0,
                'paired_advantages':{item.response.response_id:item.advantage for item in mixed.items},
                'same_group_diagnostics':group_diagnostics(block,'same_prompt'),'paired_group_diagnostics':group_diagnostics(block,'paired_prompt'),
                'interpretation':'Cross-input reward variation creates admitted groups although each input has constant observed accuracy; no within-input solution contrast is added.'},
            'zero_conditional_Jacobian_difference':{'paired_effective_groups':2,'local_descent':local_descent(mixed,equal),
                'interpretation':'Identical complete-response mean-logp Jacobians cancel despite admitted groups; effective groups do not establish visual learning.'},
            'shared_FLAIR_prior_harms_T':{'observed_mean_logp_derivatives':{'F':2.,'T':1.},'mixed_local_descent':prior_direction,
                'F_correct_probability':.75,'T_wrong_FLAIR_probability':.5,'F_logit_theta_slope':8.,'T_wrong_logit_theta_slope':2.,
                'd_T_correct_probability_along_descent':-.5*prior_direction,
                'sampling_event_probability':.75**4*.5**4,
                'interpretation':'A possible synthetic on-policy eight-FLAIR-response event yields an update that strengthens a shared FLAIR prior and lowers T correctness.'},
            'format_only_variance':{'reward_variable_groups':format_only.reward_variable_groups,'inherited_Adam60_gate_admitted_groups':format_only.effective_groups,
                'interpretation':'Reward variance alone cannot replace the retained accuracy/format/overlong predicate.'},
            'half_format_preserved':{'format_values':[.5,1.],'original_total_rewards':{'F':1.5,'T':.5},'reward_variable_groups':half_format.reward_variable_groups,'inherited_Adam60_gate_admitted_groups':half_format.effective_groups,
                'raw_advantages':{item.response.response_id:item.advantage for item in half_format.items},'loss_weights':[item.weight for item in half_format.items],
                'interpretation':'Half-format records and supplied advantages remain logged; the inherited Adam60 gate finds only wrong answers in the format1 subset here and rejects these groups. It does not require all group formats to equal1.'},
            'calibrated_receipt_preserved':{'input_accuracy_values':[.7,.3],'input_raw_accuracy_values':[1.,0.],'input_total_rewards':[9.25,8.75],
                'group_diagnostics':group_diagnostics(calibrated,'paired_prompt'),'inherited_Adam60_gate_admitted_groups':make_plan(calibrated,'paired_prompt').effective_groups,
                'interpretation':'Caller-supplied calibrated accuracy/raw correctness/total are preserved without reward recomputation; inherited Adam60 predicate uses unchanged calibrated accuracy.'},
            'gate_attribution_countercheck':{'grades':subset_fixture,'inherited_Adam60_gate_admitted':inherited_Adam60_gate(subset_fixture),'historical18_all_format_strict_gate_admitted':historical18_strict_gate(subset_fixture),'interpretation':'Valid-format subset contains correct+wrong while another response has format.5; inherited Adam60 allows this group, historical18 all-format strict rejects. Only inherited Adam60 is used by both proposed arms.'},
            'prompt_constant_fixed_length':constant,
            'prompt_constant_variable_length':variable_length_constant_reward_example(),
            'difficulty_and_std_weighting':{'p_F_correct':.3,'p_T_correct':.9,'same_prompt':bern_same,'paired_prompt':bern_mixed,
                'true_balanced_reward_gradient':true,'mixed_component_scaling':[value/target for value,target in zip(bern_mixed['expected_descent'],true)],
                'shared_theta_logit_slopes':[1.,slope],'true_balanced_shared_theta_gradient':shared(true),
                'same_prompt_shared_theta_descent':shared(bern_same['expected_descent']),'paired_shared_theta_descent':direction,
                'T_correctness_direction_under_paired':.9*.1*slope*direction,
                'interpretation':'Both normalized estimators can reweight expected-reward gradients; changing baseline/std across prompts can reverse the shared-parameter direction. This counterexample does not assign measured MRI probabilities.'}},
        'mathematical_verdict':{
            'finite_fixed_sample_conditional_surrogate':'WELL_DEFINED away from clip/cap derivative ties, if every response is scored under its own original context and token span with genuine own-old likelihoods.',
            'unbiased_original_expected_reward_gradient':'NOT_ESTABLISHED; action-dependent group std and own-sample centering change weighting. Cross-prompt difficulty changes relative weights. Original same-prompt normalization also has bias; paired grouping changes it.',
            'constant_reward_baseline':'For an action-independent conditional baseline and complete-response fixed-length joint score, E[gradient logpi|x]=0. Thus nonzero constant cross-prompt advantages have zero expected signal, but finite-sample variance. Variable T mean-score weighting need not have zero expectation.',
            'guaranteed_T1_gain':False,'group_effectiveness_is_accuracy':False,'automatic_training_GO':False},
        'attribution_contract':{'block_loss_weight':'1/8 once for every admitted response; no repeated1/K','response_identity':'Own input context/prompt/image/response tokens/oldlogps/reward receipt remain attached across grouping.',
            'external_authentication_required':'Typed equality does not prove real generation, checkpoint or reward provenance; authenticate raw receipts separately.',
            'native_adaptation_requirement':'Future native code must score each response with its own context; historical self.active singleprompt update cannot accept mixedcontext tokens unchanged.'},
        'offline_real_matched_evidence':'NOT_AVAILABLE_IN_THIS_CPU_MECHANISM_ARTIFACT; separate offline availability audit required. No real outputs read, paired, generated or fabricated here.',
        'future_matching_limit':'CPU reassignment uses exactly the same fixed-model8responses. Future on-policy A/B trajectories with pergroup updates cannot promise identical realized responses after the first update; match parent/state/source/exposure/token caps instead. No deferred stale second update claimed on-policy.',
        'failure_boundaries':['All rewards equal⇒all advantages0.', 'Format-only variance can remain inadmissible under inherited Adam60 gate.', 'Conditional Jacobian cancellation gives zero group gradient.', 'Shared parameters/class prior can harm T despite more effective groups.', 'No guarantee for parameter step, Adam direction, greedy accuracy, independent patients or cross-task transfer.', 'Groups with disparate format/overlong patterns can fail the retained gate.', 'Different checkpoint/policy or response context invalidates a matched fixed-block comparison.'],
        'mission_alignment':{'tag':'PRIMARY_VERO_TRANSFER','change':'Vero-style group construction only','evidence':'PROPOSED adaptation plus CPU counterexamples','accuracy_gain':'NOT_PROVEN'}}
