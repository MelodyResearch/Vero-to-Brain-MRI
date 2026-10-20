"""CPU-only matched evaluation tests; no native model imports or outputs."""
import copy
import pytest
from mri_rl import paired_z045_matched_f32_eval_v1 as e
from mri_rl.answer_likelihood import PREFIX


def frozen():
    records=[]
    for i in range(40):
        task='mri_sequence' if i<24 else 'mri_visit_correspondence';labels=e.LABELS[task]
        row={'id':str(i),'ability':task,'images':[f'/opaque/{i}.png'],'reward_model':{'ground_truth':labels[i%2]},'extra_info':{'question':'fixed','patient_key':str(i//2),'level':45 if i<8 else 35}}
        records.append({'id':str(i),'condition':'real_image','row_PRIVATE':row,'image_sha256':str(i).zfill(64)})
    for i in range(8):
        opposite=records[i^1];row=copy.deepcopy(opposite['row_PRIVATE']);row['id']='swap'+str(i)
        records.append({'id':row['id'],'condition':'same_patient_same_level_opposite_modality','row_PRIVATE':row,'image_sha256':opposite['image_sha256'], 'original_real_id_PRIVATE':str(i),'reuse_real_id_PRIVATE':str(i^1)})
    return {'records_PRIVATE':records,'candidate_prefix':PREFIX,'label_order':e.LABELS}


def test_exact40_real_and8_aliases_reuse_without_extra_calls():
    plan=e.compile_records(frozen());assert len(plan['real'])==40 and len(plan['aliases'])==8
    result=e.alias_outputs({'0':{'text':'x'},'1':{'text':'y'}},{'swap0':{'reuse_real_id_PRIVATE':'1'}})
    assert result['swap0']['text']=='y'
    bad=frozen();bad['records_PRIVATE'][-1]['image_sha256']='wrong'
    with pytest.raises(ValueError):e.compile_records(bad)


def test_greedy_is_exact_original_temp0_256_with_F32_final_processor():
    assert e.greedy_settings()=={'max_tokens':256,'temperature':0.,'top_p':1.,'top_k':0,'min_p':0.,'repetition_penalty':None,'verbose':False}


def test_candidate_strict_spans_exclude_prefix_and_length_metrics_are_separate():
    result=e.label_scores([-99.,-88.,-1.,-3.],2)
    assert result=={'mean':-2.,'sum':-4.,'first':-1.,'token_count':2,'label_logps_PRIVATE':[-1.,-3.]}
    with pytest.raises(ValueError):e.label_scores([-.1],2)


def test_prefix_oracle_fixed_threshold_fails_closed():
    assert e.prefix_alignment([-.1,-2.],[-.1,-2.])['max_abs_error']==0
    with pytest.raises(ValueError):e.prefix_alignment([-.1],[-.103])


def test_finite_caps_count_only_real_rows_and_refuse_extra():
    b=e.budget();assert b['greedy_generations']==80 and b['candidate_score_API_calls']==240 and b['maximum_generated_tokens']==20480
    assert b['teacherforced_input_positions']==102000 and b['optimizer_updates']==0 and b['maximum_seconds']==2400
    c=e.Counter();c.charge('teacherforced_input_positions',102000)
    with pytest.raises(ValueError):c.charge('teacherforced_input_positions',1)


def test_unknown_child_cannot_be_executed_or_substituted_with_parent():
    with pytest.raises(ValueError,match='terminal'):e.require_terminal(None,'parent')
    with pytest.raises(ValueError):e.require_terminal({'receipt_sha256':'parent'},'parent')


def test_actual_candidate_call_shape_cap_and_token_boundary():
    def encode(text):
        if text==PREFIX:return [9,8]
        if text=='FLAIR':return [1,2]
        if text=='contrast-enhanced T1':return [3,4,5,6,7]
        for label in ('FLAIR','contrast-enhanced T1'):
            if text==PREFIX+label:return encode(PREFIX)+encode(label)
        raise ValueError(text)
    plan=e.token_plan('mri_sequence',PREFIX,encode)
    assert [len(v) for v in plan['labels']]==[2,5]
    assert e.score_input_positions(405,16,5)==425
    with pytest.raises(ValueError):e.score_input_positions(406,16,5)


def test_saved_analyzer_call_schema_uses_percheckpoint_deltas():
    c=e.Counter();c.charge('greedy_generations',40);c.charge('maximum_generated_tokens',100);c.charge('teacherforced_input_positions',2000)
    result=e.call_receipt(c,{'greedy_generations':20,'maximum_generated_tokens':30,'teacherforced_input_positions':500})
    assert result['greedy_calls']==20 and result['generated_tokens']==70 and result['teacherforced_input_positions']==1500


def test_common_prefix_agreement_keeps_original_shape_guard_without_extra_forward():
    assert e.common_prefix_alignment([[-1.,-2.],[-1.,-2.]])['max_abs_error']==0
    with pytest.raises(ValueError):e.common_prefix_alignment([[-1.,-2.],[-1.,-2.003]])


def test_weights_and_Adam_must_match_before_RNG_restoration_no_optimizer_call():
    assert e.audit_before_restore({'a':'same'},{'a':'same'},{'step':60},{'step':60})
    restored=[]
    with pytest.raises(ValueError):
        e.audit_before_restore({'a':'before'},{'a':'changed'},{'step':60},{'step':60});restored.append(True)
    with pytest.raises(ValueError):e.audit_before_restore({'a':'same'},{'a':'same'},{'step':60},{'step':61})
    assert restored==[]


def test_terminal_selector_rejects_foreign_spec_schedule_or_configuration(tmp_path):
    import json
    from mri_rl import paired_z045_continuation_entry_v1 as entry
    schedule=[{'id':'fixed'}];schedule_path=tmp_path/'schedule.json';schedule_path.write_text(json.dumps({'schedule':schedule}))
    cs={'candidate_schedule':str(schedule_path),'file_sha256':{'candidate_schedule':entry.digest(schedule_path)},'resume':False,'immutable':'fixedstage'}
    cs_path=tmp_path/'continuation.json';cs_path.write_text(json.dumps(cs))
    checkpoint=tmp_path/'boundary_0001';(checkpoint/'optimizer').mkdir(parents=True)
    configuration={**entry.driver.PROFILE,'group_size':4,'max_tokens':256};parent={'receipt_sha256':'parent'}
    schedule_sha=entry.scope_digest(schedule)
    binding={'experiment':{'stage':'paired_z045_continuation_v1','adaptive_parent':parent,'spec_sha256':entry.stage_spec_binding(cs),'schedule_sha256':schedule_sha},'schedule_sha256':schedule_sha,'configuration':configuration}
    manifest={'bindings':{'run':binding,'adapter_sha256':'adapter'},'optimizer_updates':61}
    manifest_path=checkpoint/'optimizer/manifest.json';manifest_path.write_text(json.dumps(manifest))
    receipt={'schema':'continuous-rl-boundary-v1','completed_groups':1,'optimizer_updates':1,'absolute_completed_groups':130,'absolute_optimizer_updates':61,'binding':binding,'files_sha256':{'optimizer/manifest.json':entry.digest(manifest_path),'optimizer/state.safetensors':'Adam','adapter.safetensors':'adapter'}}
    receipt_path=checkpoint/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    complete=tmp_path/'complete.json';complete.write_text(json.dumps({'status':'WALL_BUDGET_REACHED','completed_groups':1,'optimizer_updates':1,'boundary':str(checkpoint)}))
    spec={'terminal_checkpoint':str(checkpoint),'terminal_receipt_sha256':entry.digest(receipt_path),'terminal_completion':str(complete),'terminal_completion_sha256':entry.digest(complete),'continuation_spec':str(cs_path),'continuation_spec_sha256':entry.digest(cs_path)}
    assert e.terminal_metadata(spec,parent)['groups']==130
    for field in ('spec','schedule','configuration'):
        changed=json.loads(json.dumps(binding))
        if field=='spec':changed['experiment']['spec_sha256']='FOREIGN-SPEC'
        elif field=='schedule':changed['experiment']['schedule_sha256']=changed['schedule_sha256']='FOREIGN-SCHEDULE'
        else:changed['configuration']={'foreign':'profile'}
        manifest['bindings']['run']=changed;manifest_path.write_text(json.dumps(manifest))
        receipt['binding']=changed;receipt['files_sha256']['optimizer/manifest.json']=entry.digest(manifest_path);receipt_path.write_text(json.dumps(receipt));spec['terminal_receipt_sha256']=entry.digest(receipt_path)
        with pytest.raises(ValueError,match='continuation'):e.terminal_metadata(spec,parent)
