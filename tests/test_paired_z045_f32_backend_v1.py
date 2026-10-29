"""CPU source/API fixtures only; no native MLX imports or model loading."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from mri_rl import paired_z045_f32_backend_v1 as p


class MX:
    float32=np.float32
    @staticmethod
    def eval(*args):pass
    @staticmethod
    def array_equal(a,b):return np.array_equal(a,b)


def test_model_view_copies_features_without_weight_or_structure_mutation():
    features={'inputs_embeds':np.ones((1,2,3),dtype=np.float16),'deepstack_visual_embeds':[np.ones((2,3),dtype=np.float16)],'position_ids':np.ones((3,1,2),dtype=np.int64)}
    language=NS(model=NS(embed_tokens=lambda ids:np.ones((*ids.shape,3),dtype=np.float16)))
    base=NS(language_model=language,get_input_embeddings=lambda *a,**k:NS(to_dict=lambda:features),config=NS(kind='fixed'))
    view=p.F32GenerationModelView(base,MX,lambda name:None,[])
    result=view.get_input_embeddings(np.ones((1,2)),np.ones((1,2)))
    assert result.inputs_embeds.dtype==np.float32 and result.to_dict()['deepstack_visual_embeds'][0].dtype==np.float32
    assert features['inputs_embeds'].dtype==np.float16 and features['deepstack_visual_embeds'][0].dtype==np.float16
    assert view.config is base.config and view.language_model.model is base.language_model.model


def test_language_view_upcasts_supplied_prefill_and_decode_embeddings_without_casting_weights():
    observed=[]
    class Language:
        model=NS(embed_tokens=lambda ids:np.ones((*ids.shape,3),dtype=np.float16))
        def __call__(self,*args,**kwargs):
            observed.append(kwargs['inputs_embeds']);return NS(logits=np.ones((1,1,7),dtype=np.float32))
    evidence=[];view=p.F32LanguageView(Language(),MX,lambda name:None,evidence)
    view(np.array([[1,2]]),inputs_embeds=np.ones((1,2,3),dtype=np.float16))
    view(np.array([[3]]))
    assert all(v.dtype==np.float32 for v in observed) and len(evidence)==2
    assert [v.shape[1] for v in observed]==[2,1]


def test_stream_capture_includes_EOS_logp_and_keeps_installed_text_segments():
    rows=[NS(token=4,logprobs=np.array([-.5]*8,dtype=np.float32),generation_tokens=1,prompt_tokens=3,text='hello',finish_reason=None),
          NS(token=7,logprobs=np.array([-.5]*8,dtype=np.float32),generation_tokens=2,prompt_tokens=3,text='!',finish_reason='stop',token_ids=[4,7])]
    result=p.capture_stream(iter(rows),prompt_tokens=3,max_tokens=256,eos_ids={7},before=lambda name:None)
    assert result['token_ids']==[4,7] and result['raw_selected_logps']==[-.5,-.5]
    assert result['text']=='hello!' and result['EOS_coverage']=='OBSERVED_AND_SCORED'


def test_stream_duplicate_final_token_not_double_counted_and_length_termination_kept():
    row=dict(token=4,logprobs=np.array([-.5]*8,dtype=np.float32),generation_tokens=1,prompt_tokens=3)
    rows=[NS(**row,text='x',finish_reason=None),NS(**row,text='y',finish_reason='length',token_ids=[4])]
    result=p.capture_stream(iter(rows),prompt_tokens=3,max_tokens=1,eos_ids={7},before=lambda name:None)
    assert result['token_ids']==[4] and result['text']=='xy' and result['finish_reason']=='length'
    with pytest.raises(ValueError):p.capture_stream(iter(rows),prompt_tokens=4,max_tokens=1,eos_ids={7},before=lambda name:None)


def test_generation_alignment_fixed_tolerance_and_nonfinite_failclosed():
    assert p.generation_alignment([-.1,-2.],[-.1,-2.])['max_abs_error']==0
    for a,b in [([-.1],[-.103]),([-.1],[]),([float('nan')],[-.1])]:
        with pytest.raises(ValueError):p.generation_alignment(a,b)


def test_update_checks_rollout_selected_before_unchanged_GSPO_super_update(monkeypatch):
    backend=p.F32NativeGSPOBackend.__new__(p.F32NativeGSPOBackend)
    backend.mx=MX;backend.before_unit=lambda name:None;backend.active={'record_id':'x'}
    backend._logps=lambda tokens:np.array([-.1],dtype=np.float32)
    called=[]
    monkeypatch.setattr(p.NativeGSPOBackend,'update',lambda self,*args:called.append(True) or {'unchanged':'loss'})
    raw={'responses':[{'token_ids':[1],'raw_selected_logps':[-.1]}]}
    assert backend.update({'id':'x'},raw,[1.])['unchanged']=='loss' and called==[True]
    raw['responses'][0]['raw_selected_logps']=[-.2]
    with pytest.raises(ValueError):backend.update({'id':'x'},raw,[1.])
    assert called==[True]


def test_effective_EOS_uses_installed_resolved_additional_ids_after_reset():
    class Stop:
        def reset(self,ids):self.eos_token_ids=[ids,99]
    tokenizer=NS(stopping_criteria=Stop())
    assert p.reset_effective_eos(tokenizer,7)=={7,99}


def test_inherited_BF16_evaluation_methods_are_failclosed_for_new_policy():
    backend=p.F32NativeGSPOBackend.__new__(p.F32NativeGSPOBackend)
    for method in ('generate_record','greedy_generate','score_record'):
        with pytest.raises(ValueError,match='F32_EVALUATION_NOT_QUALIFIED'):getattr(backend,method)({},None,256)


def test_copied_full_scorer_is_exact_original_AST_except_observer():
    import ast,inspect,textwrap
    from mri_rl.conditional_gspo_gradient_v2 import f32_activation_response_logps
    old=ast.parse(textwrap.dedent(inspect.getsource(f32_activation_response_logps))).body[0]
    new=ast.parse(textwrap.dedent(inspect.getsource(p.f32_response_logps_observed))).body[0]
    new.name=old.name;new.args=old.args
    new.body=[n for n in new.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='vocab_observer')]
    assert ast.dump(new)==ast.dump(old)


def test_vocabulary_capture_includes_one_lookahead_and_no_dtype_conversion_loss(tmp_path):
    writer=p.VocabularyWriter(tmp_path/'vocab_PRIVATE.bin');processor=p.ObservedFP32Processor(MX,writer)
    for i in range(3):processor(None,np.array([[1.,2.,3.]],dtype=np.float16))
    receipt=writer.finalize([1,2])
    assert receipt['returned_rows']==2 and receipt['lookahead_rows']==1
    assert receipt['rows_including_unreturned_lookahead']==3
    np.testing.assert_array_equal(np.fromfile(receipt['path_PRIVATE'],dtype='<f4').reshape(3,3),[[1.,2.,3.]]*3)


def test_actual_prompt_cost_must_match_fixed_ledger_before_generation():
    assert p.require_prompt_cost('mri_sequence',213)==213
    assert p.require_prompt_cost('mri_visit_correspondence',405)==405
    for task,length in [('mri_sequence',214),('mri_visit_correspondence',406),('other',213)]:
        with pytest.raises(ValueError,match='prompt.*ledger'):p.require_prompt_cost(task,length)
