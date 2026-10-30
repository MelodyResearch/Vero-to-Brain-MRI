"""Frozen source integrity and CPU-only compatibility; never load a model."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from mri_rl.historical_runtime import COMMIT, export_source, verify_source, activate, source_spec

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def frozen_source(tmp_path_factory):
    destination = tmp_path_factory.mktemp("source-parent") / "frozen-engine"
    export_source(destination, repository=ROOT)
    return destination

@pytest.fixture(scope="session")
def frozen_runtime(frozen_source):
    scope = {"historical_runtime_root": str(frozen_source), "file_sha256": {
        str((frozen_source / name).resolve()): digest for name, digest in source_spec()["files"].items()}}
    runtime = activate(scope)
    yield runtime
    runtime.close()

@pytest.fixture
def p(frozen_runtime):
    return frozen_runtime.module("paired_z045_f32_backend_v1")


def test_export_is_exact_source_only_and_refuses_overwrite(frozen_source):
    files = verify_source(frozen_source)
    assert len(files) == 90
    assert all(Path(name).suffix == ".py" for name in files)
    assert COMMIT == "1e0d7be1bb099b42e717eb4eb081facc12607ed3"
    with pytest.raises(ValueError, match="exists"):
        export_source(frozen_source, repository=ROOT)


def test_runtime_rejects_unpinned_missing_modified_or_extra_source(frozen_source, tmp_path):
    clone = tmp_path / "source"
    shutil.copytree(frozen_source, clone)
    first = next(iter(source_spec()["files"]))
    target = clone / first
    saved = target.read_bytes()
    target.write_bytes(saved + b"\n# changed\n")
    with pytest.raises(ValueError, match="SHA"):
        verify_source(clone)
    target.write_bytes(saved)
    target.unlink()
    with pytest.raises(ValueError):
        verify_source(clone)
    target.write_bytes(saved)
    (clone / "src/mri_rl/unreviewed.py").write_text("raise RuntimeError('unreviewed')")
    with pytest.raises(ValueError, match="extra"):
        verify_source(clone)



def test_runtime_rejects_special_files(frozen_source, tmp_path):
    clone = tmp_path / "source"
    shutil.copytree(frozen_source, clone)
    os.mkfifo(clone / "unreviewed_fifo")
    with pytest.raises(ValueError, match="Special"):
        verify_source(clone)


def test_native_scope_must_pin_every_actual_exported_file(frozen_source):
    with pytest.raises(ValueError, match="scope"):
        verify_source(frozen_source, {"file_sha256": {}})


def test_dynamic_source_reads_and_restore_remain_exact(frozen_runtime):
    module = frozen_runtime.module("accuracy_pilot_v2_native")
    origin, node = module._context_node()
    assert node.name == "_prepare_context"
    assert Path(origin.__file__).is_file()
    binding = frozen_runtime.module("safe_micro_native_v1").capped_init_binding()
    assert binding["source_sha256"] == "aceda9c4e212667d512e747447f3ef5533b91565f6cebc5a928b8a805c0b96e3"
    assert binding["policy"]["memory_guideline_gib"] == 12
    assert binding["policy"]["cache_limit_mib"] == 256
    restore = frozen_runtime.module("safe_micro_restore_v1")
    original = {"step": 52, "state": [{"m": np.array([1.0])}]}
    detached = restore.detach_containers(original, copy_leaf=lambda x: x.copy() if hasattr(x, "copy") else x)
    original["step"] = 53
    original["state"][0]["m"][0] = 9
    assert detached["step"] == 52 and detached["state"][0]["m"][0] == 1


def test_readonly_dependency_imports_do_not_import_native_runtime(frozen_source):
    code = '''
import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:
   raise RuntimeError('native import forbidden in CPU test: '+fullname)
sys.meta_path.insert(0, NoNative())
from mri_rl.historical_runtime import activate, source_spec
from pathlib import Path
root=Path(sys.argv[1]);spec=source_spec()
scope={'historical_runtime_root':str(root),'file_sha256':{str((root/p).resolve()):h for p,h in spec['files'].items()}}
r=activate(scope)
for name in spec['direct_entry_modules']: r.module(name)
assert not any(n.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for n in sys.modules)
r.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(frozen_source)],
                            cwd=ROOT / "src", text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_dynamic_resource_limits_and_once_only_records_are_preserved(frozen_runtime, tmp_path):
    resource = frozen_runtime.module("safe_micro_dynamic_resources_v1")
    sample = NS(available_gib=7., pressure_free_percent=38., swap_used_gib=18., process_rss_gib=.1,
                battery_percent=80, power_source='AC', thermal_state='SENSOR_UNAVAILABLE',
                other_heavy_processes=0, mlx_active_gib=None, mlx_peak_gib=None)
    for level in [1, 2]:
        assert resource.Controller().decide(sample, level, 100., 0.)['action'] == 'CONTINUE'
    assert resource.Controller().decide(sample, 4, 100., 0.)['action'] == 'PAUSE_RELEASE'
    for key, value in [('process_rss_gib', 15), ('mlx_active_gib', 13), ('available_gib', 1)]:
        bad = NS(**vars(sample)); setattr(bad, key, value)
        assert resource.Controller().decide(bad, 1, 100., 0.)['action'] == 'STOP'
    once = frozen_runtime.module("shared_rollout_gspo_core_v1").publish_once
    record = tmp_path / 'once.json'
    once(record, {'updates': 0})
    with pytest.raises(Exception):
        once(record, {'updates': 1})
    assert json.loads(record.read_text()) == {'updates': 0}

# Original F32 view/stream CPU assertions retained; engine is now an immutable dependency.
class MX:
    float32=np.float32
    @staticmethod
    def eval(*args):pass
    @staticmethod
    def array_equal(a,b):return np.array_equal(a,b)


def test_model_view_copies_features_without_weight_or_structure_mutation(p):
    features={'inputs_embeds':np.ones((1,2,3),dtype=np.float16),'deepstack_visual_embeds':[np.ones((2,3),dtype=np.float16)],'position_ids':np.ones((3,1,2),dtype=np.int64)}
    language=NS(model=NS(embed_tokens=lambda ids:np.ones((*ids.shape,3),dtype=np.float16)))
    base=NS(language_model=language,get_input_embeddings=lambda *a,**k:NS(to_dict=lambda:features),config=NS(kind='fixed'))
    view=p.F32GenerationModelView(base,MX,lambda name:None,[])
    result=view.get_input_embeddings(np.ones((1,2)),np.ones((1,2)))
    assert result.inputs_embeds.dtype==np.float32 and result.to_dict()['deepstack_visual_embeds'][0].dtype==np.float32
    assert features['inputs_embeds'].dtype==np.float16 and features['deepstack_visual_embeds'][0].dtype==np.float16
    assert view.config is base.config and view.language_model.model is base.language_model.model


def test_language_view_upcasts_supplied_prefill_and_decode_embeddings_without_casting_weights(p):
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


def test_stream_capture_includes_EOS_logp_and_keeps_installed_text_segments(p):
    rows=[NS(token=4,logprobs=np.array([-.5]*8,dtype=np.float32),generation_tokens=1,prompt_tokens=3,text='hello',finish_reason=None),
          NS(token=7,logprobs=np.array([-.5]*8,dtype=np.float32),generation_tokens=2,prompt_tokens=3,text='!',finish_reason='stop',token_ids=[4,7])]
    result=p.capture_stream(iter(rows),prompt_tokens=3,max_tokens=256,eos_ids={7},before=lambda name:None)
    assert result['token_ids']==[4,7] and result['raw_selected_logps']==[-.5,-.5]
    assert result['text']=='hello!' and result['EOS_coverage']=='OBSERVED_AND_SCORED'


def test_stream_duplicate_final_token_not_double_counted_and_length_termination_kept(p):
    row=dict(token=4,logprobs=np.array([-.5]*8,dtype=np.float32),generation_tokens=1,prompt_tokens=3)
    rows=[NS(**row,text='x',finish_reason=None),NS(**row,text='y',finish_reason='length',token_ids=[4])]
    result=p.capture_stream(iter(rows),prompt_tokens=3,max_tokens=1,eos_ids={7},before=lambda name:None)
    assert result['token_ids']==[4] and result['text']=='xy' and result['finish_reason']=='length'
    with pytest.raises(ValueError):p.capture_stream(iter(rows),prompt_tokens=4,max_tokens=1,eos_ids={7},before=lambda name:None)


def test_generation_alignment_fixed_tolerance_and_nonfinite_failclosed(p):
    assert p.generation_alignment([-.1,-2.],[-.1,-2.])['max_abs_error']==0
    for a,b in [([-.1],[-.103]),([-.1],[]),([float('nan')],[-.1])]:
        with pytest.raises(ValueError):p.generation_alignment(a,b)


def test_update_checks_rollout_selected_before_unchanged_GSPO_super_update(p, monkeypatch):
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


def test_effective_EOS_uses_installed_resolved_additional_ids_after_reset(p):
    class Stop:
        def reset(self,ids):self.eos_token_ids=[ids,99]
    tokenizer=NS(stopping_criteria=Stop())
    assert p.reset_effective_eos(tokenizer,7)=={7,99}


def test_inherited_BF16_evaluation_methods_are_failclosed_for_new_policy(p):
    backend=p.F32NativeGSPOBackend.__new__(p.F32NativeGSPOBackend)
    for method in ('generate_record','greedy_generate','score_record'):
        with pytest.raises(ValueError,match='F32_EVALUATION_NOT_QUALIFIED'):getattr(backend,method)({},None,256)


def test_copied_full_scorer_is_exact_original_AST_except_observer(p, frozen_runtime):
    import ast,inspect,textwrap
    f32_activation_response_logps = frozen_runtime.module("conditional_gspo_gradient_v2").f32_activation_response_logps
    old=ast.parse(textwrap.dedent(inspect.getsource(f32_activation_response_logps))).body[0]
    new=ast.parse(textwrap.dedent(inspect.getsource(p.f32_response_logps_observed))).body[0]
    new.name=old.name;new.args=old.args
    new.body=[n for n in new.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='vocab_observer')]
    assert ast.dump(new)==ast.dump(old)


def test_vocabulary_capture_includes_one_lookahead_and_no_dtype_conversion_loss(p, tmp_path):
    writer=p.VocabularyWriter(tmp_path/'vocab_PRIVATE.bin');processor=p.ObservedFP32Processor(MX,writer)
    for i in range(3):processor(None,np.array([[1.,2.,3.]],dtype=np.float16))
    receipt=writer.finalize([1,2])
    assert receipt['returned_rows']==2 and receipt['lookahead_rows']==1
    assert receipt['rows_including_unreturned_lookahead']==3
    np.testing.assert_array_equal(np.fromfile(receipt['path_PRIVATE'],dtype='<f4').reshape(3,3),[[1.,2.,3.]]*3)


def test_actual_prompt_cost_must_match_fixed_ledger_before_generation(p):
    assert p.require_prompt_cost('mri_sequence',213)==213
    assert p.require_prompt_cost('mri_visit_correspondence',405)==405
    for task,length in [('mri_sequence',214),('mri_visit_correspondence',406),('other',213)]:
        with pytest.raises(ValueError,match='prompt.*ledger'):p.require_prompt_cost(task,length)
