"""CPU qualification plan and original-loss AST tests; no native imports."""
import ast
import inspect
import textwrap
import pytest
from mri_rl import paired_z045_f32_qualification_v1 as q
from mri_rl.mlx_continuous_gspo import NativeGSPOBackend


def test_first_fixed_rows_and_budget_no_outcome_selection():
    schedule=[{'id':'d0','ability':'mri_visit_correspondence'},{'id':'s0','ability':'mri_sequence'},{'id':'s1','ability':'mri_sequence'}]
    assert [r['id'] for r in q.fixed_rows(schedule)]==['s0','d0']
    b=q.budget();assert b['model_loads']==1 and b['generated_response_tokens']<=2048 and b['backward_calls']==8
    assert b['language_model_calls']==2080 and b['vision_embedding_calls']==24 and b['maximum_seconds']==1200
    assert b['optimizer_updates']==0


def test_extracted_loss_AST_is_original_GSPO_not_subclass_update_or_CE():
    node=q.frozen_loss_node()
    original=ast.parse(textwrap.dedent(inspect.getsource(NativeGSPOBackend.update)))
    expected=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
    assert ast.dump(node)==ast.dump(expected)


def test_no_informative_reward_fallback():
    grade={'accuracy':1.,'format':1.,'overlong':0.,'total':1.}
    assert q.qualification_active([grade]*4) is False
    varied=[grade,dict(grade,accuracy=0.,total=.2),grade,grade]
    assert q.qualification_active(varied)


def test_actual_full_vocabulary_missing_is_partial_and_numeric_gate_uses_same_rows(tmp_path):
    import numpy as np
    from mri_rl.continuous_rl import digest
    rows=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],dtype='<f4');path=tmp_path/'actual_PRIVATE.bin';rows.tofile(path)
    response={'token_ids':[1,2],'generation_full_vocab_PRIVATE':{'path_PRIVATE':str(path),'sha256':digest(path),'rows_including_unreturned_lookahead':3,'returned_rows':2,'vocabulary':3}}
    result=q.compare_full_vocab(response,rows[:2],tmp_path/'full_PRIVATE.npy')
    assert result['max_full_vocab_normalized_logp_delta']==0 and result['max_TV']==0
    with pytest.raises(ValueError,match='PARTIAL_SELECTED_ONLY'):q.compare_full_vocab({'token_ids':[1]},rows[:1],tmp_path/'missing.npy')
    with pytest.raises(ValueError):q.compare_full_vocab(response,rows[:2]+np.array([0.,1.,0.]),tmp_path/'bad_PRIVATE.npy')
