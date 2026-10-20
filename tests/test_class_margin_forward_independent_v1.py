"""Independent synthetic execution of actual candidate scorer; no MLX imports."""
from types import SimpleNamespace
import numpy as np
from mri_rl.class_margin_v1 import _forward


def fixture():
    calls=[]
    def features(ids,pixels,**kwargs):
        length=ids.shape[-1]
        values={'inputs_embeds':np.ones((1,length,2),np.float32),
                'deepstack_visual_embeds':[np.ones((1,2),np.float32)],
                'position_ids':np.broadcast_to(np.arange(length),(3,1,length)),
                'rope_deltas':np.array([[0]])}
        return SimpleNamespace(to_dict=lambda:values)
    def language(ids,**kwargs):
        calls.append(ids.copy());length=ids.shape[-1]
        logits=(np.arange(1,length+1)[:,None]*np.arange(1,11)[None,:]/10).astype(np.float32)[None,:,:]
        return SimpleNamespace(logits=logits)
    def log_softmax(values,axis):
        centered=values-values.max(axis=axis,keepdims=True)
        return centered-np.log(np.exp(centered).sum(axis=axis,keepdims=True))
    backend=SimpleNamespace(mx=np,nn=SimpleNamespace(log_softmax=log_softmax),
        model=SimpleNamespace(get_input_embeddings=features,language_model=language))
    inputs={'input_ids':np.array([[11,12,13]]),'pixel_values':np.ones((1,2)),
            'attention_mask':np.ones((1,3)),'image_grid_thw':np.array([[1,2,2]])}
    return backend,inputs,calls,log_softmax


def test_actual_candidate_forward_selects_exact_causal_label_rows():
    backend,inputs,calls,normalizer=fixture();evidence=[]
    actual=_forward(backend,inputs,{'prefix_ids':[21,22]},[3,4],None,evidence)
    np.testing.assert_array_equal(calls[0],[[11,12,13,21,22,3]])
    logits=(np.array([5,6])[:,None]*np.arange(1,11)[None,:]/10).astype(np.float32)
    expected=normalizer(logits,axis=-1)[[0,1],[3,4]]
    np.testing.assert_array_equal(actual,expected)
    assert len(calls)==1 and evidence[0]['objective']=='LABEL_MEAN'


def test_actual_first_contrast_uses_one_shared_prefix_logit_vector():
    backend,inputs,calls,normalizer=fixture();evidence=[]
    actual=_forward(backend,inputs,{'prefix_ids':[21,22]},[],[3,7],evidence)
    np.testing.assert_array_equal(calls[0],[[11,12,13,21,22]])
    expected=normalizer((5*np.arange(1,11)[None,:]/10).astype(np.float32),axis=-1)[0,[3,7]]
    np.testing.assert_array_equal(actual,expected)
    assert len(calls)==1 and evidence[0]['objective']=='SHARED_PREFIX_FIRST_CONTRAST'
