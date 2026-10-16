"""CPU-only checkpoint integrity and continuous-Adam numerical regression."""
import json
import random

import numpy as np
import pytest

from mri_rl.optimizer_checkpoint import (
    load_optimizer_checkpoint, pack_tree, save_optimizer_checkpoint, unpack_tree,
)

BINDINGS = {"adapter_sha256": "a" * 64, "optimizer": {"name": "Adam", "lr": 1e-5,
            "betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False},
            "backend": "numpy-cpu-fixture", "contract": "test-only",
            "trainable_signature": {"layers.0.weight": {"shape": [2], "dtype": "float64"}}}


def write_arrays(path, arrays):
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def initial():
    return {"step": np.array(0, dtype=np.uint64), "learning_rate": np.array(1e-5),
            "layers": [{"weight": {"m": np.zeros(2), "v": np.zeros(2)}}]}


def adam(parameter, gradient, state):
    # Installed MLX Adam recurrence with bias_correction=False, evaluated on CPU.
    state["step"] += 1
    moments = state["layers"][0]["weight"]
    moments["m"] = 0.9 * moments["m"] + 0.1 * gradient
    moments["v"] = 0.999 * moments["v"] + 0.001 * gradient ** 2
    return parameter - state["learning_rate"] * moments["m"] / (np.sqrt(moments["v"]) + 1e-8)


def save(path, state, **kwargs):
    return save_optimizer_checkpoint(path, optimizer_state=state,
        rng_state=kwargs.pop("rng_state", {"python": random.Random(3).getstate()}),
        bindings=BINDINGS, completed_groups=int(state["step"]) + 1,
        optimizer_updates=int(state["step"]), save_arrays=kwargs.pop("save_arrays", write_arrays), **kwargs)


def load(path, **kwargs):
    return load_optimizer_checkpoint(path, expected_bindings=kwargs.pop("expected_bindings", BINDINGS),
                                     load_arrays=read_arrays, **kwargs)


def test_continuous_and_resumed_match_and_reset_does_not(tmp_path):
    state = initial()
    parameter = adam(np.ones(2), np.array([1., -2.]), state)
    save(tmp_path / "checkpoint", state)
    restored, _, manifest = load(tmp_path / "checkpoint")
    gradients = [np.array([3., -1.]), np.array([-0.5, 2.])]
    resumed = parameter.copy()
    reset = parameter.copy()
    for gradient in gradients:
        parameter = adam(parameter, gradient, state)
        resumed = adam(resumed, gradient, restored)
        reset = adam(reset, gradient, initial())
    np.testing.assert_array_equal(parameter, resumed)
    np.testing.assert_array_equal(state["layers"][0]["weight"]["m"], restored["layers"][0]["weight"]["m"])
    assert int(restored["step"]) == 3
    assert manifest["completed_groups"] == 2  # includes an attempted inactive group
    assert np.max(abs(parameter - reset)) > 1e-5


def test_roundtrip_nested_rng_and_native_tensor_dtypes(tmp_path):
    rng = random.Random(33)
    rng.random()
    save(tmp_path / "checkpoint", initial(), rng_state={"python": rng.getstate(),
            "numpy": np.random.default_rng(9).bit_generator.state,
            "mlx_fixture": [np.array([42, 99], dtype=np.uint32)]})
    state, restored, _ = load(tmp_path / "checkpoint")
    other = random.Random()
    other.setstate(restored["python"])
    assert other.random() == rng.random()
    assert isinstance(restored["python"], tuple)
    assert isinstance(state["layers"], list)
    assert state["step"].dtype == np.uint64
    assert restored["mlx_fixture"][0].dtype == np.uint32


def test_wrong_binding_rejected_before_array_decoder(tmp_path):
    save(tmp_path / "checkpoint", initial())
    with pytest.raises(ValueError, match="binding"):
        load(tmp_path / "checkpoint", expected_bindings={**BINDINGS, "adapter_sha256": "b" * 64})


def test_corruption_rejected(tmp_path):
    path = save(tmp_path / "checkpoint", initial())
    with (path / "state.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load(path)


def test_never_overwrites_and_failed_write_is_not_published(tmp_path):
    path = save(tmp_path / "checkpoint", initial())
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save(path, initial())
    assert before == (path / "manifest.json").read_bytes()
    def fail(path, arrays):
        path.write_bytes(b"partial")
        raise OSError("simulated disk failure")
    with pytest.raises(OSError):
        save(tmp_path / "failed", initial(), save_arrays=fail)
    assert not (tmp_path / "failed").exists()
    assert list(tmp_path.glob(".failed-*")) == []


def test_step_and_missing_moments_rejected(tmp_path):
    state = initial()
    state["step"] = np.array(1, dtype=np.uint64)
    del state["layers"][0]["weight"]["v"]
    with pytest.raises(ValueError, match="Incomplete Adam"):
        save(tmp_path / "bad", state)
    with pytest.raises(ValueError, match="step"):
        save_optimizer_checkpoint(tmp_path / "bad", optimizer_state=initial(),
            rng_state={"python": random.getstate()}, bindings=BINDINGS,
            completed_groups=1, optimizer_updates=1, save_arrays=write_arrays)


def test_shape_dtype_and_extra_tensor_validation():
    layout, arrays = pack_tree({"tensor": np.zeros((2,), dtype=np.float32)})
    with pytest.raises(ValueError, match="shape/dtype"):
        unpack_tree(layout, {"tensor_00000000": np.zeros(2, dtype=np.float64)})
    with pytest.raises(ValueError, match="Unreferenced"):
        unpack_tree(layout, {**arrays, "extra": np.zeros(1)})


def test_tampered_update_count_rejected(tmp_path):
    path = save(tmp_path / "checkpoint", initial())
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["optimizer_updates"] = 1
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="step"):
        load(path)


@pytest.mark.skipif((__import__("sys").platform == "darwin" and __import__("os").environ.get("MRI_REAUTHORIZED_TINY_CPU") != "1") or __import__("os").environ.get("MRI_TEST_MLX_CPU") != "1",
                    reason="Post-panic macOS MLX import prohibited; native verification requires separately authorized remote runtime")
def test_native_mlx_cpu_optimizer_and_rng_resume(tmp_path):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.set_default_device(mx.cpu)
    params = {"weight": mx.array([1., 2.]),
              "bf_weight": mx.array([1., 2.], dtype=mx.bfloat16)}
    gradient = {key: mx.array([1., -2.], dtype=value.dtype) for key, value in params.items()}
    optimizer = optim.Adam(learning_rate=1e-2, bias_correction=False)
    params = optimizer.apply_gradients(gradient, params)
    mx.eval(params, optimizer.state)
    mx.random.seed(117)
    mx.eval(mx.random.uniform(shape=(3,)))
    rng = [mx.array(x) for x in mx.random.state]
    mx.eval(rng)
    path = save_optimizer_checkpoint(tmp_path / "native", optimizer_state=optimizer.state,
        rng_state={"mlx": rng}, bindings={"backend": mx.__version__, "test_only": True, "trainable_signature": {k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in params.items()}},
        completed_groups=1, optimizer_updates=1,
        save_arrays=lambda p, arrays: mx.save_safetensors(str(p), arrays))
    state, restored_rng, _ = load_optimizer_checkpoint(path,
        expected_bindings={"backend": mx.__version__, "test_only": True, "trainable_signature": {k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in params.items()}},
        load_arrays=lambda p: mx.load(str(p)))
    resumed = optim.Adam(learning_rate=1e-2, bias_correction=False)
    resumed.state = state
    gradient = {key: mx.array([3., -1.], dtype=value.dtype) for key, value in params.items()}
    uninterrupted = optimizer.apply_gradients(gradient, params)
    restarted = resumed.apply_gradients(gradient, params)
    mx.eval(uninterrupted, restarted, optimizer.state, resumed.state)
    assert int(resumed.step.item()) == 2
    for (name, actual), (restored_name, expected) in zip(
            tree_flatten(optimizer.state), tree_flatten(resumed.state)):
        assert name == restored_name
        assert actual.dtype == expected.dtype
        assert bool(mx.array_equal(actual, expected).item())
    for name in uninterrupted:
        assert bool(mx.array_equal(uninterrupted[name], restarted[name]).item())
        assert not bool(mx.array_equal(params[name], restarted[name]).item())
    next_draw = mx.random.uniform(shape=(7,))
    mx.eval(next_draw)
    for index, value in enumerate(restored_rng["mlx"]):
        mx.random.state[index][:] = value
    replay = mx.random.uniform(shape=(7,))
    mx.eval(replay)
    assert bool(mx.array_equal(next_draw, replay).item())


def test_seed_only_rng_and_partial_moment_inventory_rejected(tmp_path):
    with pytest.raises(ValueError,match='RNG'):
        save(tmp_path/'seed',initial(),rng_state={'seed':7})
    state=initial();state['step']=np.array(1,dtype=np.uint64)
    binding={**BINDINGS,'trainable_signature':{'layers.0.weight':{'shape':[2],'dtype':'float64'},'missing':{'shape':[2],'dtype':'float64'}}}
    with pytest.raises(ValueError,match='inventory'):
        save_optimizer_checkpoint(tmp_path/'partial',optimizer_state=state,rng_state={'python':random.getstate()},bindings=binding,completed_groups=1,optimizer_updates=1,save_arrays=write_arrays)


def test_racing_destination_is_never_replaced(tmp_path):
    target=tmp_path/'race'
    def race(path,arrays):
        write_arrays(path,arrays)
        target.mkdir()
        # Even an empty racing directory must not be replaced.
    with pytest.raises(FileExistsError):save(target,initial(),save_arrays=race)
    assert target.exists() and not list(target.iterdir())


def test_invalid_numpy_rng_state_rejected(tmp_path):
    with pytest.raises(ValueError,match='NumPy'):
        save(tmp_path/'rng',initial(),rng_state={'numpy':{'bit_generator':'PCG64','state':7}})
