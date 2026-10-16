"""Backend-neutral, immutable checkpoints at completed optimizer/group boundaries.

No MLX import or dtype conversion: supply mx.save_safetensors/mx.load wrappers
for native tensors. CPU tests use NumPy NPZ wrappers. This does not checkpoint
model weights, rollouts, partial gradients, callable LR schedules, or caches.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

SCHEMA = "mri-optimizer-boundary-v1"


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def pack_tree(tree: Any) -> tuple[dict, dict]:
    """Keep dict/list/tuple structure, scalars and native array dtype/shape."""
    arrays = {}

    def visit(value):
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("State dictionaries require string keys")
            return {"kind": "dict", "items": [[k, visit(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [visit(v) for v in value]}
        if value is None or type(value) in (str, bool, int, float):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite scalar state")
            return {"kind": "scalar", "value": value}
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"Unsupported state leaf: {type(value).__name__}")
        key = f"tensor_{len(arrays):08d}"
        arrays[key] = value
        return {"kind": "tensor", "key": key, "shape": list(value.shape),
                "dtype": str(value.dtype)}

    return visit(tree), arrays


def unpack_tree(layout: dict, arrays: dict) -> Any:
    used = set()

    def visit(node):
        kind = node["kind"]
        if kind == "tensor":
            key = node["key"]
            if key in used or key not in arrays:
                raise ValueError("Missing or duplicate tensor reference")
            used.add(key)
            value = arrays[key]
            if list(value.shape) != node["shape"] or str(value.dtype) != node["dtype"]:
                raise ValueError("Tensor shape/dtype mismatch")
            return value
        if kind == "dict":
            keys = [key for key, _ in node["items"]]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate dictionary key")
            return {key: visit(value) for key, value in node["items"]}
        if kind in ("list", "tuple"):
            result = [visit(value) for value in node["items"]]
            return tuple(result) if kind == "tuple" else result
        if kind == "scalar":
            value = node["value"]
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("Invalid scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite scalar")
            return value
        raise ValueError("Unknown tree node")

    result = visit(layout)
    if used != set(arrays):
        raise ValueError("Unreferenced checkpoint tensors")
    return result


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_inventory(state, bindings):
    expected = bindings.get("trainable_signature")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Complete trainable moment inventory is required")
    actual = {}
    def visit(node, path=""):
        if isinstance(node, dict):
            if "m" in node and "v" in node:
                actual[path] = {"shape": list(node["m"].shape), "dtype": str(node["m"].dtype)}
            else:
                for key,value in node.items(): visit(value, f"{path}.{key}".lstrip("."))
        elif isinstance(node, (list,tuple)):
            for index,value in enumerate(node): visit(value, f"{path}.{index}")
    visit(state)
    # Zero-update optimizers may be wholly uninitialized; partial state is never accepted.
    step=state["step"].item() if hasattr(state["step"],"item") else state["step"]
    if (actual or step) and actual != expected:
        raise ValueError("Adam moment inventory differs from exact trainable signature")


def _validate(optimizer_state, rng_state, updates, groups):
    if type(updates) is not int or type(groups) is not int or not 0 <= updates <= groups:
        raise ValueError("Require nonnegative integer updates <= completed groups")
    step = optimizer_state.get("step")
    if step is None or tuple(getattr(step, "shape", ())) != ():
        raise ValueError("Optimizer step must be scalar")
    actual = step.item() if hasattr(step, "item") else step
    if type(actual) is not int or actual != updates:
        raise ValueError("Optimizer step does not match effective update count")
    if "learning_rate" not in optimizer_state:
        raise ValueError("Missing optimizer learning rate")
    if not isinstance(rng_state, dict) or not rng_state:
        raise ValueError("Explicit RNG state required; seeds alone are insufficient")
    if not set(rng_state).issubset({"python", "numpy", "mlx", "mlx_fixture"}):
        raise ValueError("RNG state must contain concrete supported generator states, not seeds")
    if "python" in rng_state:
        try: random.Random().setstate(rng_state["python"])
        except (TypeError, ValueError): raise ValueError("Invalid Python RNG state") from None
    if "numpy" in rng_state:
        value=rng_state["numpy"]
        if not isinstance(value,dict) or value.get("bit_generator") not in {"PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"} or "state" not in value:
            raise ValueError("Invalid NumPy generator RNG state")
        import numpy as np
        try:
            generator = getattr(np.random, value["bit_generator"])()
            generator.state = value
        except (TypeError, ValueError, KeyError):
            raise ValueError("Invalid NumPy generator RNG state") from None
    for key in ("mlx", "mlx_fixture"):
        if key in rng_state and (not isinstance(rng_state[key],(list,tuple)) or not rng_state[key] or
            any(list(getattr(v,"shape",[])) != [2] or "uint32" not in str(getattr(v,"dtype","")) for v in rng_state[key])):
            raise ValueError("Invalid MLX RNG array state")
    moments = []

    def visit(value):
        if isinstance(value, dict):
            if "m" in value or "v" in value:
                if "m" not in value or "v" not in value:
                    raise ValueError("Incomplete Adam moments")
                m, v = value["m"], value["v"]
                if (not hasattr(m, "shape") or not hasattr(v, "shape")
                        or m.shape != v.shape or str(m.dtype) != str(v.dtype)):
                    raise ValueError("Adam moment shape/dtype mismatch")
                moments.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(optimizer_state)
    if updates and not moments:
        raise ValueError("Updated Adam checkpoint requires moments")


def save_optimizer_checkpoint(
    destination: str | Path, *, optimizer_state: dict, rng_state: dict,
    bindings: dict, completed_groups: int, optimizer_updates: int,
    save_arrays: Callable[[Path, dict], None],
) -> Path:
    """Atomically publish a new directory; never overwrite an existing checkpoint.

    `bindings` must include the exact adapter SHA256, optimizer hyperparameters,
    model/trainable signature, contract/source identities and backend version.
    All used RNGs belong in rng_state (MLX, Python and NumPy as applicable).
    The caller must evaluate state and finish the group before invoking this.
    """
    _validate(optimizer_state, rng_state, optimizer_updates, completed_groups)
    if not bindings:
        raise ValueError("Nonempty exact resume bindings required")
    # Serialize before creating files; disallow arbitrary objects/NaN in bindings.
    bindings = json.loads(_json(bindings))
    _validate_inventory(optimizer_state, bindings)
    layout, arrays = pack_tree({"optimizer": optimizer_state, "rng": rng_state})
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        tensor_file = temporary / "state.safetensors"
        save_arrays(tensor_file, arrays)
        with tensor_file.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest = {"schema": SCHEMA, "boundary": "completed_group_no_pending_gradients",
                    "bindings": bindings, "completed_groups": completed_groups,
                    "optimizer_updates": optimizer_updates, "layout": layout,
                    "tensors_sha256": _hash(tensor_file)}
        with (temporary / "manifest.json").open("x") as handle:
            handle.write(_json(manifest) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(temporary)
        # Atomic reservation refuses even a racing empty directory. A reader only
        # accepts manifest.json, published last as the completion marker.
        destination.mkdir()
        os.replace(tensor_file, destination / "state.safetensors")
        _fsync_dir(destination)
        os.replace(temporary / "manifest.json", destination / "manifest.json")
        _fsync_dir(destination)
        _fsync_dir(destination.parent)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_optimizer_checkpoint(
    source: str | Path, *, expected_bindings: dict,
    load_arrays: Callable[[Path], dict],
) -> tuple[dict, dict, dict]:
    """Validate identity/integrity before decoding; return optimizer, RNG, manifest.

    Callers must also verify actual adapter bytes against expected_bindings and
    restore RNG after model setup, immediately before the next stochastic action.
    This function does not mutate an optimizer or RNG automatically.
    """
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["schema"] != SCHEMA
            or manifest["boundary"] != "completed_group_no_pending_gradients"):
        raise ValueError("Unsupported checkpoint schema/boundary")
    if _json(manifest["bindings"]) != _json(expected_bindings):
        raise ValueError("Resume binding mismatch")
    tensor_file = source / "state.safetensors"
    if _hash(tensor_file) != manifest["tensors_sha256"]:
        raise ValueError("Optimizer tensor digest mismatch")
    tree = unpack_tree(manifest["layout"], load_arrays(tensor_file))
    _validate(tree["optimizer"], tree["rng"], manifest["optimizer_updates"],
              manifest["completed_groups"])
    _validate_inventory(tree["optimizer"], expected_bindings)
    return tree["optimizer"], tree["rng"], manifest
