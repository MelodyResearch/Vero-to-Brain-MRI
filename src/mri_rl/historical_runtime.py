"""Explicit, exact historical native-engine source dependency.

CPU Git export/verification only. No network, private data, model or MLX import.
Current source packaging is not a new native-qualification result. Executing the
final evaluator still requires its original inputs, fresh scope and human grant.
"""
from __future__ import annotations
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

COMMIT = "1e0d7be1bb099b42e717eb4eb081facc12607ed3"
SPEC_SHA256 = "418269eb611e9282c7fe8317077ee6716c3fe1cd1cc9fdfce7b31b53168ce215"
ROOT = Path(__file__).resolve().parents[2]


def source_spec():
    spec = json.loads((ROOT / "results/SIMPLIFICATION_MANIFEST.json").read_text())["native_runtime"]
    digest = hashlib.sha256(json.dumps(spec["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if spec["commit"] != COMMIT or digest != SPEC_SHA256:
        raise ValueError("Historical source specification differs from pinned commit/files")
    return spec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_source(destination, *, repository=ROOT):
    """Export only reviewed Python blobs to a NEW directory, without importing them."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Source destination already exists; never overwrite")
    spec = source_spec()
    command = ["git", "-C", str(repository), "archive", COMMIT, "--", *spec["files"]]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise ValueError("Pinned historical Git objects unavailable; no automatic fetch or replacement")
    blobs = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in spec["files"]:
                raise ValueError("Unexpected historical archive member")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != spec["files"][member.name]:
                raise ValueError("Historical Git source SHA differs")
            blobs[member.name] = data
    if set(blobs) != set(spec["files"]):
        raise ValueError("Incomplete historical source export")
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in blobs.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return verify_source(destination)


def verify_source(root, scope=None):
    root = Path(root)
    spec = source_spec()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Historical source directory unavailable or symbolic")
    root = root.resolve()
    # Every object must be a plain directory/file. No alternate source or hidden
    # payload is accepted, including native extensions and import-cache files.
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symbolic historical source path")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("Special historical source object")
    if actual != set(spec["files"]):
        raise ValueError("Missing or extra historical runtime source")
    pins = scope.get("file_sha256", {}) if scope is not None else None
    for name, expected in spec["files"].items():
        path = root / name
        if _sha(path) != expected:
            raise ValueError("Historical source SHA mismatch: " + name)
        if pins is not None and pins.get(str(path)) != expected:
            raise ValueError("Exact scope must pin every actual historical source path")
    return dict(spec["files"])


class FrozenRuntime:
    """Scoped package search path for the verified, unchanged historical engine."""
    def __init__(self, root):
        import mri_rl
        self.root = Path(root).resolve()
        self.package = self.root / "src/mri_rl"
        self.allowed = {Path(p).stem: h for p, h in source_spec()["files"].items()}
        self.old_path = list(mri_rl.__path__)
        self.old_modules = set(sys.modules)
        self.closed = False
        self.old_bytecode = sys.dont_write_bytecode
        # Never silently reuse a different checkout's cached engine modules.
        for name, module in list(sys.modules.items()):
            if name.startswith("mri_rl.") and name.split(".")[-1] in self.allowed:
                stem = name.split(".")[-1]
                path = Path(getattr(module, "__file__", "")).resolve()
                if path not in (ROOT / "src/mri_rl" / (stem + ".py"), self.package / (stem + ".py")) or _sha(path) != self.allowed[stem]:
                    raise ValueError("Different historical module already imported: " + name)
        sys.dont_write_bytecode = True
        mri_rl.__path__ = [*self.old_path, str(self.package)]

    def module(self, name):
        if self.closed or name not in self.allowed or name == "__init__":
            raise ValueError("Module is outside the reviewed historical runtime")
        return importlib.import_module("mri_rl." + name)

    def close(self):
        import mri_rl
        if self.closed:
            return
        mri_rl.__path__ = self.old_path
        for name in set(sys.modules) - self.old_modules:
            if name.startswith("mri_rl."):
                module = sys.modules.pop(name)
                if getattr(mri_rl, name.split(".")[-1], None) is module:
                    delattr(mri_rl, name.split(".")[-1])
        sys.dont_write_bytecode = self.old_bytecode
        self.closed = True


def activate(scope):
    """Called only after final execution authorization, or explicit CPU fixtures."""
    root = Path(scope["historical_runtime_root"])
    verify_source(root, scope)
    return FrozenRuntime(root)
