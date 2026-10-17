"""Token accounting must count actual expanded image IDs, not add them twice."""
import importlib.util
from pathlib import Path
import pytest


def load():
    path=Path(__file__).resolve().parents[1]/'scripts/measure_native_training_tokens.py'
    assert path.exists(),'Missing processor-only native token measurement script'
    spec=importlib.util.spec_from_file_location('native_training_tokens_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_actual_expanded_token_counts_grid_consistency_and_no_double_count():
    m=load();result=m.inspect_input_tokens([[2,9,9,4]],[[1,2,4]],9,2)
    assert result=={'input_tokens':4,'visual_tokens':2,'text_and_boundary_tokens':2,'image_grid_thw':[[1,2,4]]}
    with pytest.raises(ValueError,match='grid'):m.inspect_input_tokens([[2,9,4]],[[1,2,4]],9,2)
    with pytest.raises(ValueError,match='single'):m.inspect_input_tokens([[2,9,4],[2,9,4]],[[1,2,2]],9,2)
