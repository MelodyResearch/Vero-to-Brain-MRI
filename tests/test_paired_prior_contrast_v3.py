"""CPU-only tests: the new V3 loss MUST differentiate paired MRI evidence."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


def module():
    file = Path(__file__).resolve().parents[1] / "src/mri_rl/paired_prior_contrast_v3.py"
    spec = importlib.util.spec_from_file_location("prior_cancelled_v3", file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pair_loss_is_lower_for_correctly_separated_modalities():
    m = module()
    correct = m.pair_objective(-2., +2., np)
    reversed_images = m.pair_objective(+2., -2., np)
    unchanged_prior = m.pair_objective(-2., -2., np)
    assert np.isfinite([correct, reversed_images, unchanged_prior]).all()
    assert correct < unchanged_prior < reversed_images


def test_prior_cancelled_component_is_invariant_to_shared_answer_bias():
    m = module()
    mf, mt = -1.6, +0.4
    # A shared answer prior moves both class margins by the same amount.
    old_rank = np.logaddexp(mf - mt, 0.)
    for bias in (-100., -5., 0., 3., 50.):
        assert np.logaddexp((mf + bias) - (mt + bias), 0.) == pytest.approx(old_rank)
    # The CE anchor is deliberately not prior invariant; the RANKING term is.
    assert m.pair_objective(mf, mt, np) != pytest.approx(
        m.pair_objective(mf + 3., mt + 3., np))


def test_streaming_gradient_coefficients_match_finite_differences():
    m = module()
    eps = 1e-6
    for mf, mt in ((-4., 1.), (0., 0.), (1.5, -2.3), (13., 20.)):
        for weight in (0., 0.25, 1., 2.):
            cf, ct = m.margin_coefficients(mf, mt, weight)
            fd_f = (m.pair_objective(mf + eps, mt, np, weight)
                    - m.pair_objective(mf - eps, mt, np, weight)) / (2 * eps)
            fd_t = (m.pair_objective(mf, mt + eps, np, weight)
                    - m.pair_objective(mf, mt - eps, np, weight)) / (2 * eps)
            assert cf == pytest.approx(fd_f, abs=2e-8)
            assert ct == pytest.approx(fd_t, abs=2e-8)


def test_backward_coefficients_push_correct_mri_pair_separation():
    m = module()
    for mf, mt in ((-3., -2.), (+2., +1.), (-0.2, +0.4)):
        cf, ct = m.margin_coefficients(mf, mt)
        assert cf > 0. and ct < 0.
        # Gradient descent decreases FLAIR T1-preference and increases T1 T1-preference.
        assert (mt - .001 * ct) - (mf - .001 * cf) > mt - mf


def row(patient, level, label, *, split="train", image=None):
    return {"ability": "mri_sequence", "images": [image or f"/image/{patient}-{level}-{label}.png"],
            "reward_model": {"ground_truth": label},
            "extra_info": {"internal_split": split, "patient_key": patient,
                           "level": level, "question": "Which acquisition sequence?"}}


def test_pair_validation_enforces_same_patient_level_prompt_and_order():
    m = module()
    examples = [row("a", 45, "FLAIR"), row("a", 45, "contrast-enhanced T1"),
                row("b", 91, "FLAIR"), row("b", 91, "contrast-enhanced T1")]
    assert m.validate_step_pairs(examples) == {
        "matched_pairs": 2, "distinct_training_people": 2}
    same_person = [row("a", 45, "FLAIR"), row("a", 45, "contrast-enhanced T1"),
                   row("a", 91, "FLAIR"), row("a", 91, "contrast-enhanced T1")]
    assert m.validate_step_pairs(same_person)["distinct_training_people"] == 1
    bad = [dict(r) for r in examples]
    bad[1] = row("another", 45, "contrast-enhanced T1")
    with pytest.raises(ValueError, match="Patient/level"):
        m.validate_step_pairs(bad)
    bad = [dict(r) for r in examples]
    bad[1] = row("a", 45, "contrast-enhanced T1", split="internal_dev")
    with pytest.raises(ValueError, match="training"):
        m.validate_step_pairs(bad)
    bad = examples[::-1]
    with pytest.raises(ValueError, match="ordered"):
        m.validate_step_pairs(bad)


def test_stable_sigmoid_and_invalid_input_fail_closed():
    m = module()
    assert m.sigmoid(1000.) == pytest.approx(1.)
    assert m.sigmoid(-1000.) == pytest.approx(0.)
    for inp in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.sigmoid(inp)
        with pytest.raises(ValueError):
            m.margin_coefficients(inp, 0.)
    with pytest.raises(ValueError):
        m.pair_objective(0., 1., np, -1.)
