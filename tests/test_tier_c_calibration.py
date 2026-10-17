import pytest
from mri_rl import tier_c_calibration as cal
from tests.test_two_task_training import ROWS
from tests.test_continuous_rl_driver import NumericBackend
from mri_rl.continuous_rl import digest


def test_binary_support_gives_zero_iqr_and_blocks_without_inventing_a_floor():
    s = cal.stats_from_accuracies({'mri_sequence': [1.0] * 20 + [0.0] * 8, 'mri_visit_correspondence': [0.0] * 48})
    assert s['mri_sequence']['iqr'] == 1.0 and not s['mri_sequence']['blocked'] and s['mri_sequence']['p50'] == 1.0
    assert s['mri_visit_correspondence']['blocked'] and s['mri_visit_correspondence']['iqr'] == 0
    with pytest.raises(ValueError): cal.calibration_for_arms(s)


def test_graded_support_yields_frozen_statistics_usable_by_the_grader():
    s = cal.stats_from_accuracies({'mri_sequence': [0, .25, .5, .75, 1.0, .5, .25, .75],
                                   'mri_visit_correspondence': [0, .25, .5, .75, 1.0, .5, .5, .25]})
    c = cal.calibration_for_arms(s)
    assert set(c) == {'mri_sequence', 'mri_visit_correspondence'} and all(v['iqr'] > 0 for v in c.values())
    with pytest.raises(ValueError): cal.stats_from_accuracies({'mri_sequence': [.5]})


class CalBackend(NumericBackend):
    def rollout(self, row, index, directory):
        raw = super().rollout(row, index, directory); raw['responses'] = raw['responses'] * 2   # 4 samples
        raw['image_sha256'] = digest(row['images'][0]); return raw


def test_run_is_durable_resumable_never_repeats_rows_and_reports_cost(tmp_path):
    seen = []
    class Counting(CalBackend):
        def rollout(self, row, index, directory):
            seen.append(row['id']); return super().rollout(row, index, directory)
    sub = ROWS[:3] + ROWS[28:31]
    out = cal.run_calibration(rows=sub, backend_factory=lambda: Counting(None, {}), authorize=lambda c: None, output=tmp_path / 'c',
                              group_size=4, input_tokens=lambda r: 213)
    assert out['rows'] == 6 and len(seen) == 6 and out['cost_tokens'] > 6 * 4 * 213
    again = cal.run_calibration(rows=sub, backend_factory=lambda: Counting(None, {}), authorize=lambda c: None, output=tmp_path / 'c',
                                group_size=4, input_tokens=lambda r: 213, resume=True)
    assert len(seen) == 6 and again['rows'] == 6                                  # nothing regenerated
    with pytest.raises(FileExistsError):
        cal.run_calibration(rows=sub, backend_factory=lambda: Counting(None, {}), authorize=lambda c: None, output=tmp_path / 'c',
                            group_size=4, input_tokens=lambda r: 213)
