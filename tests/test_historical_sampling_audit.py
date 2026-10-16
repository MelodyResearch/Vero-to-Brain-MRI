import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('sampling',Path(__file__).resolve().parents[1]/'scripts/audit_historical_sft_sampling.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_distribution_counts_repeats_and_mask_quantities():
    rows=[{'category':'presence','label':'yes','level':45,'patient':'a','image_sha256':'same','target':'yes','mask_area':2},
          {'category':'presence','label':'no','level':45,'patient':'b','image_sha256':'same','target':'no','mask_area':0}]
    out=m.summarize(rows)['presence']
    assert out['labels']=={'yes':1,'no':1}
    assert out['positive_fraction']==0.5
    assert out['image_reuse_excess']==1
    assert out['mask_area']['positive_min']==2
    assert out['distinct_patients']==2
