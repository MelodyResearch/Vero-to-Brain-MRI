"""Prespecified lightweight C1 real-pixel shortcut probe; no VLM/MLX imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def lopo_predictions(rows,features):
    result=[]
    for i,row in enumerate(rows):
        candidates=[j for j,r in enumerate(rows) if r['patient']!=row['patient']]
        if not candidates:raise ValueError('No other-patient neighbor')
        nearest=min(candidates,key=lambda j:(float(np.sum((features[i]-features[j])**2)),rows[j]['record_id']))
        result.append(rows[nearest]['label'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--subset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    private=Path(__file__).resolve().parents[1]/'local_data'
    if not a.output.resolve().is_relative_to(private) or a.output.exists():raise ValueError('Newprivateoutputrequired')
    subset=json.loads(a.subset.read_text());records={r['id']:r for r in map(json.loads,(a.root/'vero_training_v1/train.verl.jsonl').read_text().splitlines())}
    result={'status':'TESTED_CPU','method':'leave-one-patient-out1NN; Euclidean distance; record-ID ties; no fitting/tuning',
            'source_sha256':sha(__file__),'subset_sha256':sha(a.subset),'numpy_version':np.__version__,
            'features':'raw16x16 bilinear grayscale pixels /255; separately16-bin fullimage grayscale histogram normalized toprobability',
            'interpretation':'Exploratory shortcut diagnostics on training-selected cohort. Not independentgeneralization or lesionunderstanding.', 'categories':{}}
    for category,solution in subset['categories'].items():
        rows=solution['selected_PRIVATE'];thumb=[];hist=[]
        for row in rows:
            path=Path(records[row['record_id']]['images'][0])
            if sha(path)!=row['image_sha256']:raise ValueError('Imagehashchanged')
            with Image.open(path) as image:
                gray=image.convert('L');thumb.append(np.asarray(gray.resize((16,16),Image.Resampling.BILINEAR),dtype=np.float64).ravel()/255)
                values=np.asarray(gray);h=np.histogram(values,bins=16,range=(0,256))[0];hist.append(h/h.sum())
        methods={}
        for name,features in [('thumbnail16',np.array(thumb)),('histogram16',np.array(hist))]:
            pred=lopo_predictions(rows,features);methods[name]={'correct':sum(p==r['label'] for p,r in zip(pred,rows)),
                'n':len(rows),'confusion':{f'{gold}->{guess}':sum(r['label']==gold and p==guess for r,p in zip(rows,pred)) for gold in ('no','yes') for guess in ('no','yes')},
                'predictions_PRIVATE':[{'record_id':r['record_id'],'patient':r['patient'],'gold':r['label'],'prediction':p} for r,p in zip(rows,pred)]}
        result['categories'][category]=methods
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for methods in result['categories'].values():
        for values in methods.values():values.pop('predictions_PRIVATE')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
