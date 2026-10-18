"""TRAIN-only sequence qualification utilities; no neural model dependency."""
from __future__ import annotations
from collections import Counter,defaultdict
import random
import re
import numpy as np
from PIL import Image


def require(condition,message):
    if not condition:raise ValueError(message)


def blind_pairs(rows,seed):
    groups=defaultdict(list)
    for row in rows:
        e=row['extra_info'];require(row['ability']=='mri_sequence' and e['internal_split']=='train','Only TRAIN sequence accepted')
        m=re.search(r'_z(045|091|136)_(flair|t1ce)_sequence$',row['id']);require(m is not None,'Unexpected source sequence ID')
        require(row['reward_model']['ground_truth']=={'flair':'FLAIR','t1ce':'contrast-enhanced T1'}[m[2]],'Source/target conflict')
        groups[e['patient_key'],int(m[1])].append(row)
    require(len({r['id'] for r in rows})==len(rows),'Duplicate source IDs')
    require(all(Counter(r['reward_model']['ground_truth'] for r in g)=={'FLAIR':1,'contrast-enhanced T1':1} for g in groups.values()),'Missing or duplicated pair truth')
    rng=random.Random(seed);keys=sorted(groups);rng.shuffle(keys);out=[]
    for key in keys:
        pair=sorted(groups[key],key=lambda r:r['id']);rng.shuffle(pair)
        out.append({'case_id':f'{rng.getrandbits(64):016x}','left':pair[0],'right':pair[1]})
    require(len({r['case_id'] for r in out})==len(out),'Random case ID collision')
    return out


def source_sequence_render(volume,level,size=(364,436)):
    require(volume.ndim==3 and 0<=level<volume.shape[2] and np.isfinite(volume).all(),'Invalid source volume/level')
    nonzero=volume[volume!=0][::16];require(nonzero.size>1,'Insufficient source foreground')
    lo,hi=np.percentile(nonzero,[.5,99.5]);require(hi>lo,'Degenerate normalization')
    plane=np.rot90(volume[:,:,level]);pixels=np.uint8(np.round(np.clip((plane-lo)/(hi-lo),0,1)*255))
    return Image.fromarray(pixels).resize(size,Image.Resampling.BILINEAR).convert('RGB'),[float(lo),float(hi)]


def lopo_nearest_neighbor(features,labels,patients):
    """Fixed Euclidean retrieval against other TRAIN people; no fitted parameters.

    Inputs already use fixed normalization. No scaler, centroid, classifier head,
    optimizer or hyperparameter search is fitted here.
    """
    x=np.asarray(features,dtype=float);y=np.asarray(labels,dtype=int);p=np.asarray(patients)
    require(x.ndim==2 and len(x)==len(y)==len(p) and np.isfinite(x).all(),'Invalid CPU diagnostic arrays')
    require(set(y)=={0,1} and len(set(p))>=2,'Both labels/multiple TRAIN patients required')
    predicted=np.zeros(len(y),dtype=int);neighbors=[]
    for i in range(len(y)):
        eligible=np.flatnonzero(p!=p[i]);require(len(eligible)>0,'No other-person reference')
        j=int(eligible[np.argmin(((x[eligible]-x[i])**2).sum(axis=1))])
        predicted[i]=y[j];neighbors.append(j)
    folds=[]
    for person in sorted(set(p)):
        test=p==person;fit=~test;require(set(y[fit])=={0,1},'Training fold lacks a class')
        folds.append({'patient_PRIVATE':str(person),'test_rows':int(test.sum()),'fit_rows':int(fit.sum()),'fit_people':int(len(set(p[fit]))),'correct':int((predicted[test]==y[test]).sum())})
    return {'rows':len(y),'correct':int((predicted==y).sum()),'accuracy':float((predicted==y).mean()),
        'patient_macro':float(np.mean([r['correct']/r['test_rows'] for r in folds])),
        'confusion':{f'{a}->{b}':int(((y==a)&(predicted==b)).sum()) for a in (0,1) for b in (0,1)},
        'same_patient_fit_overlap':0,'folds_PRIVATE':folds,'prediction_PRIVATE':predicted.tolist(),'neighbor_indices_PRIVATE':neighbors,
        'fitted_parameters':0,'calibration':'No fitting; fixed normalized feature Euclidean1NN, reference-index tie break; all same-patient rows excluded'}


def unblind_judgments(pairs,judgments):
    by_case={r['case_id']:r for r in judgments}
    require(len(by_case)==len(judgments)==len(pairs) and set(by_case)=={r['case_id'] for r in pairs},'Frozen blind judgment inventory differs')
    translate={'FLAIR':'FLAIR','T1ce':'contrast-enhanced T1','contrast-enhanced T1':'contrast-enhanced T1','ABSTAIN':None}
    details=[]
    for pair in pairs:
        judgment=by_case[pair['case_id']];correct=[];abstained=[]
        for side in ('left','right'):
            guess=judgment[f'{side}_modality_guess'];require(guess in translate,'Unknown blind guess')
            truth=pair['sides_PRIVATE'][side]['source_row_PRIVATE']['reward_model']['ground_truth']
            abstained.append(translate[guess] is None);correct.append(translate[guess]==truth)
        require(all(abstained) or not any(abstained),'Partial pair abstention needs explicit handling')
        details.append({'case_id':pair['case_id'],'answered':not any(abstained),'both_sides_correct':all(correct),
            'side_correct':sum(correct),'coverage':judgment['coverage'],'confidence':judgment['confidence'],
            'eligible_visual_review':judgment['eligible_visual_review']})
    answered=[r for r in details if r['answered']]
    return {'pairs':len(pairs),'answered_pairs':len(answered),'abstained_pairs':len(pairs)-len(answered),
        'correct_answered_pairs':sum(r['both_sides_correct'] for r in answered),'incorrect_answered_pairs':sum(not r['both_sides_correct'] for r in answered),
        'correct_answered_sides':sum(r['side_correct'] for r in answered),'answered_sides':2*len(answered),
        'coverage_counts':dict(Counter(r['coverage'] for r in details)),'confidence_counts':dict(Counter(r['confidence'] for r in details)),
        'pair_details_PRIVATE':details,'clinical_scientific_admission':False,
        'scope':'Nonexpert AI paired discrimination with one-of-each design disclosed; not independent isolated-image accuracy or clinical adjudication'}
