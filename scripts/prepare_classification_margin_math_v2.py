"""CPU replay of v1/v2 auxiliary math at the authenticated saved Adam60 point."""
import argparse,json,hashlib,math
from pathlib import Path
from mri_rl.paired_classification_signal_v2 import classification_loss,score_derivatives,margins
from mri_rl.paired_interaction_signal_v1 import pair_loss,score_derivatives as v1_derivatives

def main():
    p=argparse.ArgumentParser();p.add_argument('--saved-complete',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.saved_complete);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if sha!='252d44fb879ef8926b4dda301168265be6702641d37c44554fe5f306a6f1a911':raise ValueError('Use exact authenticated previous TRAIN diagnostic only')
    data=json.loads(source.read_text());lookup={c['context']:c for c in data['contexts']}
    scores=[[p['mean'] for p in lookup[k]['per_label']] for k in ('SEQUENCE_OPPOSITE_SAME_PERSON_Z045','SEQUENCE_REAL')]
    for k in ('SEQUENCE_REAL','SEQUENCE_OPPOSITE_SAME_PERSON_Z045'):
        if [p['label'] for p in lookup[k]['per_label']]!=['FLAIR','contrast-enhanced T1']:raise ValueError('Canonical label order changed')
    m=margins(scores);g=score_derivatives(scores);old=v1_derivatives(scores)
    r={'schema':'classification-margin-v2-saved-CPU-math-v1','status':'TESTED_CPU_NOT_NATIVE_GRADIENT_QUALIFIED',
        'saved_reference_complete_sha256':sha,'correct_margins':{'FLAIR':m[0],'T1':m[1]},
        'v1_interaction_loss':pair_loss(scores),'v2_perclass_margin_loss':classification_loss(scores),
        'v1_score_derivatives':old,'v2_score_derivatives':g,
        'v1_common_additive_FLAIR_score_bias_derivative':old[0][0]+old[1][0],
        'v2_common_additive_FLAIR_score_bias_derivative':g[0][0]+g[1][0],
        'score_space_correct_margin_changes_along_v2_descent':{'FLAIR':-g[0][0]+g[0][1],'T1':-g[1][1]+g[1][0]},
        'meaning':'v1 can lower loss with a wrong absolute T1 margin; v2 directly penalizes each wrong class margin and responds to shared FLAIR score bias.',
        'boundaries':['These are score-space derivatives, not measured model-parameter or Adam directions.',
            'An image-independent model can reduce v2 through prior neutralization; that alone does not establish image discrimination.',
            'Shared model Jacobians and retained Adam moments can harm one class despite v2 score-space signs.',
            'Mean canonical logp is length-normalized by declaration; sum and first readouts remain separately reported.',
            'Old v1 never trained; no empirical accuracy superiority over v1 is claimed.'],
        'native_model_loads_this_CPU_replay':0,'training_runs':0,'optimizer_updates':0,'training_authorized':False}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'L1':r['v1_interaction_loss'],'L2':r['v2_perclass_margin_loss'],'F_bias_derivative':r['v2_common_additive_FLAIR_score_bias_derivative'],'model_loads':0}))
if __name__=='__main__':main()
