"""CPU mathematical counterexamples; no native/model/patient-data imports."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def logps(theta, slopes):
    probabilities = (.2, .3, .5)  # correct, wrong, unobserved other
    logits = [math.log(p)+theta*k for p,k in zip(probabilities, slopes)]
    maximum = max(logits)
    denominator = maximum+math.log(sum(math.exp(v-maximum) for v in logits))
    return [v-denominator for v in logits]


def counterexample():
    """Both are valid normalized categorical models, not free logp vectors."""
    probabilities = (.2, .3, .5)
    group_gradient = 1.
    examples = []
    for wrong_gradient in (0., 2.):
        correct_gradient = 1.
        other_gradient = -(probabilities[0]*correct_gradient+probabilities[1]*wrong_gradient)/probabilities[2]
        slopes = (correct_gradient, wrong_gradient, other_gradient)
        assert abs(sum(p*k for p,k in zip(probabilities,slopes))) < 1e-15
        h = 1e-6
        plus = logps(h,slopes); minus = logps(-h,slopes)
        measured_correct = (plus[0]-minus[0])/(2*h)
        measured_wrong = (plus[1]-minus[1])/(2*h)
        derivative = -group_gradient*(correct_gradient-wrong_gradient)
        numerical = -group_gradient*((plus[0]-plus[1])-(minus[0]-minus[1]))/(2*h)
        assert abs(measured_correct-correct_gradient) < 1e-9
        assert abs(measured_wrong-wrong_gradient) < 1e-9
        assert abs(numerical-derivative) < 1e-9
        examples.append(dict(saved_group_gradient=group_gradient,observed_correct_logp_gradient=correct_gradient,
            unobserved_same_prefix_wrong_logp_gradient=wrong_gradient,other_logp_gradient=other_gradient,
            normalized_categorical_probabilities=probabilities,correct_minus_wrong_margin_derivative_under_negative_group=derivative,
            central_difference_derivative=numerical))
    assert examples[0]['correct_minus_wrong_margin_derivative_under_negative_group'] < 0 < examples[1]['correct_minus_wrong_margin_derivative_under_negative_group']
    return examples


def main():
    examples=counterexample()
    # Equal mean label gradient does not identify its first-token gradient.
    first_cases=[]
    for token_gradients in ((1.,-1.),(-1.,1.)):
        mean_gradient=sum(token_gradients)/2
        assert mean_gradient==0.
        first_cases.append(dict(label_token_gradients=token_gradients,mean_gradient=mean_gradient,
                                first_token_derivative_under_negative_unit_group=-token_gradients[0]))
    assert first_cases[0]['first_token_derivative_under_negative_unit_group'] == -first_cases[1]['first_token_derivative_under_negative_unit_group']
    # Mean/sum are different declared measurements with different length weights.
    lengths=(1,3);candidate_mean_gradients=(.1,.09)
    mean_direction=-(candidate_mean_gradients[0]-candidate_mean_gradients[1])
    sum_direction=-(lengths[0]*candidate_mean_gradients[0]-lengths[1]*candidate_mean_gradients[1])
    assert mean_direction<0<sum_direction
    source=Path(__file__).resolve();output=source.parents[1]/'results/MARGIN_IDENTIFIABILITY_INDEPENDENT_MATH_V1.json'
    result=dict(schema='margin-identifiability-independent-math-v1',verdict='CONFIRMED_SAVED_B3_INSUFFICIENT_TO_IDENTIFY_CLASS_MARGIN',
        reviewer_kind='isolated_codex',reviewer_id='Codex /root/full_arm_execution_contract',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        normalized_categorical_counterexample=examples,first_token_gradient_counterexample=first_cases,
        mean_sum_length_weight_counterexample=dict(candidate_label_lengths=lengths,candidate_mean_gradients=candidate_mean_gradients,
            mean_margin_direction=mean_direction,sum_margin_direction=sum_direction),
        measurement_contract=dict(common_context='Same image, native chat prompt and exact answer prefix for both canonical candidates.',
            primary='M_mean=mean(label logps truth)-mean(label logps alternative).',
            secondary='M_sum=sum(label logps truth)-sum(label logps alternative); M_first=first truth logp-first alternative logp.',
            direction='D M[-g]=-dot(saved actual group_gradient, new margin_gradient). No K/a/r conversion of saved group is needed.',
            reuse='Per-candidate gradient(sum)=label_length*gradient(mean). Equality of first and mean gradients is guaranteed for one-token labels; otherwise independently measure first or mark its derivative NOT_MEASURED.',
            conditioning='Common task-answer prefix and saved Dallasresponse0 reasoning prefix are separate experiments; do not compare candidates with unequal prefixes.',
            token_boundary='Freeze full rendered prompt/processor-expanded IDs, canonical labels, exact prefix/label/suffix partition; score only declared label tokens, no EOS/suffix salvage.',
            state='Same original parent weights and numerical policy as saved group gradients; verify state/RNG, strict new scope, no training/generation/update.'),
        limitations=['Synthetic normalized categorical and scalar arithmetic; not observations of a model or MRI task.',
            'Saved full/strict-label response gradients use different saved reasoning contexts and may omit alternative canonical completion; their dot products cannot substitute for a new same-prefix class contrast.',
            'Mean label score is length-normalized scoring, not the log probability of a completed class event. Sum label score is a joint label-span score; first score is a next-token contrast. None by itself is MRI accuracy.',
            'Negative local margin direction is not a finite-step loss of accuracy or generalization. Counterexamples motivate bounded measurement, not a new loss/training authorization.'],
        native_calls=0,model_loads=0,backward_calls=0,optimizer_updates=0,generation_calls=0,patient_or_final100_reads=False,training_readiness='NO')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),native_calls=0)))


if __name__=='__main__':main()
