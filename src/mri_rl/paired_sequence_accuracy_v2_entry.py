"""Small, separately reviewed accuracy-focused MRI V2; never edits frozen RL.

Commands: plan (CPU counts), scope (CPU exact private scope), train (reviewed
one-heavyweight step), eval (frozen internal-dev candidate accuracy). Native
runs require the existing private approval and Mac resource gates.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from mri_rl import paired_sequence_accuracy_v2 as paired

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ("paired_sequence_accuracy_v2.py", "paired_sequence_accuracy_v2_entry.py",
                "answer_likelihood.py", "mlx_continuous_gspo.py", "continuous_rl.py",
                "optimizer_checkpoint.py", "paced_resources.py")
CONFIG = {"profile": paired.VERSION, "objective": "label-only-paired-margin-plus-two-class",
          "class_weight": paired.CLASS_WEIGHT, "future_sentinel_id":paired.FUTURE_SENTINEL_TOKEN_ID,
          "optimizer":{"name":"Adam","learning_rate":1e-5,"betas":[.9,.999],
                       "eps":1e-8,"bias_correction":False},
          "trainable":"language_lora4_frozen_vision", "matched_original_RL_compute":False}


def read_jsonl(path):
    rows=[];ids=set()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row=json.loads(line)
            paired.require(row["id"] not in ids, "Duplicate source record")
            ids.add(row["id"]);rows.append(row)
    return rows


def source_hashes():
    return {n:paired.sha(ROOT/"src"/"mri_rl"/n) for n in SOURCE_NAMES}


def make_step(pair):
    f=pair[paired.LABELS[0]]
    return {"id":pair["id"],"ability":paired.VERSION,"data_source":"OpenNeuro_ds007045",
            "images":f["images"],
            "extra_info":{"reward_type":paired.VERSION,"internal_split":"train",
                          "patient_lineage":"BIDS_SUBJECT_VERIFIED","patient_key":pair["patient_key"]},
            **pair}


def validate_step(step):
    f=step[paired.LABELS[0]];t=step[paired.LABELS[1]]
    paired.require(step["ability"] == paired.VERSION and step["extra_info"]["reward_type"] == paired.VERSION,
                   "Unexpected paired training route")
    paired.require(step["images"] == f["images"] and step["extra_info"]["patient_key"] == step["patient_key"],
                   "Step source image/patient mismatch")
    pairs=paired.patient_pairs([f,t],levels=(step["level"],))
    paired.require(len(pairs)==1 and pairs[0]["patient_key"]==step["patient_key"] and
                   pairs[0]["level"]==step["level"],"Step has invalid patient/slice provenance")


def prepare(train,dev,*,seed,steps,levels=(45,91,136)):
    tr=read_jsonl(train); dv=read_jsonl(dev)
    pairs=paired.patient_pairs(tr,levels=levels)
    schedule=[make_step(x) for x in paired.schedule_pairs(pairs,seed=seed,steps=steps)]
    for step in schedule:validate_step(step)
    tr_people={r["extra_info"]["patient_key"] for r in tr if r["ability"]=="mri_sequence"}
    dev_seq=[r for r in dv if r.get("ability")=="mri_sequence"]
    paired.require(dev_seq, "No internal-dev sequence records")
    dev_people=set()
    for row in dev_seq:
        info=row["extra_info"]
        paired.require(info.get("internal_split")=="internal_dev" and
                       row["data_source"]=="OpenNeuro_ds007045" and
                       info.get("patient_lineage")=="BIDS_SUBJECT_VERIFIED" and
                       row["reward_model"]["ground_truth"] in paired.LABELS and
                       len(row["images"])==1 and Path(row["images"][0]).is_file(),
                       "Only existing independent internal-dev MRI allowed")
        dev_people.add(info["patient_key"])
    paired.require(not tr_people.intersection(dev_people), "Development patient overlaps training")
    return {"train":tr,"dev":dev_seq,"schedule":schedule,
            "train_people":len(tr_people),"dev_people":len(dev_people),"pairs":len(pairs)}


def all_pair_image_digest(schedule):
    from mri_rl import continuous_rl as cr
    hashes={s["id"]:{label:paired.sha(s[label]["images"][0]) for label in paired.LABELS}
            for s in schedule}
    return hashlib.sha256(cr.canonical(hashes).encode()).hexdigest()


def run_binding(prepared, *, train, dev, adapter, seed, model, contract):
    from mri_rl import continuous_rl as cr
    evidence={"v2_version":paired.VERSION,"train_sha256":paired.sha(train),
              "dev_sha256":paired.sha(dev),"starting_adapter_sha256":paired.sha(adapter),
              "contract_sha256":paired.sha(contract),"seed":seed,
              "model_config_sha256":paired.sha(Path(model)/"config.json"),
              "source_sha256":source_hashes(),
              "all_source_image_sha256":all_pair_image_digest(prepared["schedule"])}
    return cr._binding(prepared["schedule"], evidence, CONFIG, validate_step)


def training_scope(binding, *, output, resume, max_new_steps, cache_binding=None):
    from mri_rl import continuous_rl as cr
    state=cr.inspect_boundaries(output,binding) if resume else None
    paired.require(resume or not Path(output).exists(),"Fresh output exists; explicit resume required")
    paired.require(not resume or state is not None,"Resume needs a complete checkpoint chain")
    cursor=state["completed_groups"] if state else 0
    paired.require(type(max_new_steps) is int and 1<=max_new_steps<=4,"Small 1–4 pair step budget required")
    return {"kind":"paired-MRI-visual-accuracy-V2", "execution":"TRAIN_NEW_V2_ONLY",
            "binding_sha256":hashlib.sha256(cr.canonical(binding).encode()).hexdigest(),
            "completed_steps":cursor,"maximum_new_optimizer_steps":max_new_steps,
            "next_record_ids":binding["schedule_record_ids"][cursor:cursor+max_new_steps],
            "starting_receipt_sha256":cr.digest(state["path"]/"receipt.json") if state else None,
            "output":str(Path(output).resolve()),"derived_cache":cache_binding,
            "raw_study_changed":False,"dev_used_for_training":False,
            "freeze_final_test":True,"native_actual_accuracy":"NOT_RUN"}


def train_paired(*, output, schedule, binding, starting_adapter, backend_factory,
                 authorize, resume=False, max_new_steps=1):
    """Atomic model/Adam/RNG continuation using existing continuous_rl boundaries."""
    from mri_rl import continuous_rl as cr
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    paired.require(callable(authorize),"Explicit native workload authorizer required")
    paired.require(1<=max_new_steps<=4,"Too many concurrent V2 training steps")
    path=Path(output)
    paired.require(not path.exists() or resume,"Existing run must explicitly resume")
    paired.require(binding["experiment"]["all_source_image_sha256"] == all_pair_image_digest(schedule),
                   "Complete schedule image hashes changed before training")
    path.mkdir(parents=True,exist_ok=True)
    with (path/".writer.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("A paired V2 trainer is already running") from None
        current=cr.inspect_boundaries(path,binding)
        paired.require(not resume or current is not None,"Missing complete resume checkpoint")
        cursor=current["completed_groups"] if current else 0
        if cursor==len(schedule):
            return {"status":"V2_SCHEDULE_COMPLETE","completed_steps":cursor,
                    "optimizer_updates":current["optimizer_updates"]}
        paired.require(not list(path.glob(".pending-*")),
                       "Interrupted V2 candidate must be reconciled before a retry; do not double-charge work")
        adapter=current["path"]/"adapter.safetensors" if current else starting_adapter
        authorize("load")
        backend=backend_factory(adapter,CONFIG)
        try:
            if current:
                expected={"run":binding,"adapter_sha256":cr.digest(adapter),
                          "trainable_signature":backend.trainable_signature()}
                state,rng,_=load_optimizer_checkpoint(current["path"]/"optimizer",
                                     expected_bindings=expected,load_arrays=backend.load_arrays)
                backend.restore(state,rng)
            else:
                paired.require(cr._step(backend)==0,"Fresh V2 requires zero-step Adam")
                work=Path(tempfile.mkdtemp(prefix=".pending-",dir=path))
                current=cr._publish(path,work,backend,binding,0,0,None,
                                    {"updated":False,"record_id":None,"schedule_record_id":None})
            for idx in range(cursor,min(cursor+max_new_steps,len(schedule))):
                step=schedule[idx]
                authorize("pair")
                actual={label:paired.sha(step[label]["images"][0]) for label in paired.LABELS}
                paired.require(binding["experiment"]["all_source_image_sha256"] == all_pair_image_digest(schedule),
                               "Complete schedule image hashes changed during training")
                old_step=cr._step(backend)
                paired.require(old_step==current["optimizer_updates"],"Lost Adam step count")
                work=Path(tempfile.mkdtemp(prefix=".pending-",dir=path))
                cr.write_json(work/"attempt_reservation.json",{
                    "kind":"V2-paired-MRI-attempt", "record_id":step["id"],
                    "parent_receipt_sha256":cr.digest(current["path"]/"receipt.json"),
                    "image_sha256":actual,"training_step":idx})
                cr.sync_dir(work);cr.sync_dir(path)
                previous=cr.state_fingerprint(backend,backend.optimizer_state(),work/"optimizer_before.safetensors")
                numeric=backend.paired_step(step,actual)
                paired.require(all(paired.sha(step[label]["images"][0])==actual[label] for label in paired.LABELS),
                               "Source MRI changed during V2 update")
                paired.require(cr._step(backend)==old_step+1 and
                               numeric["optimizer_step"]==old_step+1 and
                               math.isfinite(numeric["gradient_l2"]) and numeric["gradient_l2"]>0,
                               "V2 needs a real finite optimizer update")
                cr.write_json(work/"rollout_PRIVATE.json",numeric)
                following=cr.state_fingerprint(backend,backend.optimizer_state(),work/"optimizer_after.safetensors")
                check=work/"adapter_check.safetensors"
                backend.save_adapter(check)
                changed=cr.digest(check)!=current["files_sha256"]["adapter.safetensors"]
                check.unlink()
                paired.require(changed,"Optimizer step did not change adapter")
                current=cr._publish(path,work,backend,binding,idx+1,old_step+1,
                            cr.digest(current["path"]/"receipt.json"),
                            {"updated":True,"record_id":step["id"],"schedule_record_id":step["id"],
                             "optimizer_before_fingerprint":previous,"optimizer_after_fingerprint":following,
                             "paired_objective":numeric["objective"],
                             "gradient_l2":numeric["gradient_l2"],
                             "scope":"V2_EXPLORATORY_NOT_ORIGINAL_18_ARM"})
            return {"status":"V2_SCHEDULE_COMPLETE" if current["completed_groups"]==len(schedule)
                    else "V2_BOUNDED_STAGE_COMPLETE", "completed_steps":current["completed_groups"],
                    "optimizer_updates":current["optimizer_updates"],
                    "final_boundary_receipt_sha256":cr.digest(current["path"]/"receipt.json")}
        finally:backend.close()


def score_internal_dev(*,rows,backend,output,contract,max_tokens=256,resume=False):
    """Actually measure both candidate and strict greedy MRI accuracy, atomically by row."""
    from mri_rl import continuous_rl as cr
    from mri_rl.vero_mri_reward import compute_score_from_data_source
    path=Path(output)
    paired.require(not path.exists() or resume,"Do not overwrite development predictions")
    path.parent.mkdir(parents=True,exist_ok=True)
    existing=[]
    if resume:
        paired.require(path.is_file(),"No existing development prefix to resume")
        raw=path.read_bytes()
        paired.require(not raw or raw.endswith(b"\n"),"Torn development result")
        existing=[json.loads(line) for line in raw.splitlines() if line.strip()]
        paired.require(len(existing)<=len(rows) and all(existing[i]["record_id"]==rows[i]["id"] for i in range(len(existing))),
                       "Existing development prefix differs from frozen row order")
        # Already saved predictions remain immutable. Rechecks run only on raw stored results.
    counts={label:{"greedy_correct":0,"candidate_correct":0,"total":0} for label in paired.LABELS}
    units={};completed=0
    with path.open("a" if resume else "x") as handle:
        for index,row in enumerate(rows):
            target=row["reward_model"]["ground_truth"]
            paired.require(target in counts and row["extra_info"]["internal_split"]=="internal_dev",
                           "Not an internal-dev MRI sequence input")
            if index<len(existing):
                answer=existing[index]
            else:
                img_hash=paired.sha(row["images"][0])
                answer=backend.score_sequence(row,img_hash)
                native=backend.generate_record({"record_id":row["id"],"image":row["images"][0],
                          "prompt":[{"role":"system","content":contract.system_instruction},*row["prompt"]]},
                          img_hash,max_tokens)
                reward=compute_score_from_data_source(row["data_source"],native["response"],target,
                                                     row["extra_info"])
                answer["native_greedy_response_PRIVATE"]=native["response"]
                answer["native_greedy_finish_reason"]=native.get("finish_reason")
                answer["native_greedy_format_score"]=reward["format"]
                answer["native_greedy_correct"]=bool(reward["accuracy"])
                answer["native_greedy_generation_tokens"]=native["generation_tokens"]
                paired.require(paired.sha(row["images"][0])==img_hash,"Development image changed after native inference")
                handle.write(cr.canonical(answer)+"\n");handle.flush();os.fsync(handle.fileno())
            counts[target]["total"]+=1
            counts[target]["candidate_correct"]+=int(answer["forced_prefix_correct"])
            counts[target]["greedy_correct"]+=int(answer["native_greedy_correct"])
            person=row["extra_info"]["patient_key"]
            units.setdefault(person,[]).append((int(answer["forced_prefix_correct"]),int(answer["native_greedy_correct"])))
            completed+=1
    return {"status":"ACTUAL_INTERNAL_DEV_CANDIDATE_AND_GREEDY",
            "records":completed,"patients":len(units),"per_class":counts,
            "candidate_patient_mean":sum(sum(v[0] for v in x)/len(x) for x in units.values())/len(units),
            "greedy_patient_mean":sum(sum(v[1] for v in x)/len(x) for x in units.values())/len(units),
            "candidate_accuracy":sum(c["candidate_correct"] for c in counts.values())/completed,
            "strict_greedy_accuracy":sum(c["greedy_correct"] for c in counts.values())/completed,
            "independent_generalization":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("plan","scope","train","eval"))
    for a in ("train","dev","model","contract","adapter","output"):
        p.add_argument("--"+a,type=Path,required=True)
    for a in ("scope-output","review-verdict","cache-dir","cache-certificate","evaluation-adapter"):
        p.add_argument("--"+a,type=Path)
    p.add_argument("--expected-review-sha256")
    p.add_argument("--cache-certificate-sha256")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--steps",type=int,default=28)
    p.add_argument("--max-new-steps",type=int,default=1)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    for k,value in vars(args).items():
        if isinstance(value,Path):setattr(args,k,value.resolve())
    paired.require(args.output.is_relative_to(ROOT/"local_data"),"Outputs must stay private under local_data")
    prepared=prepare(args.train,args.dev,seed=args.seed,steps=args.steps)
    if args.command=="plan":
        print(json.dumps({"status":"PAIR_PLAN_CPU_ONLY","steps":len(prepared["schedule"]),
            "matched_train_pairs":prepared["pairs"],"unique_train_people":prepared["train_people"],
            "internal_dev_people":prepared["dev_people"],"native_accuracy":"NOT_MEASURED"}));return
    from mri_rl.experiment_contract import _check_review,load_contract
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from mri_rl.paced_resources import PacedGate
    contract=load_contract(args.contract)
    paired.require(contract.train_manifest_sha256 == paired.sha(args.train) and
                   contract.dev_manifest_sha256 == paired.sha(args.dev),
                   "Contract must bind the exact patient-disjoint V2 training/dev source manifests")
    cache_binding=cache_binding_for(args)
    cache={"dir":str(args.cache_dir)} if cache_binding else None
    chosen_adapter=args.evaluation_adapter if args.command=="eval" else args.adapter
    paired.require(chosen_adapter is not None and chosen_adapter.is_file(),"Exact native adapter required")
    binding=run_binding(prepared,train=args.train,dev=args.dev,adapter=args.adapter,
                        seed=args.seed,model=args.model,contract=args.contract)
    if args.command=="eval":
        partial_sha=paired.sha(args.output) if args.resume and args.output.is_file() else None
        scope={"kind":"paired-MRI-visual-accuracy-V2", "execution":"READ_ONLY_DEV_CANDIDATE_AND_GREEDY",
               "binding_sha256":hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
               "evaluation_adapter_sha256":paired.sha(chosen_adapter),
               "evaluation_rows_sha256":hashlib.sha256(json.dumps([r["id"] for r in prepared["dev"]]).encode()).hexdigest(),
               "evaluation_existing_prefix_sha256":partial_sha,
               "output":str(args.output),"derived_cache":cache_binding,
               "sealed_final_test":"NOT_ACCESSED"}
    else:
        scope=training_scope(binding,output=args.output,resume=args.resume,
                             max_new_steps=args.max_new_steps,cache_binding=cache_binding)
    if args.command=="scope":
        paired.require(args.scope_output is not None and args.scope_output.is_relative_to(ROOT/"local_data"),
                       "Reviewed scope must be saved privately")
        with args.scope_output.open("x") as f:json.dump(scope,f,indent=2,sort_keys=True);f.write("\n")
        print(json.dumps({"status":"V2_SCOPE_ONLY_NO_NATIVE","scope_sha256":hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':')).encode()).hexdigest()}));return
    _check_review(review_verdict_path=args.review_verdict,
                  expected_review_sha256=args.expected_review_sha256,scope=scope)
    gate=PacedGate(args.output.with_name(args.output.name+"_resource_events.jsonl"))
    def before(name):return gate.before_unit(name,loading=is_loading_unit(name,cache))
    cls=paired.native_backend_class()
    if args.command=="train":
        result=train_paired(output=args.output,schedule=prepared["schedule"],binding=binding,
            starting_adapter=args.adapter,resume=args.resume,max_new_steps=args.max_new_steps,
            backend_factory=lambda a,c:cls(a,c,model_path=args.model,contract=contract,
                                           seed=args.seed,before_unit=before,cache=cache),
            authorize=lambda phase:before("paired_v2_authorize_"+phase))
    else:
        backend=cls(chosen_adapter,CONFIG,model_path=args.model,
                    contract=contract,seed=args.seed,before_unit=before,cache=cache)
        try:result=score_internal_dev(rows=prepared["dev"],backend=backend,output=args.output,
                                      contract=contract,max_tokens=contract.generation["max_tokens"],resume=args.resume)
        finally:backend.close()
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
