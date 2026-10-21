# Micro pilot resource revision v2 —9GiB/50% with independent child guardian

**Source/CPU preparation only. New model execution, GSPO updates and accuracy: NOT_RUN.** The previous scope stopped before loading; it is preserved and cannot be automatically retried. The successful historical GPU cache is reused, never rebuilt.

## Exact change

| Loading admission | Previous | This local micro version |
|---|---:|---:|
|Available memory minimum|10GiB|**9GiB**|
|Memory-pressure free minimum|40%|**50%**|

These are the only changed MODEL_POLICY fields. Both must pass. Resident6GiB/free35%,RSS14GiB,MLX12GiB/localcache256MiB,diskforecast+20GiBreserve,battery20%,boundedHOLD120s,all originalSTOPs and the originalswapbaseline13.96966796875GiB remain unchanged. Netgrowthcap1.5GiB yields15.46966796875GiB effective ceiling; absolute16GiB is also retained. Ordinary swaprate alone is not STOP. No global OS/training gate or V1 file is modified.

## Independent loading-period observation

A separate non-MLX parent monitors its exact Popen-owned child approximately every2s: childRSS,availablememory,pressure-free,OSpressure,swap,battery anddisk. Child guards still check actualMLX at native boundaries. The parent cannot query another process's MLXallocator; it does not invent zero readings. Actual child reported peaks after return and exact wait4RSS after exit are distinct evidence.

Existing hard-risk conditions stop only this experiment's child:OSCRITICAL/free≤15%,available<2GiB or<4GiB for30s,RSS>14GiB,original-baseline net swap>1.5GiB orabsolute>16GiB,battery<20%,diskbelow26GiBforecast+reserve. Unknownessential sensors use the existing120s bound; invalidpercentages/OSenums count as unknown. Softloading shortfall or ordinary swaprate does not trigger parenttermination.

On guardianSTOP the parent sends SIGTERM only to its own unreaped child, thenSIGKILL after10s if needed. Ownership is checked withwait4 before emergency signaling; an already reaped PID is never signaled. Monitor-log failure also stops that child. An interrupted parent cleans up its child. Closedexit/log witnesses remain durable before optionalpost-sensors. GuardianSTOP blocks progression even if the child happened to return0.

Sampling, delayed signals and atomic native/kernel operations cannot guarantee against OOM, severe stalls or systemfailure. No other apps/process groups are terminated; no OSprotections/shutdown/reboot changes.

## Preserved scientific experiment

The originalAdam52/boundary114,its adapter/fullAdam/RNG,existingfirstTRAINpair+2Dallas,allrealpixels/prompts,GSPOloss/reward/gate/LR/LoRA/F32 andDEV24 remain unchanged. New authority is a resource-v2 subclass accepted by the unchanged native typeguard; exactscope/action/review still require fresh human permission. V1 authority cannot launch V2.

|Stage|maxreturned|maxTFpositions|maxbackward|maxupdates|maxloads|Operation-start window|
|---|---:|---:|---:|---:|---:|---:|
|2:load/F32/2MRIresponses|512|2808|0|0|1|900s|
|3:shared16,Direct/Paired-Prompt|4096|27072|32|2total/1each|1|1800s|
|4:allDEV24greedyParent/A/B|18432|0|0|0|3|2400s|

Total23040returnedtokens,54scorerforwards,29880TFpositions,32backward,2updates,5serialloads,5100soperation-startwindow. Stage4 has0candidateforwards. Zero-update exactparent aliases remain explicit. No resampling, extra steps, new preprocessing or futureSmallPilot expansion. Each next stage requires actualPASS,allocatorrelease,closedchildexit,unlockedGPU andfreshresources under the new combined grant. TrueSTOP ends the attempt without restart.

## Actual preparation evidence

Three read-only samples08:40:22–32UTC:available9.7736–10.1941GiB/free54–56%,OS1,swap14.525205GiB stable,AC/battery72%,otherheavy0/lockfree,disk294.64GiB. AllCONTINUE under9/50 at observation; actual launch must resample. ThermalSENSOR_UNAVAILABLE and uninitializedMLX are recorded as unknown, not normal/zero.

Root targeted43CPUtests passed in.43s, including the same independentownershipcounterexample. All process-signal tests target only synthetic tinyCPUchildren created by those tests. Independentreview found sensor-range and emergency-PIDownership errors; their unchanged counterexamples changedRED→GREEN. No MRI/model/GPU call or newoptimizerupdate occurred.

Unsigned initialV2scope41e76... was superseded after the ownership fix; no grant or execution used it. FinalscopeFILE:
`5dd24d8dc73241d11460e18fc4b25ac8597bbf6a3d0b2f6283df12b43a2c039c`
Canonical:
`c96a71b4e9680f6af36b666f6aa2aa6e85c74906809790efff12430502434665`

[Preparation aggregate](../results/SAFE_MICRO_PILOT234_RESOURCE_V2_PREPARATION_ACTUAL_V1.json).
[Independent review](../results/SAFE_MICRO_PILOT234_RESOURCE_V2_INDEPENDENT_REVIEW_V1.json).

The broader frozen research plan/lock is unchanged. RealParent/A/B accuracy remainsNOT_RUN; a new exact execution grant is required before anynativecall.

Final independent review: Codex `/root/resource_v2_final_review`, **PASS_WITH_LIMITATIONS**, 22 focused CPU checks passed in0.37s; exact92source/20metadata scope reproduced and all source bindings rechecked. The review is preparation evidence, not execution permission. Fresh human authorization remainsNOT_ISSUED.
