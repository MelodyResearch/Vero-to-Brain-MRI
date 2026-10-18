"""Read-only, serial one-image CE gradient diagnostic; no optimizer updates."""
import gc,hashlib,json,math,os
from pathlib import Path
CONDITIONS=('REAL_PAIR','SWAPPED_PIXEL_TENSORS_FIXED_TRUTH','ZERO_NORMALIZED_PIXEL_TENSOR')
BUDGET={'logical_pair_conditions':3,'primitive_one_image_CE_gradients':6,'candidate_forwards':12,'generation_tokens':0,'optimizer_updates':0,'images':2,'simultaneous_image_graphs':1}
def require(ok,message):
 if not ok:raise ValueError(message)
def route_pixels(condition,pixels,zeros):
 require(condition in CONDITIONS and len(pixels)==2,'Exactly one paired condition required')
 require(pixels[0].shape==pixels[1].shape,'Pair pixel shapes differ')
 return pixels if condition==CONDITIONS[0] else pixels[::-1] if condition==CONDITIONS[1] else [zeros(v) for v in pixels]
def validate_pair(rows):
 require(len(rows)==2,'One fixed pair required');a,b=rows;ea,eb=a['extra_info'],b['extra_info'];require(a['ability']==b['ability']=='mri_sequence' and ea['internal_split']==eb['internal_split']=='train' and (ea['patient_key'],ea['level'],ea['question'])==(eb['patient_key'],eb['level'],eb['question']),'Same-person/level/question training pair required');require([r['reward_model']['ground_truth'] for r in rows]==['FLAIR','contrast-enhanced T1'] and a['images']!=b['images'],'Ordered opposite-label real images required')
def serial_pair_gradient(vg,model,inputs,pixels,partitions,tree_map,evaluate,before,clear):
 summed=None;losses=[];scores=[]
 for truth,(prepared,pixel,parts) in enumerate(zip(inputs,pixels,partitions)):
  before('pair_gradient_one_image_'+str(truth));kwargs={k:v for k,v in prepared.items() if k not in ('input_ids','attention_mask','pixel_values')};require(not any(k in kwargs for k in ('cached_image_features','image_features','cache','inputs_embeds')),'Cached features/cache forbidden')
  (loss,candidate_scores),grad=vg(model,prepared['input_ids'],pixel,kwargs,parts,truth);evaluate(loss,candidate_scores,grad);losses.append(float(loss.item()));scores.append(candidate_scores.tolist());scaled=tree_map(lambda g:g/2,grad);summed=scaled if summed is None else tree_map(lambda a,b:a+b,summed,scaled);evaluate(summed);del loss,candidate_scores,grad,scaled;clear()
 return summed,losses,scores

def vector_comparison(chunks):
 import numpy as np
 aa=bb=ab=dd=0.;maximum=0.;count=0
 for name,shape,a,b in chunks:
  require(a.shape==b.shape and a.size==math.prod(shape) and np.isfinite(a).all() and np.isfinite(b).all(),'Gradient vectors invalid');x=a.astype(np.float64);y=b.astype(np.float64);d=x-y;aa+=float(np.dot(x,x));bb+=float(np.dot(y,y));ab+=float(np.dot(x,y));dd+=float(np.dot(d,d));maximum=max(maximum,float(np.max(np.abs(d))));count+=1
 require(count==504,'504 gradient tensors required');return {'tensors':count,'left_l2':math.sqrt(aa),'right_l2':math.sqrt(bb),'difference_l2':math.sqrt(dd),'cosine':ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None,'max_abs_difference':maximum}

def run_native(backend,rows,plan,paired,al,utility,out,before):
 """Exactly3 logical conditions,6 serial CE autograd calls/12 forwards. No update."""
 from mri_rl.optimizer_checkpoint import pack_tree
 validate_pair(rows);mx=backend.mx;backend.model.eval();out=Path(out);inputs=[];parts=[];raw_pixels=[]
 def state_fingerprint():
  layout,arrays=pack_tree({'optimizer':backend.optimizer_state(),'rng':backend.rng_state()});return {'layout':layout,'arrays':utility._fingerprints(backend,arrays)}
 before('paired_grad_initial_integrity');weights_before=utility._weights(backend);state_before=state_fingerprint();state=backend.optimizer_state();rng=backend.rng_state();require(int(state['step'].item())==56,'Exact inherited Adam56 required')
 for i,row in enumerate(rows):
  before('paired_grad_prepare_'+str(i));messages=[{'role':'system','content':backend.contract.system_instruction},{'role':'user','content':row['extra_info']['question']}];prompt=backend.template(backend.processor,backend.model.config,messages,num_images=1,enable_thinking=False,add_generation_prompt=True);require(prompt.count('<|image_pad|>')==1 and prompt.endswith(backend.contract.assistant_prefix),'Frozen image prompt/prefix changed');p=backend.prepare_inputs(backend.processor,images=row['images'],prompts=prompt,image_token_index=backend.model.config.image_token_index,add_special_tokens=backend.special_tokens(backend.model.config.model_type,backend.processor));require(not any(k in p for k in ('cached_image_features','image_features','cache','inputs_embeds')),'Cached image route forbidden');P=int(p['input_ids'].shape[-1]);require(P==plan['input_tokens'][row['id']] and bool(mx.all(p['attention_mask']==1).item()),'Measured prompt/mask mismatch');grid=p['image_grid_thw'].tolist();visual=sum(t*h*w//int(backend.processor.image_processor.merge_size)**2 for t,h,w in grid);sp=[al.partition_target(backend.processor.tokenizer,label,prompt_length=P) for label in paired.LABELS]
  for label,part in zip(paired.LABELS,sp):require(part['target_ids']==plan['target_ids'][row['id']][label],'Frozen target mask changed');paired.validate_candidate_alignment(backend.processor.tokenizer,prompt,p['input_ids'][0].tolist(),al.PREFIX+label+al.SUFFIX,part,backend.model.config.image_token_index,visual)
  require(utility.sha(row['images'][0])==plan['image_sha256'][row['id']],'Actual source MRI changed');inputs.append(p);parts.append(sp);raw_pixels.append(p['pixel_values'])
 require(inputs[0]['input_ids'].tolist()==inputs[1]['input_ids'].tolist() and inputs[0]['image_grid_thw'].tolist()==inputs[1]['image_grid_thw'].tolist() and raw_pixels[0].shape==raw_pixels[1].shape,'Paired swap requires same token/grid/shape contract')
 fp=[utility._fingerprints(backend,p) for p in inputs];require(all(not v['pixel_values']['all_zero'] for v in fp),'Real source pixel tensor is empty')
 def loss_fn(model,prompt_ids,pixels,kwargs,partition,truth):
  chosen=[];P=int(prompt_ids.shape[-1])
  for part in partition:
   before('paired_grad_candidate_forward');target=mx.array(part['target_ids'],dtype=prompt_ids.dtype);ids=mx.concatenate((prompt_ids,target[None,:]),axis=1);logits=model(ids[:,:-1],pixels,mx.ones_like(ids[:,:-1]),**kwargs).logits;require(int(logits.shape[1])==int(ids.shape[1])-1,'Causal logits shape changed');lp=backend.nn.log_softmax(logits[:,P-1:,:].astype(mx.float32),axis=-1);chosen.append(mx.take_along_axis(lp,target[None,:,None],axis=-1)[0,:,0])
  loss=paired.discrimination_loss(chosen,partition,truth,mx);scores=mx.stack([mx.mean(v[len(p['prefix_ids']):len(p['prefix_ids'])+len(p['label_ids'])]) for v,p in zip(chosen,partition)]);return loss,scores
 vg=backend.nn.value_and_grad(backend.model,loss_fn);conditions={}
 for name in CONDITIONS:
  before('paired_grad_condition_'+name);backend.restore(state,rng);pixel=route_pixels(name,raw_pixels,mx.zeros_like);mx.eval(pixel);pixel_fp=utility._fingerprints(backend,{str(i):v for i,v in enumerate(pixel)})
  gradient,losses,scores=serial_pair_gradient(vg,backend.model,inputs,pixel,parts,backend.tree_map,mx.eval,before,lambda:(gc.collect(),mx.clear_cache()));arrays={k:v.astype(mx.float32) for k,v in backend.tree_flatten(gradient)};require(set(arrays)==set(weights_before) and all(bool(mx.all(mx.isfinite(v)).item()) for v in arrays.values()),'Gradient inventory/nonfinite failure');require(all(math.isfinite(x) and x>=0 for x in losses) and all(math.isfinite(x) for row in scores for x in row),'Nonfinite candidate loss/score');path=out/(name+'_gradient_PRIVATE.safetensors');backend.save_arrays(path,arrays);utility._fsync(path);del gradient,arrays,pixel;gc.collect();mx.clear_cache();backend.restore(state,rng);require(state_fingerprint()==state_before,'Diagnostic changed optimizer/RNG');after=utility._weights(backend);require(after==weights_before,'Diagnostic changed adapter weights')
  margins=[v[1]-v[0] for v in scores];conditions[name]={'mean_pair_CE_loss':sum(losses)/2,'per_image_CE_losses':losses,'candidate_mean_label_logps':scores,'t1_minus_flair_margins':margins,'paired_separation':margins[1]-margins[0],'gradient_sha256':utility.sha(path),'pixel_tensor_fingerprints_PRIVATE':pixel_fp,'optimizer_RNG_unchanged':True,'adapter_digest':utility.canonical(after),'primitive_gradients':2,'candidate_forwards':4};utility.durable(out/(name+'_PRIVATE.json'),conditions[name])
 comparisons={}
 for name in CONDITIONS[1:]:comparisons[name+'_vs_REAL_PAIR']=vector_comparison(utility._gradient_chunks(out/(CONDITIONS[0]+'_gradient_PRIVATE.safetensors'),out/(name+'_gradient_PRIVATE.safetensors')))
 before('paired_grad_final_integrity');backend.restore(state,rng);require(state_fingerprint()==state_before and utility._weights(backend)==weights_before,'Final state preservation failure');private={'input_fingerprints':fp,'weights_before':weights_before,'weights_after':utility._weights(backend),'state_before':state_before,'state_after':state_fingerprint()};utility.durable(out/'integrity_PRIVATE.json',private)
 safeconditions={name:{k:v for k,v in value.items() if not k.endswith('_PRIVATE')} for name,value in conditions.items()};result={'schema':'paired-native-readonly-gradient-measurements-v1','status':'ACTUAL_GRADIENT_MEASUREMENTS_PENDING_RELEASE_AND_INTERPRETATION','conditions':safeconditions,'comparisons':comparisons,'budget':BUDGET,'adapter_tensors_checked':504,'adapter_before_digest':utility.canonical(weights_before),'adapter_after_digest':utility.canonical(weights_before),'optimizer_step_before':56,'optimizer_step_after':56,'optimizer_RNG_preserved':True,'integrity_sha256':utility.sha(out/'integrity_PRIVATE.json'),'numerical_backward_independently_verified':False,'limitations':['Conditional image gradient sensitivity is not MRI accuracy, anatomical grounding or generalization.','ZERO_NORMALIZED_PIXEL_TENSOR is OOD with unchanged grid, not a rendered black image or no-image prompt.','SWAPPED_PIXEL_TENSORS_FIXED_TRUTH deliberately contradicts displayed image labels for a diagnostic; no relabelled data are used for training.','CE candidate score gradient is a different diagnostic objective from the completed GSPO update, not a new trained method.','Six primitive one-image CE gradients are aggregated into three pair conditions; at most one image autograd graph at a time.']}
 # Drop all model-associated input/state references before caller closes.
 del inputs,parts,raw_pixels,state,rng,p,sp,vg,fp;gc.collect();mx.clear_cache();return result
