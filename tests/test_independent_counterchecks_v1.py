# Root-supplied synthetic independent counterchecks (assertions unchanged; module loading adapted to the package).
import copy
import numpy as np,pytest
from PIL import Image

from mri_rl import mri_spatial_task_pilot_v1 as sp, mri_slice_comparison_v1 as sc

def panel_canvas(tmp_path):
 gray=np.arange(364*436,dtype=np.uint32).reshape(436,364)%251;gray=gray.astype(np.uint8)
 im=Image.new('RGB',(752,470),'white');im.paste(Image.fromarray(np.uint8(250-gray)).convert('RGB'),(4,28));im.paste(Image.fromarray(gray).convert('RGB'),(384,28));f=tmp_path/'synthetic.png';im.save(f);return f,gray

def test_paired_canvas_right_t1_crop_and_coordinate_offset(tmp_path):
 f,g=panel_canvas(tmp_path);q=sp.compare_png_panel(f,g,'t1ce');assert q['status']=='EXACT_PANEL' and q['panel']=='t1ce' and q['crop'].size==(364,436)
 assert sp.canvas_to_panel_xy(384+100,28+50,'t1ce')==(100,50)
 assert not q['matches']['flair']

def test_luminance_equal_colored_panel_is_not_exact_raw_render(tmp_path):
 gray=np.full((436,364),128,np.uint8);im=Image.new('RGB',(752,470),'white');p=Image.fromarray(gray).convert('RGB');p.putpixel((30,40),(129,128,125));assert p.convert('L').getpixel((30,40))==128;im.paste(p,(384,28));f=tmp_path/'synthetic_overlay.png';im.save(f)
 assert sp.compare_png_panel(f,gray,'t1ce')['status']!='EXACT_PANEL'

def rows_and_meta():
 labels=sc.ev.LABELS['mri_sequence'];rows=[];meta={}
 for i,label in enumerate(labels):
  oid='r'+str(i)+'_inA';meta[oid]={'pair_key':'p1','modality':['flair','t1ce'][i],'arm':'inA'}
  rows.append({'id':oid,'task':'mri_sequence','truth':label,'greedy':{'text':f'<think>x</think><answer>\\boxed{{{label}}}</answer>'},'candidates':{'per_label':[{'label':l,'mean':-float(i!=j),'sum':-float(i!=j),'first':-float(i!=j)} for j,l in enumerate(labels)]},'costs':dict(greedy_calls=1,candidate_score_API_calls=3,generated_tokens=1,teacherforced_input_positions=20)})
 return rows,meta

def test_duplicate_response_ids_must_not_inflate_condition_metrics():
 rows,meta=rows_and_meta()
 with pytest.raises(ValueError):sc.condition_metrics(rows+[copy.deepcopy(rows[0])],meta)

def test_inA_response_must_not_be_reported_as_inB_data_effect():
 rows,meta=rows_and_meta()
 with pytest.raises(ValueError):sc.analyze({'parent|inA':rows,'parent|inB':copy.deepcopy(rows)},meta,provenance='SYNTHETIC_FIXTURE')

def test_unverified_receipt_string_cannot_mint_actual_model_output_claim():
 rows,meta=rows_and_meta()
 with pytest.raises(ValueError):sc.analyze({'parent|inA':rows},meta,provenance='ACTUAL_MODEL_OUTPUT',receipt_sha256='not-a-receipt')

def test_empty_or_multifocal_remain_abstain_without_clinical_admission():
 m=np.zeros((218,182),np.uint8);q=sp.plane_report(m,'right-top');assert q['technical_status']=='ABSTAIN' and q['scientific_admitted']==0
 m[20:50,130:160]=2;m[160:190,20:50]=3;q=sp.plane_report(m,'right-top');assert q['technical_status']=='ABSTAIN' and 'MULTICOMPONENT_2D' in q['abstain_reasons'] and q['scientific_admitted']==0

def test_fractional_mask_rejected_before_any_endpoint(monkeypatch):
 shape=(182,218,1);aff=np.diag([-1.,1.,1.,1.]);g=dict(shape=list(shape),axcodes='LAS',units='mm',spacing_mm=[1.,1.,1.],qform_code=1,sform_code=1,flags=[],affine=aff.tolist())
 monkeypatch.setattr(sp,'_stat',lambda _: (1,1));monkeypatch.setattr(sp,'_sha',lambda _: '0'*64);monkeypatch.setattr(sp.dp,'inspect_nifti',lambda _: g)
 class I:
  def __init__(self,a):self.dataobj=a
  def get_fdata(self,dtype=None):return self.dataobj.astype(dtype) if dtype is not None else self.dataobj
 v=np.ones(shape,np.float32);mask=np.zeros(shape,np.float32);mask[0,0,0]=.5
 monkeypatch.setattr(sp.nib,'load',lambda p:I(v if p=='image' else mask))
 q,_=sp.process_case(dict(opaque_id='r1',image_nifti_path='image',mask_nifti_path='mask',legacy_slice_index=0,split='train',image_modality='t1ce'))
 assert q['technical_status']=='ABSTAIN' and 'MASK_NONINTEGER_LABELS' in q['abstain_reasons'] and 'grid' not in q
