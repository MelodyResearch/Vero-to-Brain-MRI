"""CPU TRAIN qualification primitives. No model imports or clinical admission."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require(value, message):
    if not value:
        raise ValueError(message)

def require_train(rows):
    require(bool(rows) and all(r['extra_info']['internal_split']=='train' for r in rows), 'TRAIN-only rows required')

def render_slice(volume, z):
    """Reconstruct historical image pixels; never consult mask to select level."""
    lo,hi=np.percentile(volume[volume!=0][::16],[.5,99.5])
    require(hi>lo and np.isfinite(volume).all(), 'Invalid MRI')
    pixels=np.uint8(np.round(np.clip((np.rot90(volume[:,:,z])-lo)/(hi-lo),0,1)*255))
    return Image.fromarray(pixels).resize((364,436),Image.Resampling.BILINEAR)

def grid(x,y,width,height):
    return ('left','center','right')[min(2,int(x//(width/3)))]+ '-'+('top','middle','bottom')[min(2,int(y//(height/3)))]

def spatial_geometry(mask):
    """Engineering geometry only; mask nonzero does not establish visibility."""
    rotated=np.rot90(np.asarray(mask,dtype=bool)); coords=np.argwhere(rotated)
    if not len(coords):
        return {'area':0,'components':0,'geometric_flags':['MASK_NEGATIVE_NOT_VISUAL_NEGATIVE']}
    cy,cx=coords.mean(axis=0);h,w=rotated.shape
    labels,n=ndimage.label(rotated,structure=np.ones((3,3)))
    counts=np.bincount(labels.ravel())[1:]; largest=int(np.argmax(counts))+1
    ly,lx=np.argwhere(labels==largest).mean(axis=0)
    cell=grid(cx,cy,w,h);distance=float(min(abs(cx-w/3),abs(cx-2*w/3),abs(cy-h/3),abs(cy-2*h/3))*2)
    flags=[]
    if len(coords)<1000:flags.append('SMALL_MASK_AREA_LT1000_NOT_VISIBILITY_THRESHOLD')
    if distance<10:flags.append('CENTROID_WITHIN10_DISPLAY_PX_OF_GRID_BOUNDARY')
    if n>1:flags.append('MULTICOMPONENT_MASK_REQUIRES_VISUAL_ADJUDICATION')
    if not rotated[int(round(cy)),int(round(cx))]:flags.append('CENTROID_OFF_MASK')
    if grid(lx,ly,w,h)!=cell:flags.append('GLOBAL_VS_LARGEST_COMPONENT_GRID_DISAGREEMENT')
    if grid(cx+.25,cy+.25,w,h)!=cell:flags.append('PIXEL_CENTER_CHANGES_GRID')
    return {'area':len(coords),'components':n,'component_areas':sorted(map(int,counts),reverse=True),
            'centroid_xy_display_pixelcenter':[float(2*cx+.5),float(2*cy+.5)],
            'global_grid':cell,'largest_component_grid':grid(lx,ly,w,h),
            'boundary_distance_display_pixels':distance,'geometric_flags':flags}

def cranial_roi_occlusion(panel):
    """Fixed geometric occlusion, NOT skull stripping or validated brain-only MRI."""
    require(panel.size==(384,384),'Dallas panel geometry mismatch')
    mask=Image.new('L',panel.size,0);draw=ImageDraw.Draw(mask)
    for row in range(3):
        for col in range(3):
            x,y=128*col,128*row
            draw.ellipse((x+25,y+10,x+103,y+91),fill=255)
    return Image.composite(panel,Image.new(panel.mode,panel.size),mask)

def sheet(images, identifiers, path, columns=2, tile_width=768, tile_height=408):
    require(len(images)==len(identifiers),'Sheet identity mismatch')
    canvas=Image.new('RGB',(columns*tile_width,((len(images)+columns-1)//columns)*tile_height),'#111111')
    draw=ImageDraw.Draw(canvas)
    for i,(im,key) in enumerate(zip(images,identifiers)):
        im=im.convert('RGB');im.thumbnail((tile_width-8,tile_height-24))
        x=(i%columns)*tile_width;y=(i//columns)*tile_height
        canvas.paste(im,(x,y));draw.text((x+3,y+tile_height-20),key,fill='white')
    canvas.save(path)
