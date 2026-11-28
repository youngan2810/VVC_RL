from pathlib import Path
import hashlib,json,time
import numpy as np
from .config import FB,atomic_json,digest,canonical
from .locked.data import VideoCatalog,FrozenDetector

DETECTOR_ID={'name':'YOLOv5n-v7-FP32','source':'915bbf294bb74c859f0b41f1c23bc395014ea679','weights':'4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3','conf':.25,'nms':.45,'size':640,'class':2,'color':'locked-BT709-TV-libswscale'}

def open_detector(tools,device):
    tools=Path(tools)
    return FrozenDetector(tools/'yolov5',tools/'yolov5n.pt',device)

def prepared_clip(input_root,cache,video,frames,detector,split):
    catalog=VideoCatalog(input_root,mode='mini',seed=20261111)
    if split not in ('train','val') or video not in catalog.selected[split]:raise ValueError('Video is not in the locked permitted split')
    source=catalog.source(video)
    cache_contract={'video':video,'split':split,'mov':digest(source),'frames':frames,'start':0,'detector':DETECTOR_ID,'preprocessing':catalog.lock['preprocessing']}
    key=hashlib.sha256(canonical(cache_contract).encode()).hexdigest()
    folder=Path(cache)/key;folder.mkdir(parents=True,exist_ok=True);yuv=folder/'source.yuv';meta=folder/'cache.json'
    if meta.exists():
        m=json.loads(meta.read_text())
        if m.get('contract')!=cache_contract or yuv.stat().st_size!=frames*FB or digest(yuv)!=m['source_sha256'] or digest(folder/'boxes.json')!=m['boxes_sha256']:raise ValueError('Source/teacher cache corrupted or incompatible')
        boxes=json.loads((folder/'boxes.json').read_text())
        if len(boxes)!=frames:raise ValueError('Source teacher cache POC count mismatch')
        return yuv,boxes,key
    started=time.perf_counter()
    if catalog.prepare(video,yuv,frames)!=frames:raise ValueError('Locked source window shorter than required')
    preprocess_seconds=time.perf_counter()-started;started=time.perf_counter()
    raw=np.memmap(yuv,dtype=np.uint8,mode='r',shape=(frames,FB))
    try:boxes=[detector(raw[p]).tolist() for p in range(frames)]
    finally:raw._mmap.close()
    atomic_json(folder/'boxes.json',boxes)
    source_detector_seconds=time.perf_counter()-started
    atomic_json(meta,{'contract':cache_contract,'video':video,'split':split,'source_sha256':digest(yuv),'boxes_sha256':digest(folder/'boxes.json'),'frames':frames,'detector':DETECTOR_ID,
      'timing':{'preprocess_seconds':preprocess_seconds,'source_detector_ms_per_frame':source_detector_seconds*1000/frames,'source_detector_timing_note':'Includes conversion/inference/host transfer; cached runs reuse these measurements'}})
    return yuv,boxes,key


# Authenticated split-specific labels: Train reward or Validation selection.
# Never included in Actor observations; the loader rejects Test.
import collections,re
from .config import VALIDATION_SELECTION,W,H,canonical
from .locked.data import ASSETS

def normalize(s):return re.sub('[^a-z0-9]','',str(s).lower())


def label_columns(df):
    """Accept known flattened BDD schemas; reject unknown/ambiguous schemas."""
    lookup=collections.defaultdict(list)
    for c in df.columns:lookup[normalize(c)].append(c)
    def one(aliases):
        found=list(dict.fromkeys(c for k in aliases for c in lookup.get(k,[])))
        if len(found)!=1:raise ValueError('Unknown/ambiguous annotation columns '+str(aliases)+'; actual='+str(list(df.columns)))
        return found[0]
    columns={'frame':one(['frameindex','frameidx','frameid']),
             'category':one(['category','class','classname'])}
    for name in ['x1','y1','x2','y2']:columns[name]=one([name,'box2d'+name,'bbox'+name])
    choices=[c for c in df.columns if normalize(c) in ['video','videoid','videoname','name','filename','image','imagepath']]
    video=[]
    for c in choices:
        sample=df[c].dropna().astype(str).head(1000)
        if len(sample) and sample.str.contains(r'[0-9a-f]{8}-[0-9a-f]{8}',regex=True).mean()>.9:video.append(c)
    # Prefer explicit video fields if both frame name and video name are present.
    explicit=[c for c in video if normalize(c) in ['video','videoid','videoname']]
    if explicit:video=explicit
    if len(video)!=1:raise ValueError('Cannot unambiguously identify BDD video column: '+str(video))
    columns['video']=video[0]
    return columns



def validation_ground_truth(input_root,frames):
    return task_ground_truth(input_root,frames,'val')

def training_ground_truth(input_root,frames):
    return task_ground_truth(input_root,frames,'train')

def task_ground_truth(input_root,frames,split):
    """Authenticate the original annotation file, then select only the locked split.

    Missing/all-null labels are unknown, not negative examples. Object-annotated
    frames with no cars ARE negatives. No interpolation or Test result import.
    """
    import pandas as pd
    from .config import TRAINING_REWARD_PROTOCOL
    if split not in ('train','val'):raise ValueError('GT loader accepts locked Train/Validation only; never Test')
    task_protocol=TRAINING_REWARD_PROTOCOL if split=='train' else VALIDATION_SELECTION
    lock=json.loads((ASSETS/'dataset_lock.json').read_text(encoding='utf8'))
    protocol=lock['annotation_protocol']
    expected=VALIDATION_SELECTION['annotation_sha256']
    if (protocol['source_file_sha256']!=expected or not protocol['manual_qa_confirmed']
            or protocol['annotation_fps_assumption']!=5 or protocol['bbox_coordinate_mode']!='display'):
        raise ValueError('Unsupported GT annotation mapping')
    paths=sorted(p for p in Path(input_root).rglob('mot_labels.parquet') if digest(p)==expected)
    if not paths:raise FileNotFoundError('Add Driving Video with Object Tracking including the locked mot_labels.parquet; teacher boxes/Test SAFE are not authenticated task GT')
    df=pd.read_parquet(paths[0])
    if any(x is not None for x in df.index.names):df=df.reset_index()
    if 'box2d' in df.columns:
        def parse(x):
            if isinstance(x,dict):return x
            if x is None or (isinstance(x,float) and np.isnan(x)):return {}
            return json.loads(x) if isinstance(x,str) else dict(x)
        b=pd.json_normalize(df['box2d'].map(parse));b.index=df.index
        df=df.drop(columns=['box2d']).join(b.add_prefix('box2d.'))
    cols=label_columns(df)
    ids=VideoCatalog(input_root,mode='mini',seed=20261111).selected[split]
    if len(ids)!=(16 if split=='train' else 4):raise ValueError('GT selection differs from locked split size')
    vids=df[cols['video']].astype(str).str.extract(r'([0-9a-f]{8}-[0-9a-f]{8})',expand=False)
    df=df.loc[vids.isin(ids)].copy();df['_vid']=vids.loc[df.index]
    records={x['video_id']:x for x in lock['video_records']}
    result={}
    for vi,vid in enumerate(ids):
        rows=[]
        for index,group in df[df['_vid']==vid].groupby(cols['frame'],dropna=False):
            if pd.isna(index) or float(index)!=int(index) or int(index)<0:raise ValueError('Malformed GT frame index')
            poc=int(index)*6
            if poc>=frames:continue
            objects=[]
            for _,row in group.iterrows():
                box=[row[cols[k]] for k in ('x1','y1','x2','y2')];category=row[cols['category']]
                missing=[pd.isna(x) for x in box]
                if pd.isna(category) and all(missing):continue
                if pd.isna(category) or any(missing):raise ValueError('Malformed nonempty annotation')
                box=np.asarray(box,float)
                if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid GT box')
                objects.append((str(category),box))
            if not objects:continue
            dw,dh=records[vid]['display_size'];cars=[]
            for category,box in objects:
                if category!='car':continue
                box=box*np.array([W/dw,H/dh,W/dw,H/dh])
                box[[0,2]]=np.clip(box[[0,2]],0,W);box[[1,3]]=np.clip(box[[1,3]],0,H)
                if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Car box outside display')
                if box.tolist() in cars:raise ValueError('Duplicate car annotation')
                cars.append(box.tolist())
            rows.append(dict(image_id=(10000000 if split=='train' else 0)+vi*100000+poc+1,video_id=vid,poc=poc,boxes=cars))
        if len(rows)<min(10,len(range(0,frames,6))):raise ValueError('Insufficient explicit '+split+' labels: '+vid)
        rows=sorted(rows,key=lambda x:x['poc'])
        if not any(x['boxes'] for x in rows):raise ValueError('No car GT in '+split+' video: '+vid)
        result[vid]={'protocol':task_protocol,'frames':rows,'gt_sha256':hashlib.sha256(canonical(rows).encode()).hexdigest()}
    return result


def validation_ap_prediction(detector,yuv8):
    """Sequential low-conf AP inference; ALWAYS restore the teacher threshold.

    One detector belongs to one collector. No background thread shares this model.
    The .25 teacher state/auxiliary inference has already completed for this frame.
    """
    model=detector.model
    if (model.conf,model.iou,model.max_det,model.classes,model.amp)!=(.25,.45,300,[2],False):
        raise ValueError('Teacher/AP detector settings differ from locked protocol')
    try:
        model.conf=.001
        return detector(yuv8).tolist()
    finally:model.conf=.25
