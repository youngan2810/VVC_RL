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
