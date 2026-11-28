"""Pinned native + detector toolchain. Does not run a training session."""
from pathlib import Path
import argparse,json,os,subprocess,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from build_native import build,run
from vvcrl.config import atomic_json,digest
from vvcrl.data import DETECTOR_ID

DECODER='649f0b2fafee977c998d7e4d674f8b88d952e3a3'

def pinned_repo(url,path,commit):
    if not (path/'.git').is_dir():subprocess.run(['git','clone',url,str(path)],check=True)
    subprocess.run(['git','-C',str(path),'fetch','--no-tags','origin',commit],check=True)
    head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if head!=commit:
        if subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Dependency has local edits: '+str(path))
        subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=commit:raise ValueError('Dependency pin mismatch')

def setup(tools,jobs=2):
    if os.name=='nt':raise RuntimeError('Kaggle setup requires Linux; see build_native.py for local Windows encoder build')
    tools=Path(tools).resolve();tools.mkdir(parents=True,exist_ok=True)
    subprocess.run(['apt-get','update','-qq'],check=True)
    subprocess.run(['apt-get','install','-y','-qq','cmake','g++','ffmpeg','pkg-config','libswscale-dev','libavutil-dev'],check=True)
    # Preserve Kaggle PyTorch/CUDA. Install detector dependencies before workers start.
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','setuptools==80.9.0','gitpython==3.1.45','requests','PyYAML','pandas','pyarrow','pycocotools==2.0.11','seaborn','scipy','pillow','tqdm','psutil','opencv-python-headless'],check=True)
    binaries=build(jobs)
    decoder=tools/'vvdec';pinned_repo('https://github.com/fraunhoferhhi/vvdec.git',decoder,DECODER)
    run(['cmake','-S',decoder,'-B',decoder/'build','-DCMAKE_BUILD_TYPE=Release'],tools/'decoder_configure.log')
    run(['cmake','--build',decoder/'build','--config','Release','--target','vvdecapp','-j',jobs],tools/'decoder_build.log')
    binaries['decoder']=str(decoder/'bin/release-static/vvdecapp')
    pinned_repo('https://github.com/ultralytics/yolov5.git',tools/'yolov5',DETECTOR_ID['source'])
    weights=tools/'yolov5n.pt'
    if not weights.exists():
        tmp=weights.with_suffix('.download');urllib.request.urlretrieve('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',tmp);tmp.replace(weights)
    if digest(weights)!=DETECTOR_ID['weights']:raise ValueError('Detector weights digest differs from locked baseline')
    atomic_json(tools/'TOOLCHAIN.json',binaries);print('PINNED_TOOLCHAIN_READY',json.dumps(binaries),flush=True);return binaries

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tools',required=True);p.add_argument('--jobs',type=int,default=2);a=p.parse_args();setup(a.tools,a.jobs)
