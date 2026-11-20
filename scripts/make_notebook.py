"""Generate a preflight-first notebook bound to this exact functional source."""
import json,sys,textwrap,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vvcrl.config import fingerprint
from vvcrl.repository import verify_checkout

def generate():
    cells=[]
    def markdown(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(s).strip()+'\n'))
    markdown('''# VVC_RL — kiểm chứng QPA + PPO trước training
    Bật Internet và 2 GPU. Input: dataset chứa video BDD100K `.mov` gốc đã khóa; không cần checkpoint cũ.
    Notebook mặc định chỉ preflight: kiểm tra action/bitstream/decode, state thực, hai GPU và một PPO update cơ học trên Train.
    Chỉ đổi `RUN_MINI_TRAINING=True` sau khi xem đầy đủ gate PASS. Test30 chưa sử dụng.
    Source và thiết kế mới không tương thích checkpoint Gaussian/D3QN cũ.''')
    code(f'''
    from pathlib import Path
    import os,sys,time,json,subprocess,threading
    SESSION_STARTED=time.monotonic()
    EXPECTED_SOURCE_SHA256='{fingerprint(ROOT)}'
    REPOSITORY='https://github.com/youngan2810/VVC_RL.git'
    REPO=Path('/kaggle/working/VVC_RL')
    TOOLS=Path('/kaggle/working/vvc_rl_tools')
    INPUT=Path('/kaggle/input')
    FRAMES=256
    RUN_MINI_TRAINING=False
    ADDITIONAL_PPO_UPDATES=10  # 2 complete 256-frame clips per update; not epochs
    RESUME_CHECKPOINT=''      # only a checkpoint from this exact implementation + identity JSON
    SESSION_HOURS=11.0        # includes setup; clips finish before soft stop when possible
    assert 0<SESSION_HOURS<=11.75
    PREFLIGHT=Path('/kaggle/working')/('vvc_rl_preflight_'+str(time.time_ns()))
    TRAIN=Path('/kaggle/working/vvc_rl_mini')
    os.environ['YOLOv5_AUTOINSTALL']='false'
    def live(command,cwd=None):
        print('$',*[str(v) for v in command],flush=True)
        process=subprocess.Popen(list(map(str,command)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        last=[time.monotonic()]
        def heartbeat():
            while process.poll() is None:
                time.sleep(15)
                if process.poll() is None and time.monotonic()-last[0]>30:print('RUNNING — elapsed',round((time.monotonic()-SESSION_STARTED)/60,1),'min',flush=True)
        threading.Thread(target=heartbeat,daemon=True).start()
        try:
            for line in process.stdout:last[0]=time.monotonic();print(line,end='',flush=True)
            if process.wait():raise RuntimeError('Command failed; keep logs and SAFE export. Do not remove assertions.')
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=30)
    ''')
    code('import hashlib\n'+inspect.getsource(verify_checkout)+'\n'+textwrap.dedent('''
    import torch
    print('PYTORCH',torch.__version__,'CUDA',torch.version.cuda,'GPU_COUNT',torch.cuda.device_count())
    assert torch.cuda.is_available() and torch.cuda.device_count()>=2,'Enable two Kaggle GPUs'
    if not REPO.exists():live(['git','clone','--branch','main',REPOSITORY,REPO])
    print('TRACKED_RAW_BYTES_PASS',verify_checkout(REPO))
    live(['git','-C',REPO,'fetch','origin','main'])
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','origin/main'],text=True).strip()
    live(['git','-C',REPO,'checkout','--detach',commit])
    print('CHECKED_OUT_RAW_BYTES_PASS',verify_checkout(REPO))
    sys.path.insert(0,str(REPO))
    from vvcrl.config import fingerprint,atomic_json
    assert fingerprint(REPO)==EXPECTED_SOURCE_SHA256,'Functional source changed: obtain the matching notebook; do not overwrite this hash'
    print('EXACT_FUNCTIONAL_SOURCE_PASS',commit,EXPECTED_SOURCE_SHA256)
    TOOLS.mkdir(parents=True,exist_ok=True)
    atomic_json(TOOLS/'SOURCE_IDENTITY.json',{'commit':commit,'functional_sha256':EXPECTED_SOURCE_SHA256})
    '''))
    code('''
    # Verify the bundled lock and all 16 Train + 4 Validation videos before builds.
    from vvcrl.locked.data import VideoCatalog
    catalog=VideoCatalog(INPUT,mode='mini',seed=20261111)
    catalog.check_available(('train','val'))
    for split in ('train','val'):
        for video in catalog.selected[split]:catalog.source(video)  # size + SHA256, no result-based selection
    print('LOCKED_MINI_DATASET_PASS',catalog.selected)
    live([sys.executable,'-m','unittest','discover','-s','tests','-v'],REPO)
    ''')
    code('''
    # Builds vanilla/instrumented VVenC and pinned VVdeC; checks frozen YOLO source/weights.
    live([sys.executable,'-u',REPO/'scripts/setup_kaggle.py','--tools',TOOLS,'--jobs','2'],REPO)
    ''')
    code('''
    # Full preflight uses real Train clips for one mechanical PPO update and Val without updates.
    # Synthetic native tests alone never open the training gate.
    live([sys.executable,'-u','-m','vvcrl','preflight','--output',PREFLIGHT,'--toolchain',TOOLS/'TOOLCHAIN.json','--input',INPUT,'--tools',TOOLS,'--frames',FRAMES],REPO)
    report=json.loads((PREFLIGHT/'PREFLIGHT_GATE.json').read_text())
    print('PREFLIGHT_STATUS',report['status'])
    assert report['status']=='PASS_READY_FOR_MINI'
    print('PREFLIGHT_DIR',PREFLIGHT)
    print('Download /kaggle/working/VVC_RL_PREFLIGHT_KAGGLE_SAFE.zip and send it for review before mini training.')
    ''')
    code('''
    if RUN_MINI_TRAINING:
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_STARTED)/3600
        assert remaining>.5,'Insufficient session time; preserve SAFE and use a fresh session'
        command=[sys.executable,'-u','-m','vvcrl','train-mini','--output',TRAIN,'--toolchain',TOOLS/'TOOLCHAIN.json','--input',INPUT,'--tools',TOOLS,
                 '--preflight',PREFLIGHT,'--frames',FRAMES,'--updates',ADDITIONAL_PPO_UPDATES,'--hours',remaining,'--start-mini-training']
        if RESUME_CHECKPOINT:command+=['--resume',RESUME_CHECKPOINT]
        live(command,REPO)
    else:print('MINI_TRAINING_DISABLED — review preflight results first')
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'kaggle':{'isGpuEnabled':True,'isInternetEnabled':True}},'nbformat':4,'nbformat_minor':5}
    output=ROOT/'notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb';output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');return output

if __name__=='__main__':print(generate())
