"""Generate a preflight-first notebook bound to this exact functional source."""
import json,sys,textwrap,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vvcrl.config import fingerprint
from vvcrl.repository import verify_checkout

def generate(source_commit):
    if len(source_commit)!=40 or any(c not in "0123456789abcdef" for c in source_commit):raise ValueError("Pass an exact source commit SHA")
    cells=[]
    def markdown(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()+'\n'))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(s).strip()+'\n'))
    markdown('''# VVC_RL — kiểm chứng QPA + PPO trước training
    Bật Internet và 2 GPU. Input: dataset chứa video BDD100K `.mov` gốc đã khóa; không cần checkpoint cũ.
    Notebook mặc định chỉ preflight: kiểm tra action/bitstream/decode, state thực, hai GPU và một PPO update cơ học trên Train.
    Chỉ đổi `RUN_MINI_TRAINING=True` sau khi xem đầy đủ gate PASS. Test30 chưa sử dụng.
    Source và thiết kế mới không tương thích checkpoint Gaussian/D3QN cũ.''')
    markdown('''## Semantic-region policy; unchanged V2 objective
    PPO learns two categorical residuals (source ROI/background), mapped to the 12 CTUs.
    QPA remains active. Both factors may choose -2,-1,0,+1,+2; no forced ROI protection.
    Inactive factors are marginalized; log-probability counts active factors once.
    Reward = mean F1 gain - price * max(0, (clip bits - cap)/target bits).
    cap = 1.02 * min(target bits, paired QPA bits). No spare-bit bonus below cap.
    The bitrate penalty is applied once at the last coding step. State/reward/GAE/Aux/native hooks are unchanged.
    Old V1/V2 update40 models are frozen comparisons, not resume inputs. Keep their exports.
    Mini budget: 40 updates = 80 clips = 20,480 transitions. No Test30.
    A fresh preflight is required because the policy/action distribution changed.
    Set RUN_MINI_TRAINING=True only when ready to run the mini after all new gates pass.
    The preflight uses disposable weights; mini always starts from the same seed.
    Default tolerance is 2%; set RATE_TOLERANCE=0 before preflight for a strict cap.
    A soft penalty is not a hard RC guarantee: validation rejects checkpoints over cap.''')
    code(f'''
    from pathlib import Path
    import os,sys,time,json,subprocess,threading
    SESSION_STARTED=time.monotonic()
    SOURCE_COMMIT='{source_commit}'
    EXPECTED_SOURCE_SHA256='{fingerprint(ROOT)}'
    REPOSITORY='https://github.com/youngan2810/VVC_RL.git'
    REPO=Path('/kaggle/working/VVC_RL')
    TOOLS=Path('/kaggle/working/vvc_rl_tools')
    INPUT=Path('/kaggle/input')
    FRAMES=256
    RUN_MINI_TRAINING=False
    REWARD_VERSION='task-budget-v2'
    RATE_TOLERANCE=.02
    SEED=20261120
    ADDITIONAL_PPO_UPDATES=40  # 2 complete 256-frame clips per update; not epochs
    RESUME_CHECKPOINT=''      # only a checkpoint from this exact implementation + identity JSON
    SESSION_HOURS=11.0        # includes setup; clips finish before soft stop when possible
    assert 0<SESSION_HOURS<=11.75
    PREFLIGHT=Path('/kaggle/working')/('vvc_rl_preflight_'+str(time.time_ns()))
    TRAIN=Path('/kaggle/working/vvc_rl_mini_regions')
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
    live(['git','-C',REPO,'fetch','--no-tags','origin',SOURCE_COMMIT])
    commit=SOURCE_COMMIT
    live(['git','-C',REPO,'checkout','--detach',commit])
    print('CHECKED_OUT_RAW_BYTES_PASS',verify_checkout(REPO))
    sys.path.insert(0,str(REPO))
    # A failed earlier bootstrap may have imported config from the old commit.
    import importlib
    import vvcrl.config as _source_config
    importlib.reload(_source_config)
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
    live([sys.executable,'-u','-m','vvcrl','preflight','--output',PREFLIGHT,'--toolchain',TOOLS/'TOOLCHAIN.json','--input',INPUT,'--tools',TOOLS,'--frames',FRAMES,'--reward-version',REWARD_VERSION,'--rate-tolerance',RATE_TOLERANCE,'--seed',SEED],REPO)
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
                 '--preflight',PREFLIGHT,'--frames',FRAMES,'--updates',ADDITIONAL_PPO_UPDATES,'--hours',remaining,'--reward-version',REWARD_VERSION,'--rate-tolerance',RATE_TOLERANCE,'--seed',SEED,'--start-mini-training']
        if RESUME_CHECKPOINT:command+=['--resume',RESUME_CHECKPOINT]
        live(command,REPO)
    else:print('MINI_TRAINING_DISABLED — review preflight results first')
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'kaggle':{'isGpuEnabled':True,'isInternetEnabled':True}},'nbformat':4,'nbformat_minor':5}
    output=ROOT/'notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb';output.parent.mkdir(exist_ok=True);output.write_bytes((json.dumps(notebook,ensure_ascii=False,indent=1)+'\n').encode('utf-8'));return output

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);a=p.parse_args();print(generate(a.source_commit))
