"""CPU-only check of the distributed notebook/source binding on each host OS."""
import ast,json,sys,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vvcrl.config import fingerprint
from vvcrl.repository import verify_checkout

def verify_runtime(root=ROOT,expected_source_sha256=None):
    """Verify the checked-out runtime, independently of distributed notebooks.

    Notebook pins are publication metadata, excluded from the functional hash.
    A runtime commit can legitimately carry older launchers. Do not compare
    their historical pins against this runtime's fingerprint.
    """
    root=Path(root)
    count=verify_checkout(root)
    extra=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard','--',
        'vvcrl','native','vendor','configs','scripts','tests'],text=True)
    if extra.strip():raise ValueError('Untracked functional source present')
    actual=fingerprint(root)
    if expected_source_sha256 is not None and actual!=expected_source_sha256:
        raise ValueError('Runtime functional fingerprint differs from requested source')
    print('RUNTIME_SOURCE_PASS',count,actual,flush=True)
    return actual

def verify_d_series(root=ROOT):
    """Check all current launchers together; historical B stays separately pinned."""
    root=Path(root);actual=fingerprint(root);pins=[]
    names=['VVC_RL_D_Series_01_Preflight_Kaggle.ipynb',
           'VVC_RL_D_Series_02_Train10_Kaggle.ipynb',
           'VVC_RL_D_Series_03_Resume30_Kaggle.ipynb',
           'VVC_RL_D_Series_Training_Kaggle.ipynb']
    for name in names:
        notebook=json.loads((root/'notebooks'/name).read_text(encoding='utf8'));values={};sources=[]
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code':continue
            source=cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
            tree=ast.parse(source);compile(tree,name+':'+str(i),'exec');sources.append(source)
            if cell.get('execution_count') is not None or cell.get('outputs'):raise ValueError('Published launcher contains execution artifacts')
            for node in tree.body:
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id in ('SOURCE_COMMIT','EXPECTED_SOURCE_SHA256','VARIANT','CREDIT_MODE'):
                            values[target.id]=ast.literal_eval(node.value)
        code='\n'.join(sources);commit=values.get('SOURCE_COMMIT','')
        if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('D-series requires immutable source pin')
        if values.get('EXPECTED_SOURCE_SHA256')!=actual:raise ValueError('D-series functional fingerprint differs: '+name)
        if values.get('VARIANT')!='D4' or values.get('CREDIT_MODE')!='full-clip':raise ValueError('D4 launchers must default to full-clip credit')
        if "'--gae-lambda',str(GAE_LAMBDA)" not in code or "GAE_LAMBDA = 1.0 if CREDIT_MODE == 'full-clip' else 0.95" not in code:
            raise ValueError('D4 preflight/train/resume estimator differs: '+name)
        if 'PPOConfig(gae_lambda=float(lam))' not in code or 'STATE_VERSION,REWARD_VERSION,FRAMES,GAE_LAMBDA' not in code:
            raise ValueError('Gate recheck ignores selected credit estimator: '+name)
        pins.append(commit)
    if len(set(pins))!=1:raise ValueError('D-series source commits differ across stages')
    print('D_SERIES_NOTEBOOKS_PASS',len(names),pins[0],actual,flush=True)
    return actual

def verify(root=ROOT):
    root=Path(root);notebook=json.loads((root/'notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb').read_text(encoding='utf-8'));assignments={}
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code':continue
        tree=ast.parse(cell['source'],filename=f'cell_{i}');compile(tree,f'cell_{i}','exec')
        for node in tree.body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ('EXPECTED_SOURCE_SHA256','RUN_MINI_TRAINING','SOURCE_COMMIT','REWARD_VERSION','EXPERIMENT','GAE_GAMMA'):assignments[target.id]=ast.literal_eval(node.value)
    commit=assignments.get('SOURCE_COMMIT','')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Notebook requires an exact source commit')
    if assignments.get('REWARD_VERSION')!='task-budget-v2':raise ValueError('Current notebook must default to V2 reward')
    actual=fingerprint(root)
    expected=assignments.get('EXPECTED_SOURCE_SHA256')
    # Historical B20 notebook stays immutable on experimental branches. Verify
    # its exact commit, not a silently substituted current checkout.
    pinned=actual
    if expected!=actual:
        if subprocess.run(['git','-C',str(root),'cat-file','-e',commit+'^{commit}'],capture_output=True).returncode:
            subprocess.run(['git','-C',str(root),'fetch','--no-tags','origin',commit],check=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pinned'
            subprocess.run(['git','-C',str(root),'worktree','add','--detach',str(path),commit],check=True)
            try:pinned=fingerprint(path)
            finally:subprocess.run(['git','-C',str(root),'worktree','remove','--force',str(path)],check=True)
    if expected!=pinned:raise ValueError('Published notebook pin differs from exact historical source fingerprint')
    if assignments.get('RUN_MINI_TRAINING') is not False:raise ValueError('Distributed notebook must default to preflight only')
    if assignments.get('EXPERIMENT')!='B' or assignments.get('GAE_GAMMA')!=1.0:
        raise ValueError('Distributed notebook must default to B with gamma=1')
    code='\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type']=='code')
    if code.count("'--gae-lambda',GAE_LAMBDA")!=2 or code.count("'--train-schedule',TRAIN_SCHEDULE")!=2:
        raise ValueError('Both preflight and training must pass the selected PPO/schedule settings')
    if "EXPERIMENT_SPECS={'A':('blocked',.95),'B':('mixed',.95),'C':('mixed',1.0)}" not in code:
        raise ValueError('Notebook A/B/C settings are not correctly declared')
    report=json.loads((root/'docs/VERIFICATION.json').read_text(encoding='utf-8'))
    if report['functional_source_sha256']!=pinned:raise ValueError('Verification report/source binding differs')
    print('NOTEBOOK_SOURCE_PIN_PASS',pinned,'CURRENT_FUNCTIONAL_SOURCE',actual,flush=True);return pinned

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Verify runtime bytes or the full published notebook distribution')
    parser.add_argument('--scope',choices=['runtime','distribution'],default='runtime')
    parser.add_argument('--expected-source-sha256')
    args=parser.parse_args()
    verify_runtime(expected_source_sha256=args.expected_source_sha256)
    if args.scope=='distribution':verify();verify_d_series()
