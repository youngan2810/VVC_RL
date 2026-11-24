"""CPU-only check of the distributed notebook/source binding on each host OS."""
import ast,json,sys,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vvcrl.config import fingerprint

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

if __name__=='__main__':verify()
