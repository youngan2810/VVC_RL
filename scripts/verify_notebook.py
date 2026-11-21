"""CPU-only check of the distributed notebook/source binding on each host OS."""
import ast,json,sys
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
                    if isinstance(target,ast.Name) and target.id in ('EXPECTED_SOURCE_SHA256','RUN_MINI_TRAINING','SOURCE_COMMIT','REWARD_VERSION'):assignments[target.id]=ast.literal_eval(node.value)
    commit=assignments.get('SOURCE_COMMIT','')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Notebook requires an exact source commit')
    if assignments.get('REWARD_VERSION')!='task-budget-v2':raise ValueError('Current notebook must default to V2 reward')
    actual=fingerprint(root)
    if assignments.get('EXPECTED_SOURCE_SHA256')!=actual:raise ValueError('Published notebook pin differs from this host source fingerprint')
    if assignments.get('RUN_MINI_TRAINING') is not False:raise ValueError('Distributed notebook must default to preflight only')
    report=json.loads((root/'docs/VERIFICATION.json').read_text(encoding='utf-8'))
    if report['functional_source_sha256']!=actual:raise ValueError('Verification report/source binding differs')
    print('NOTEBOOK_SOURCE_PIN_PASS',actual,flush=True);return actual

if __name__=='__main__':verify()
