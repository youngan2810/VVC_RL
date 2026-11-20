from pathlib import Path
import hashlib,io,json,zipfile
from .config import atomic_json,digest

def export_run(root,destination=None):
    root=Path(root);destination=Path(destination) if destination else root.parent/'VVC_RL_PREFLIGHT_KAGGLE_SAFE.zip'
    if destination.is_relative_to(root):raise ValueError('Export must live outside run directory')
    payload=destination.with_name(destination.stem+'_payload.bin');tmp=payload.with_suffix('.tmp');files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if p.is_file() and p.suffix not in ('.yuv','.tmp','.pyc') and '__pycache__' not in rel.parts and 'recon' not in rel.parts and 'source_cache' not in rel.parts and p.name!='EXPORT_MANIFEST.json':files[rel.as_posix()]=digest(p)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for n in files:z.write(root/n,n)
        z.writestr('EXPORT_MANIFEST.json',json.dumps({'schema':'vvc-rl-safe/1','files':files},indent=2))
    tmp.replace(payload);tmp=destination.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:z.write(payload,'VVC_RL_payload.bin');z.writestr('payload_sha256.txt',digest(payload))
    tmp.replace(destination);print('SAFE_EXPORT_READY',destination,flush=True);return destination

def import_run(package,destination):
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise ValueError('Resume destination must be empty')
    package=Path(package)
    if package.suffix=='.bin':raw=package.read_bytes()
    else:
        with zipfile.ZipFile(package) as outer:
            raw=outer.read('VVC_RL_payload.bin')
            if hashlib.sha256(raw).hexdigest()!=outer.read('payload_sha256.txt').decode().strip():raise ValueError('Outer SAFE digest mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        m=json.loads(z.read('EXPORT_MANIFEST.json'));names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(m['files'])|{'EXPORT_MANIFEST.json'}:raise ValueError('SAFE manifest mismatch')
        validated=[]
        for n,h in m['files'].items():
            p=(destination/n).resolve()
            if n.startswith(('/','\\')) or ':' in n or '..' in n.replace('\\','/').split('/') or not p.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
            b=z.read(n)
            if hashlib.sha256(b).hexdigest()!=h:raise ValueError('Corrupt SAFE member')
            validated.append((p,b))
        for p,b in validated:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    return destination
