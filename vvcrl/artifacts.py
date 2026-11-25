from pathlib import Path
import hashlib,json,zipfile,tempfile,shutil,stat,os
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

def _copy_hash(src,dst):
    h=hashlib.sha256()
    for chunk in iter(lambda:src.read(1024*1024),b''):
        h.update(chunk);dst.write(chunk)
    return h.hexdigest()

def _safe_name(n):
    return isinstance(n,str) and bool(n) and not any(c in n for c in ('\\',':','\0')) and all(p not in ('','.','..') for p in n.split('/'))

def import_run(package,destination):
    """Bound memory by manifest size + streaming buffer, not payload size.
    Verify and stage everything on the destination filesystem before publish.
    Raw .bin has per-member integrity only, as in the historical importer.
    """
    destination=Path(destination);package=Path(package)
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError('Resume destination must be empty')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.safe-import-',dir=destination.parent) as td:
        td=Path(td);stage=td/'staged';stage.mkdir()
        if package.suffix=='.bin':payload=package
        else:
            payload=td/'payload.bin'
            with zipfile.ZipFile(package) as outer:
                if sorted(outer.namelist())!=['VVC_RL_payload.bin','payload_sha256.txt']:raise ValueError('Outer SAFE manifest mismatch')
                if outer.getinfo('payload_sha256.txt').file_size>128:raise ValueError('Invalid outer digest')
                expected=outer.read('payload_sha256.txt').decode().strip()
                with outer.open('VVC_RL_payload.bin') as src,payload.open('wb') as dst:
                    if _copy_hash(src,dst)!=expected:raise ValueError('Outer SAFE digest mismatch')
        with zipfile.ZipFile(payload) as z:
            m=json.loads(z.read('EXPORT_MANIFEST.json'));names=z.namelist()
            if m.get('schema')!='vvc-rl-safe/1' or not isinstance(m.get('files'),dict) or len(names)!=len(set(names)) or set(names)!=set(m['files'])|{'EXPORT_MANIFEST.json'}:raise ValueError('SAFE manifest mismatch')
            if 'EXPORT_MANIFEST.json' in m['files']:raise ValueError('Manifest cannot include itself')
            seen=set()
            for n,h in m['files'].items():
                if not _safe_name(n) or n.casefold() in seen or not isinstance(h,str) or len(h)!=64:raise ValueError('Unsafe archive path or digest')
                seen.add(n.casefold())
                info=z.getinfo(n)
                if info.is_dir() or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive member')
                p=stage/n
                if not p.resolve().is_relative_to(stage.resolve()):raise ValueError('Unsafe archive path')
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,p.open('xb') as dst:
                    if _copy_hash(src,dst)!=h:raise ValueError('Corrupt SAFE member')
        if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):raise ValueError('Resume destination changed')
        if destination.exists():destination.rmdir()
        stage.replace(destination)
    return destination
