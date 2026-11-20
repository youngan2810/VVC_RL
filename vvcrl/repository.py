"""Verify tracked working-tree bytes against Git objects, without clean filters."""
import hashlib,subprocess
from pathlib import Path

def verify_checkout(repo,ref='HEAD'):
    repo=Path(repo).resolve()
    raw=subprocess.check_output(['git','-C',str(repo),'ls-tree','-rz','--full-tree',ref])
    failures=[];count=0
    for record in raw.split(b'\0'):
        if not record:continue
        header,name=record.split(b'\t',1);mode,kind,want=header.split();name=name.decode('utf-8');p=(repo/name).resolve()
        if kind!=b'blob' or not p.is_relative_to(repo) or not p.is_file():failures.append(name);continue
        content=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest().encode()
        if actual!=want:failures.append(name)
        count+=1
    if failures or not count:raise ValueError('Tracked source bytes differ from '+ref+': '+repr(failures[:20]))
    return count
