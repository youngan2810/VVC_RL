"""Verify every vendored upstream Git blob plus the explicit native patch."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vvcrl.config import digest

def verify():
    manifest=json.loads((ROOT/'native/online_manifest.json').read_text());tree=json.loads((ROOT/'upstream_tree.json').read_text());checked=0
    if tree['sha']!=manifest['commit']:raise ValueError('Upstream tree pin differs')
    for x in tree['tree']:
        if x['type']!='blob':continue
        path=ROOT/'vendor/vvenc'/x['path'];b=path.read_bytes()
        if x['path'] in manifest['after']:
            if digest(path)!=manifest['after'][x['path']]:raise ValueError('Native patch changed: '+x['path'])
        elif hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=x['sha']:raise ValueError('Upstream blob changed: '+x['path'])
        checked+=1
    for n,h in manifest['after'].items():
        if digest(ROOT/'vendor/vvenc'/n)!=h:raise ValueError('Added native header differs')
    for n,h in manifest['before'].items():
        if digest(ROOT/'native/originals'/Path(n).name)!=h:raise ValueError('Original source differs')
    if (ROOT/'native/VvcRlSession.h').read_bytes()!=(ROOT/'vendor/vvenc/source/Lib/EncoderLib/VvcRlSession.h').read_bytes():raise ValueError('Online header copies differ')
    return {'upstream_commit':tree['sha'],'upstream_blobs_checked':checked,'modified_or_added_files':len(manifest['after'])}

if __name__=='__main__':print(json.dumps(verify()))
