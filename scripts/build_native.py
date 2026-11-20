"""Build restored upstream anchor and instrumented encoder from the same tree."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vvcrl.config import digest,atomic_json

def run(args,log,cwd=None):
    print('BUILD',*[str(a) for a in args],flush=True)
    with Path(log).open('wb') as f:subprocess.run(list(map(str,args)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)

def copy_executable(source,target):
    # copyfile discards POSIX execute bits, leaving a successfully built encoder
    # unusable on Linux. Preserve metadata and explicitly verify executability.
    shutil.copy2(source,target)
    if os.name!='nt' and not os.access(target,os.X_OK):raise PermissionError('Copied binary is not executable: '+str(target))

def build(jobs=2):
    build=ROOT/'_build';build.mkdir(exist_ok=True);out=build/'bin';out.mkdir(exist_ok=True)
    vendor=ROOT/'vendor/vvenc';manifest=json.loads((ROOT/'native/online_manifest.json').read_text())
    for n,h in manifest['after'].items():
        if digest(vendor/n)!=h:raise ValueError('Unverified native source: '+n)
    upstream=json.loads((ROOT/'upstream_tree.json').read_text())
    for entry in upstream['tree']:
        if entry['type']!='blob' or entry['path'] in manifest['after']:continue
        content=(vendor/entry['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=entry['sha']:raise ValueError('Vendored upstream blob changed: '+entry['path'])
    vanilla=build/'vanilla_source'
    if not vanilla.exists():
        shutil.copytree(vendor,vanilla,ignore=shutil.ignore_patterns('bin','lib','build','.git','__pycache__'))
        for n,h in manifest['before'].items():
            original=ROOT/'native/originals'/('vvenc_gitattributes.original' if n=='.gitattributes' else Path(n).name)
            if digest(original)!=h:raise ValueError('Original native source corrupted')
            shutil.copyfile(original,vanilla/n)
        for n in set(manifest['after'])-set(manifest['before']):(vanilla/n).unlink()
    # Check the restored anchor every build, including resumed setup.
    for n,h in manifest['before'].items():
        if digest(vanilla/n)!=h:raise ValueError('Restored anchor differs')
    extra=[];suffix='.exe' if os.name=='nt' else ''
    if os.name=='nt':
        compiler=Path('C:/msys64/ucrt64/bin');os.environ['PATH']=str(compiler)+';'+os.environ.get('PATH','')
        extra=['-G','MinGW Makefiles','-DCMAKE_C_COMPILER='+str(compiler/'gcc.exe'),'-DCMAKE_CXX_COMPILER='+str(compiler/'g++.exe'),'-DCMAKE_MAKE_PROGRAM='+str(compiler/'mingw32-make.exe')]
    for name,source in [('vanilla',vanilla),('online',vendor)]:
        directory=build/('portable_'+name)
        run(['cmake','-S',source,'-B',directory,'-DCMAKE_BUILD_TYPE=Release',*extra],build/(name+'_configure.log'))
        run(['cmake','--build',directory,'--config','Release','--target','vvencapp','-j',str(jobs)],build/(name+'_build.log'))
        target=out/('vvenc_'+('rl' if name=='online' else 'vanilla')+suffix)
        copy_executable(source/'bin/release-static'/('vvencapp'+suffix),target)
    return {'encoder':str((out/('vvenc_rl'+suffix)).resolve()),'vanilla':str((out/('vvenc_vanilla'+suffix)).resolve())}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=2);a=p.parse_args();print(json.dumps(build(a.jobs),indent=2))
