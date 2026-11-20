"""One-time source instrumentation. All normative coding changes remain the tested QG delta path."""
from pathlib import Path
import json,hashlib,shutil,sys
sys.path.insert(0,str(Path(__file__).parent))
from diagnostic_patch import patch,change

def apply(root):
    root=Path(root);vendor=root/'vendor/vvenc';evidence=root/'native/originals';manifest=root/'native/online_manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        for n,h in m['after'].items():
            if hashlib.sha256((vendor/n).read_bytes()).hexdigest()!=h:raise ValueError('Instrumented source differs: '+n)
        return m
    evidence.mkdir(parents=True,exist_ok=True)
    result=patch(vendor,root/'native/QpaProbe_original.h',evidence)
    names=['source/Lib/EncoderLib/EncSlice.cpp','source/Lib/EncoderLib/EncGOP.cpp']
    for n in names:
        shutil.copyfile(vendor/n,evidence/Path(n).name)
        result['before'][n]=hashlib.sha256((vendor/n).read_bytes()).hexdigest()
    q=vendor/'source/Lib/EncoderLib/QpaProbe.h';s=q.read_text(encoding='utf-8')
    s=change(s,'#include <vector>','#include <vector>\n#include "VvcRlSession.h"')
    s=change(s,'  if (!enabled()) return;','  if (!enabled()) return;\n  if(std::string(stage)=="activity") VvcRl::activity(poc,ctu,a);')
    s=change(s,'  static Actions a;if(a.values.empty())return 0;','  if(VvcRl::enabled())return VvcRl::delta(poc,x,y,w,h);\n  static Actions a;if(a.values.empty())return 0;')
    q.write_text(s,encoding='utf-8')
    n=names[0];p=vendor/n;s=p.read_text(encoding='utf-8')
    s=change(s,'#include "EncSlice.h"','#include "EncSlice.h"\n#include "VvcRlSession.h"')
    s=change(s,'  xInitSliceLambdaQP( slice );','  xInitSliceLambdaQP( slice );\n  VvcRl::before(*slice,*m_pcEncCfg);')
    p.write_text(s,encoding='utf-8')
    n=names[1];p=vendor/n;s=p.read_text(encoding='utf-8')
    s=change(s,'#include "EncGOP.h"','#include "EncGOP.h"\n#include "VvcRlSession.h"')
    marker='  const bool skipFirstPass = ( ! m_pcRateCtrl->rcIsFinalPass || m_isPreAnalysis ) && outPic->gopEntry->m_skipFirstPass;'
    s=change(s,marker,'  VvcRl::after(*outPic,auList);\n'+marker);p.write_text(s,encoding='utf-8')
    shutil.copyfile(root/'native/VvcRlSession.h',vendor/'source/Lib/EncoderLib/VvcRlSession.h')
    changed=list(result['before'])+['source/Lib/EncoderLib/QpaProbe.h','source/Lib/EncoderLib/VvcRlSession.h']
    result['after']={n:hashlib.sha256((vendor/n).read_bytes()).hexdigest() for n in changed}
    result['schema']='vvc-rl-online-patch/1'
    manifest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result
if __name__=='__main__':print(json.dumps(apply(Path(__file__).resolve().parents[1])))
