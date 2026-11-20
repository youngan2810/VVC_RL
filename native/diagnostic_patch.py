"""Narrow, fail-closed instrumentation for one pinned official VVenC commit."""
from pathlib import Path
import hashlib,json,argparse
COMMIT='1a4f83bd9feeb91c6792fb8e0676b1c8f3683d4f'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def change(text,old,new):
    if text.count(old)!=1:raise ValueError('Patch anchor not unique: '+old[:100])
    return text.replace(old,new,1)
def patch(repo,header,output):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan={}
    names=['source/Lib/EncoderLib/BitAllocation.cpp','source/Lib/EncoderLib/EncCu.cpp','source/Lib/EncoderLib/EncPicture.cpp','source/Lib/CommonLib/Quant.cpp']
    for n in names:
        if 'QpaProbe::' in (repo/n).read_text(encoding='utf-8'):raise ValueError('Probe already present; verify saved patch manifest instead of applying twice')
    def read(name):return (repo/name).read_text(encoding='utf-8')
    name='source/Lib/EncoderLib/BitAllocation.cpp';s=read(name)
    s=change(s,'#include "BitAllocation.h"','#include "BitAllocation.h"\n#include "QpaProbe.h"')
    s=change(s,'pic->ctuQpaLambda[ctuRsAddr] = hpEner[1]; // temporary backup',
      'QpaProbe::row("activity",slice->poc,int(ctuRsAddr),ctuArea.x,ctuArea.y,ctuArea.width,ctuArea.height,0,0,0,0,hpEner[1],spVisActCTU);\n        pic->ctuQpaLambda[ctuRsAddr] = hpEner[1]; // temporary backup')
    s=change(s,'int adaptedLumaQP = Clip3 (minQP, MAX_QP, sliceQP + apprI3Log2 (hpEnerCTU * hpEnerPicNorm, isSccStrongRC));',
      'int adaptedLumaQP = Clip3 (minQP, MAX_QP, sliceQP + apprI3Log2 (hpEnerCTU * hpEnerPicNorm, isSccStrongRC));\n      QpaProbe::row("activity_qp",slice->poc,int(ctuRsAddr),0,0,0,0,sliceQP,adaptedLumaQP,0,0,hpEnerCTU,hpEnerPicNorm);')
    s=change(s,'if (encCfg->m_cuQpDeltaSubdiv == 0 || !slice->isIntra()) averageAdaptedLumaQP += applyDeltaQpPeakSmoothing (pic, encCfg, ctuStartAddr, ctuBoundingAddr);',
      'for(uint32_t q=ctuStartAddr;q<ctuBoundingAddr;++q) QpaProbe::row("before_smoothing",slice->poc,int(q),0,0,0,0,pic->ctuAdaptedQP[q],pic->ctuAdaptedQP[q],pic->ctuQpaLambda[q],pic->ctuQpaLambda[q]);\n    if (encCfg->m_cuQpDeltaSubdiv == 0 || !slice->isIntra()) averageAdaptedLumaQP += applyDeltaQpPeakSmoothing (pic, encCfg, ctuStartAddr, ctuBoundingAddr);\n    for(uint32_t q=ctuStartAddr;q<ctuBoundingAddr;++q) QpaProbe::row("after_smoothing",slice->poc,int(q),0,0,0,0,pic->ctuAdaptedQP[q],pic->ctuAdaptedQP[q],pic->ctuQpaLambda[q],pic->ctuQpaLambda[q]);')
    s=change(s,'int clippedLumaQP = std::max (0, pic->ctuAdaptedQP[ctuRsAddr] + rcQpDiff);',
      'int clippedLumaQP = std::max (0, pic->ctuAdaptedQP[ctuRsAddr] + rcQpDiff);\n    QpaProbe::row("rc_shift",slice->poc,int(ctuRsAddr),0,0,0,0,pic->ctuAdaptedQP[ctuRsAddr],clippedLumaQP,0,0,rcQpDiff,0);')
    s=change(s,'clippedLumaQP = std::min (MAX_QP, clippedLumaQP);',
      'QpaProbe::row("noise_limit",slice->poc,int(ctuRsAddr),0,0,0,0,std::max(0,pic->ctuAdaptedQP[ctuRsAddr]+rcQpDiff),clippedLumaQP,0,0,isEncPass,0);\n    clippedLumaQP = std::min (MAX_QP, clippedLumaQP);')
    s=change(s,'return averageAdaptedLumaQP;\n}',
      'for(uint32_t q=ctuStartAddr;q<ctuBoundingAddr;++q) QpaProbe::row("qpa_final",slice->poc,int(q),0,0,0,0,sliceQP,pic->ctuAdaptedQP[q],sliceLambda,pic->ctuQpaLambda[q],averageAdaptedLumaQP,slice->pic->isPreAnalysis);\n  return averageAdaptedLumaQP;\n}')
    s=change(s,'return adaptedSubCtuQP;\n}',
      'QpaProbe::row("subctu_qpa",slice->poc,-1,lumaArea.x,lumaArea.y,lumaArea.width,lumaArea.height,pic->picInitialQP,adaptedSubCtuQP,0,0,hpEnerSubCTU,hpEnerPicNorm);\n  return adaptedSubCtuQP;\n}')
    for stage in ('activity','activity_qp','before_smoothing','after_smoothing','rc_shift','noise_limit','qpa_final','subctu_qpa'):
        s=s.replace('QpaProbe::row("'+stage+'",','QpaProbe::row(pic->isPreAnalysis ? "analysis_'+stage+'" : "'+stage+'",')
    s='\n'.join(line.replace('slice->poc','pic->poc') if 'QpaProbe::row(' in line else line for line in s.split('\n'))
    plan[name]=s
    name='source/Lib/EncoderLib/EncCu.cpp';s=read(name)
    s=change(s,'#include "EncCu.h"','#include "EncCu.h"\n#include "QpaProbe.h"')
    marker='  if (partitioner.currQtDepth == 0)\n  {\n    m_MergeSimpleFlag = 0;'
    insertion='''  if (QpaProbe::enabled() && !slice.pic->isPreAnalysis && isLuma(partitioner.chType) && partitioner.currQgEnable())
  {
    const int nativeQP=tempCS->baseQP;
    const int requested=QpaProbe::delta(slice.poc,lumaArea.x,lumaArea.y,lumaArea.width,lumaArea.height);
    int targetQP=Clip3(-slice.sps->qpBDOffset[CH_L],MAX_QP,nativeQP+requested);
    const int maxD=int(m_pcEncCfg->m_maxDeltaQP);
    if(maxD>=0 && requested!=0) targetQP=Clip3(std::max(-slice.sps->qpBDOffset[CH_L],slice.sliceQp-maxD),std::min(MAX_QP,slice.sliceQp+maxD),targetQP);
    if(requested==0) targetQP=nativeQP;
    const double oldLambda=m_cRdCost.getLambda(true);
    if(targetQP!=nativeQP) {
      tempCS->currQP[partitioner.chType]=tempCS->baseQP=targetQP;
      bestCS->currQP[partitioner.chType]=bestCS->baseQP=targetQP;
      setUpLambda(slice,oldLambda*std::pow(2.0,double(targetQP-nativeQP)/3.0),targetQP,false,true);
    }
    QpaProbe::row("qg_action",slice.poc,-1,lumaArea.x,lumaArea.y,lumaArea.width,lumaArea.height,nativeQP,targetQP,oldLambda,m_cRdCost.getLambda(true),requested,partitioner.currSubdiv);
  }
'''
    s=change(s,marker,insertion+marker);plan[name]=s
    name='source/Lib/EncoderLib/EncPicture.cpp';s=read(name)
    s=change(s,'#include "EncPicture.h"','#include "EncPicture.h"\n#include "QpaProbe.h"')
    s=change(s,'  xCalcDistortion( pic, *slice->sps );', '''  xCalcDistortion( pic, *slice->sps );
  if(QpaProbe::enabled() && !pic.isPreAnalysis) {
    for(const auto& cu:pic.cs->cus) if(isLuma(cu->chType))
      QpaProbe::row("winning_cu",pic.poc,-1,cu->Y().x,cu->Y().y,cu->Y().width,cu->Y().height,cu->qp,cu->qp,0,0,cu->rootCbf,0);
    QpaProbe::recon(pic.poc,pic.getRecoBuf());
  }''');plan[name]=s
    name='source/Lib/CommonLib/Quant.cpp';s=read(name)
    s=change(s,'#include "Quant.h"','#include "Quant.h"\n#include "Picture.h"\n#include "../EncoderLib/QpaProbe.h"')
    s=change(s,'  int qpy        = tu.cu->qp;', '  int qpy        = tu.cu->qp;\n  if(isLuma(compID) && !tu.cu->slice->pic->isPreAnalysis) QpaProbe::quant(tu.cu->slice->poc,tu.Y().x,tu.Y().y,tu.Y().width,tu.Y().height,qpy);')
    plan[name]=s
    before={n:h(repo/n) for n in plan}
    # All anchors validated before any tracked file is changed.
    for n,t in plan.items():
        original=repo/n;(output/Path(n).name).write_bytes(original.read_bytes());original.write_text(t,encoding='utf-8')
    target=repo/'source/Lib/EncoderLib/QpaProbe.h';target.write_bytes(Path(header).read_bytes())
    result={'commit':COMMIT,'before':before,'after':{n:h(repo/n) for n in plan},'header_sha256':h(target),'stages':['activity','activity_qp','before_smoothing','after_smoothing','rc_shift','noise_limit','qpa_final','subctu_qpa','qg_action','winning_cu','quant_unique_candidate']}
    (output/'PATCH_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(patch(a.repo,Path(__file__).with_name('QpaProbe.h'),a.output)))
