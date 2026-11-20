#pragma once
// Online picture handshake. Inactive unless VVCRL_SESSION_DIR is explicitly set.
// The existing QG/lambda path stays in QpaProbe.h; this adds observation/outcome IO.
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../../../thirdparty/nlohmann_json/single_include/nlohmann/json.hpp"

namespace vvenc { namespace VvcRl {
using Json=nlohmann::json;
inline const char* env(const char* k){const char* s=std::getenv(k);return s?s:"";}
inline bool enabled(){return *env("VVCRL_SESSION_DIR")!=0;}
struct Session {
  std::mutex mutex;
  std::map<int,std::vector<double>> activity;
  std::vector<int> actions;
  int seq=0,poc=-1,width=0,height=0,ctu=0;
  uint64_t bits=0;
  bool pending=false;
};
inline Session& state(){static Session s;return s;}
inline std::string path(const std::string& n){return std::string(env("VVCRL_SESSION_DIR"))+"/"+n;}
inline void atomic(const std::string& name,const Json& j){
  const std::string p=path(name),tmp=p+".tmp";
  {std::ofstream f(tmp,std::ios::binary);if(!f)throw std::runtime_error("Cannot create RL message");f<<j.dump();f.flush();if(!f)throw std::runtime_error("RL message write failed");}
  if(std::rename(tmp.c_str(),p.c_str())!=0)throw std::runtime_error("Cannot publish RL message");
}
inline void activity(int poc,int index,double value){
  if(!enabled())return;
  auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  auto& a=s.activity[poc];if(index<0)throw std::runtime_error("Invalid RL CTU index");
  if(a.size()<=size_t(index)){a.resize(index+1,-1.0);}
  a[index]=value;
}
template<class SliceT,class CfgT> inline void before(const SliceT& slice,const CfgT& cfg){
  if(!enabled() || slice.pic->isPreAnalysis)return;
  auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  if(s.pending)throw std::runtime_error("RL observed next picture before previous AU/RC outcome");
  s.width=cfg.m_SourceWidth;s.height=cfg.m_SourceHeight;s.ctu=int(slice.pic->cs->pcv->maxCUSize);
  if(s.width!=480 || s.height!=320 || s.ctu!=128 || cfg.m_maxParallelFrames!=0)
    throw std::runtime_error("RL v1 requires 480x320 CTU128 and serial pictures");
  const int n=((s.width+s.ctu-1)/s.ctu)*((s.height+s.ctu-1)/s.ctu);
  auto it=s.activity.find(slice.poc);
  if(it==s.activity.end() || it->second.size()!=size_t(n))throw std::runtime_error("Missing live QPA activity snapshot");
  std::vector<int> qp(n);for(int i=0;i<n;++i){qp[i]=slice.pic->ctuAdaptedQP[i];if(!std::isfinite(it->second[i]) || it->second[i]<0)throw std::runtime_error("Invalid activity");}
  s.poc=slice.poc;s.pending=true;
  Json j={{"schema","vvc-rl-online/1"},{"episode",env("VVCRL_EPISODE")},{"k",s.seq},{"poc",s.poc},
    {"width",s.width},{"height",s.height},{"ctu",s.ctu},{"bits_before",s.bits},
    {"qp",qp},{"activity",it->second},{"slice_qp",slice.sliceQp},{"temporal_layer",slice.TLayer},
    {"reference",bool(slice.pic->isReferenced)},{"slice_type",int(slice.sliceType)}};
  atomic("obs_"+std::to_string(s.seq)+".json",j);
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(300);
  const std::string action=path("action_"+std::to_string(s.seq)+".json");
  Json a;
  while(true){
    std::ifstream f(action,std::ios::binary);
    if(f){f>>a;break;}
    if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("Timed out waiting for RL action");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if(a.at("episode").get<std::string>()!=env("VVCRL_EPISODE") || a.at("k").get<int>()!=s.seq || a.at("poc").get<int>()!=s.poc)
    throw std::runtime_error("RL action identity mismatch");
  s.actions=a.at("deltas").get<std::vector<int>>();
  if(s.actions.size()!=size_t(n))throw std::runtime_error("RL action shape mismatch");
  for(int d:s.actions)if(d < -2 || d > 2)throw std::runtime_error("RL delta outside integer bins");
}
inline int delta(int poc,int x,int y,int w,int h){
  auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  if(!s.pending || poc!=s.poc || x<0 || y<0 || w<=0 || h<=0 || x>=s.width || y>=s.height)
    throw std::runtime_error("RL action used outside pending picture");
  const int nx=(s.width+s.ctu-1)/s.ctu;
  const int right=std::min(s.width,x+w),bottom=std::min(s.height,y+h);
  double sum=0,area=0;
  for(int cy=y/s.ctu;cy<=(bottom-1)/s.ctu;++cy)for(int cx=x/s.ctu;cx<=(right-1)/s.ctu;++cx){
    const int a=(std::min(right,(cx+1)*s.ctu)-std::max(x,cx*s.ctu))*(std::min(bottom,(cy+1)*s.ctu)-std::max(y,cy*s.ctu));
    sum+=a*s.actions[cy*nx+cx];area+=a;
  }
  if(area<=0){throw std::runtime_error("Empty RL QG intersection");}
  return int(std::round(sum/area));
}
template<class PicT,class AuT> inline void after(const PicT& pic,const AuT& au){
  if(!enabled() || pic.isPreAnalysis)return;
  auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  if(!s.pending || s.poc!=pic.poc || au.poc!=pic.poc)throw std::runtime_error("RL AU outcome identity mismatch");
  // Match VVEncImpl's Annex-B start-code accounting, including all SEI/headers.
  uint64_t bytes=0;bool first=true;
  for(const auto* unit:au){
    const auto t=unit->m_nalUnitType;
    const bool longCode=first || t==VVENC_NAL_UNIT_DCI || t==VVENC_NAL_UNIT_SPS || t==VVENC_NAL_UNIT_VPS || t==VVENC_NAL_UNIT_PPS || t==VVENC_NAL_UNIT_PREFIX_APS || t==VVENC_NAL_UNIT_SUFFIX_APS;
    bytes+=(longCode?4:3)+unit->m_nalUnitData.str().size();first=false;
  }
  if(!bytes){throw std::runtime_error("Empty RL AU");}
  s.bits+=bytes*8;
  atomic("out_"+std::to_string(s.seq)+".json",{{"schema","vvc-rl-online/1"},{"episode",env("VVCRL_EPISODE")},
    {"k",s.seq},{"poc",s.poc},{"bits",bytes*8},{"bits_total",s.bits}});
  s.activity.erase(s.poc);s.actions.clear();s.pending=false;++s.seq;
}
} }
