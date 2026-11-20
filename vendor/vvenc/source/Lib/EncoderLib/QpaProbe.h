#pragma once
// Diagnostic only. No learned policy; disabled unless a trace/actions path is set.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "VvcRlSession.h"
namespace vvenc { namespace QpaProbe {
inline const char* env(const char* key) { const char* s=std::getenv(key); return s ? s : ""; }
inline bool enabled() { return *env("VVENC_QPA_PROBE_TRACE") != 0; }
inline std::mutex& guard() { static std::mutex m; return m; }
inline void row(const char* stage,int poc,int ctu,int x,int y,int w,int h,
                double before,double after,double lb=0,double la=0,double a=0,double b=0) {
  if (!enabled()) return;
  if(std::string(stage)=="activity") VvcRl::activity(poc,ctu,a);
  std::lock_guard<std::mutex> lock(guard());
  static std::ofstream out(env("VVENC_QPA_PROBE_TRACE"));
  static bool header=false;
  if (!out) throw std::runtime_error("Cannot write QPA diagnostic trace");
  if (!header) { out<<"stage,poc,ctu,x,y,w,h,qp_before,qp_after,lambda_before,lambda_after,aux_a,aux_b\n";header=true; }
  out<<stage<<','<<poc<<','<<ctu<<','<<x<<','<<y<<','<<w<<','<<h<<','<<std::setprecision(17)
     <<before<<','<<after<<','<<lb<<','<<la<<','<<a<<','<<b<<'\n';
}
struct Actions {
  int width=0,height=0,ctu=0,frames=0; std::vector<int> values;
  Actions() {
    const char* file=env("VVENC_QPA_PROBE_ACTIONS"); if (!*file) return;
    std::ifstream in(file); if (!(in>>width>>height>>ctu>>frames) || width<=0 || height<=0 || ctu<=0 || frames<=0)
      throw std::runtime_error("Invalid QPA probe action header");
    int count=frames*((width+ctu-1)/ctu)*((height+ctu-1)/ctu);values.resize(count);
    for (int& v:values) if (!(in>>v) || v < -4 || v > 4) throw std::runtime_error("Invalid integer QPA action");
    std::string extra;if (in>>extra) throw std::runtime_error("Unexpected trailing action data");
  }
};
inline int delta(int poc,int x,int y,int w,int h) {
  if(VvcRl::enabled())return VvcRl::delta(poc,x,y,w,h);
  static Actions a;if(a.values.empty())return 0;
  if(poc<0 || poc>=a.frames || x<0 || y<0 || w<=0 || h<=0 || x>=a.width || y>=a.height)
    throw std::runtime_error("QPA action geometry/POC outside contract");
  int nx=(a.width+a.ctu-1)/a.ctu,ny=(a.height+a.ctu-1)/a.ctu;
  int right=std::min(a.width,x+w),bottom=std::min(a.height,y+h);double total=0,area=0;
  for(int cy=y/a.ctu;cy<=(bottom-1)/a.ctu;++cy)for(int cx=x/a.ctu;cx<=(right-1)/a.ctu;++cx){
    int dx=std::min(right,(cx+1)*a.ctu)-std::max(x,cx*a.ctu);
    int dy=std::min(bottom,(cy+1)*a.ctu)-std::max(y,cy*a.ctu);
    total+=dx*dy*a.values[(poc*ny+cy)*nx+cx];area+=dx*dy;
  }
  if(area<=0)throw std::runtime_error("Empty QPA action intersection");
  return int(std::round(total/area));
}
inline void quant(int poc,int x,int y,int w,int h,int qp) {
  if (!enabled() || !(poc==0 || poc==1 || poc==31 || poc==32 || poc==63)) return;
  typedef std::tuple<int,int,int,int,int,int> Key;
  static std::mutex m;static std::set<Key> seen;
  {std::lock_guard<std::mutex> lock(m);if(!seen.insert(Key(poc,x,y,w,h,qp)).second)return;}
  row("quant_unique_candidate",poc,-1,x,y,w,h,qp,qp);
}
template<class Buf> inline void recon(int poc,const Buf& buf) {
  const char* dir=env("VVENC_QPA_PROBE_RECON");if(!*dir)return;
  std::ofstream f(std::string(dir)+"/"+std::to_string(poc)+".yuv",std::ios::binary);
  if(!f)throw std::runtime_error("Cannot write diagnostic reconstruction");
  for(int c=0;c<3;++c){const auto& p=buf.bufs[c];for(int y=0;y<p.height;++y)
    f.write(reinterpret_cast<const char*>(p.buf+y*p.stride),p.width*sizeof(*p.buf));}
}
} }
