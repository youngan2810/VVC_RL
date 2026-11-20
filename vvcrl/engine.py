"""One encoder process, one live picture handshake, coding-order rewards."""
import gzip,csv,json,os,subprocess,time,uuid
from pathlib import Path
import numpy as np
from .config import Profile,FB,W,H,atomic_json,digest

class NativeProcess:
    def __init__(self,encoder,source,folder,profile,rate,timeout=300):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.profile=profile.validate();self.rate=rate;self.timeout=timeout;self.k=0;self.bits=0;self.seen=set();self.episode=uuid.uuid4().hex
        self.bridge=self.folder/'bridge';self.bridge.mkdir();self.recon=self.folder/'recon';self.recon.mkdir()
        self.stream=self.folder/'stream.266';self.log=self.folder/'encoder.log';self.logfile=self.log.open('wb')
        env={k:v for k,v in os.environ.items() if not k.startswith(('VVCRL_','VVENC_QPA_PROBE_','VVENC_RCDOT_','VVENC_RL_'))}
        env.update(VVCRL_SESSION_DIR=str(self.bridge.resolve()),VVCRL_EPISODE=self.episode,
          VVENC_QPA_PROBE_TRACE=str((self.folder/'trace.csv').resolve()),VVENC_QPA_PROBE_RECON=str(self.recon.resolve()))
        if os.name=='nt' and Path('C:/msys64/ucrt64/bin').is_dir():env['PATH']='C:/msys64/ucrt64/bin;'+env.get('PATH','')
        self.command=profile.command(Path(encoder).resolve(),Path(source).resolve(),self.stream.resolve(),rate)
        atomic_json(self.folder/'command.json',self.command)
        self.process=subprocess.Popen(self.command,stdout=self.logfile,stderr=subprocess.STDOUT,env=env)
    def _wait(self,kind):
        p=self.bridge/f'{kind}_{self.k}.json';deadline=time.monotonic()+self.timeout
        while not p.exists():
            if self.process.poll() is not None:
                self.logfile.flush();raise RuntimeError(f'Encoder exited before {p.name}: '+self.log.read_text(errors='replace')[-3000:])
            if time.monotonic()>deadline:raise TimeoutError('Native handshake timeout: '+p.name)
            time.sleep(.002)
        # Windows can briefly deny opening a just-renamed file. The producer
        # publishes atomically, so retry sharing violations, never partial JSON.
        while True:
            try:msg=json.loads(p.read_text(encoding='utf-8'));break
            except PermissionError:
                if time.monotonic()>deadline:raise
                time.sleep(.002)
        if msg.get('schema')!='vvc-rl-online/1' or msg.get('episode')!=self.episode or msg.get('k')!=self.k:raise ValueError('Native message identity mismatch')
        return msg
    def observation(self):
        msg=self._wait('obs')
        if msg['bits_before']!=self.bits or msg['poc'] in self.seen:raise ValueError('Noncausal native state or duplicate POC')
        self.current=msg;return msg
    def step(self,deltas):
        d=np.asarray(deltas)
        if d.size!=12 or not np.isfinite(d).all() or (d!=np.round(d)).any() or (abs(d)>2).any():raise ValueError('Not 12 legal integer deltas')
        atomic_json(self.bridge/f'action_{self.k}.json',{'episode':self.episode,'k':self.k,'poc':self.current['poc'],'deltas':d.astype(int).ravel().tolist()})
        out=self._wait('out')
        if out['poc']!=self.current['poc'] or out['bits']<=0 or out['bits_total']!=self.bits+out['bits']:raise ValueError('Native AU accounting mismatch')
        p=self.recon/f"{out['poc']}.yuv";b=p.read_bytes()
        if len(b)!=FB*2:raise ValueError('Native reconstruction shape/bitdepth mismatch')
        self.bits=out['bits_total'];self.seen.add(out['poc']);self.k+=1
        return out,np.frombuffer(b,dtype='<u2').copy()
    def finish(self):
        if self.k!=self.profile.frames or self.seen!=set(range(self.profile.frames)):raise ValueError('Incomplete native episode')
        rc=self.process.wait(timeout=self.timeout);self.logfile.close()
        if rc:raise RuntimeError(self.log.read_text(errors='replace')[-3000:])
        if self.bits!=self.stream.stat().st_size*8:raise ValueError('AU sum does not equal actual Annex-B stream size')
        return {'bits':self.bits,'actual_kbps':self.bits*30/self.profile.frames/1000,'bitstream_sha256':digest(self.stream)}
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if not self.logfile.closed:self.logfile.close()

def verify_decode(folder,decoder,frames,timeout=600):
    folder=Path(folder);dest=folder/'independent.yuv'
    env=os.environ.copy()
    if os.name=='nt' and Path('C:/msys64/ucrt64/bin').is_dir():env['PATH']='C:/msys64/ucrt64/bin;'+env.get('PATH','')
    with (folder/'decoder.log').open('wb') as f:
        subprocess.run([str(decoder),'-b',str(folder/'stream.266'),'-o',str(dest),'--threads','2'],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout,check=True)
    if dest.stat().st_size!=frames*FB*2:raise ValueError('Independent decoder size mismatch')
    with dest.open('rb') as f:
        for poc in range(frames):
            if f.read(FB*2)!=(folder/'recon'/f'{poc}.yuv').read_bytes():raise ValueError('Decoder/native parity mismatch at POC '+str(poc))
    dest.unlink();return True

def audit_trace(folder):
    folder=Path(folder);counts={};requested={};clipped=0;zeroed=0;rows=0;quant=0;winning={};coverage={};mapping={}
    for obsfile in (folder/'bridge').glob('obs_*.json'):
        obs=json.loads(obsfile.read_text());action=json.loads(obsfile.with_name('action_'+str(obs['k'])+'.json').read_text())
        mapping[obs['poc']]=np.asarray(action['deltas']).reshape(3,4)
    with (folder/'trace.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['stage']=='quant_unique_candidate':quant+=1
            if r['stage']=='winning_cu':
                poc=int(r['poc']);x,y,w,h=[int(r[n]) for n in ('x','y','w','h')]
                w=min(w,W-x);h=min(h,H-y)
                if min(x,y)<0 or min(w,h)<=0:raise ValueError('Invalid winning CU geometry')
                canvas=coverage.setdefault(poc,np.zeros((H,W),np.uint8))
                if canvas[y:y+h,x:x+w].any():raise ValueError('Overlapping winning CU coverage')
                canvas[y:y+h,x:x+w]=1;qp=str(int(float(r['qp_after'])));winning[qp]=winning.get(qp,0)+w*h
            if r['stage']!='qg_action':continue
            d=float(r['qp_after'])-float(r['qp_before']);want=float(r['aux_a']);rows+=1
            x,y,w,h=[int(r[n]) for n in ('x','y','w','h')];right=min(W,x+w);bottom=min(H,y+h)
            if int(r['poc']) not in mapping or x<0 or y<0 or x>=right or y>=bottom:raise ValueError('QG action has no matching live action')
            weighted=area=0
            for cy in range(y//128,(bottom-1)//128+1):
                for cx in range(x//128,(right-1)//128+1):
                    a=(min(right,(cx+1)*128)-max(x,cx*128))*(min(bottom,(cy+1)*128)-max(y,cy*128));weighted+=a*mapping[int(r['poc'])][cy,cx];area+=a
            average=weighted/area;rounded=int(np.sign(average)*np.floor(abs(average)+.5))
            if want!=rounded:raise ValueError('Requested CTU action differs from native QG aggregation')
            if not np.isclose(float(r['lambda_after']),float(r['lambda_before'])*2**(d/3),rtol=1e-10,atol=1e-10):raise ValueError('QP/lambda audit failed')
            counts[str(int(d))]=counts.get(str(int(d)),0)+1;requested[str(int(want))]=requested.get(str(int(want)),0)+1;clipped+=int(d!=want);zeroed+=int(want!=0 and d==0)
    if not rows or not quant:raise ValueError('Native QG/quantization trace missing')
    if set(coverage)!=set(mapping) or any(not c.all() for c in coverage.values()):raise ValueError('Winning CU coverage incomplete')
    return {'qg_candidate_rows':rows,'requested_delta_histogram':requested,'effective_delta_histogram':counts,'clipped_rows':clipped,'quant_unique_candidates':quant,
            'qg_mapping_verified':True,'winning_cu_pixels':sum(winning.values()),'winning_qp_pixel_histogram':winning,'winning_coverage_verified':True,
            'nonzero_requests_zeroed':zeroed,'nonzero_requests_zeroed_fraction':zeroed/max(rows-requested.get('0',0),1),
            'note':'Candidate calls are not winning pixel area. Final CU QP includes residual-free CUs.'}
