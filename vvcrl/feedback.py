"""Read only the completed winning-CU trace for the preceding coding step.

Native publishes the outcome after the reconstruction callback flushes the
trace in feedback mode. Candidate QG rows are deliberately not treated as
winning CU QP. No future POC, decoded detections or anchor is consulted.
"""
import csv
import numpy as np
from .config import W,H,CTU,NY,NX

def winning_map(rows,poc):
    coverage=np.zeros((H,W),dtype=bool)
    canvas=np.zeros((H,W),dtype=np.float64)
    for row in rows:
        if row['stage']!='winning_cu':continue
        if int(row['poc'])!=poc:raise ValueError('Winning feedback POC mismatch')
        x,y,w,h=(int(row[n]) for n in ('x','y','w','h'))
        right,bottom=min(W,x+w),min(H,y+h)
        qp=float(row['qp_after'])
        if min(x,y)<0 or right<=x or bottom<=y or not np.isfinite(qp) or not -12<=qp<=63:
            raise ValueError('Invalid winning CU feedback')
        if coverage[y:bottom,x:right].any():raise ValueError('Overlapping winning CU feedback')
        coverage[y:bottom,x:right]=True;canvas[y:bottom,x:right]=qp
    if not coverage.all():raise ValueError('Incomplete winning CU feedback; refuse noncausal/partial state')
    return np.array([[canvas[y*CTU:min(H,(y+1)*CTU),x*CTU:min(W,(x+1)*CTU)].mean()
                      for x in range(NX)] for y in range(NY)],dtype=np.float32)

class WinningTrace:
    def __init__(self,path):
        self.path=path;self.offset=0;self.header=None;self.seen=set()
    def completed(self,poc):
        if poc in self.seen:raise ValueError('Duplicate winning feedback')
        rows=[]
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                start=stream.tell();line=stream.readline()
                if not line.endswith(b'\n'):
                    self.offset=start;break
                fields=next(csv.reader([line.decode('ascii')]))
                if self.header is None:
                    self.header=fields
                    if not {'stage','poc','x','y','w','h','qp_after'}<=set(fields):raise ValueError('Trace header missing')
                else:
                    if len(fields)!=len(self.header):raise ValueError('Malformed native trace row')
                    row=dict(zip(self.header,fields))
                    if row['stage']=='winning_cu':rows.append(row)
        result=winning_map(rows,poc)
        self.seen.add(poc)
        return result
