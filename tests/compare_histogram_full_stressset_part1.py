"""Full-resolution 3-bin vs sigma3 stress-set comparison after full-corpus coarse equivalence."""
from pathlib import Path
import importlib.util, sys, math, csv, time
import cv2, numpy as np
C=Path('/mnt/data/snippets/threshold_finder_candidate_v2.py');ROOT=Path('/mnt/data/corpus_compare');OUT=Path('/mnt/data/tests/histogram_full_stressset_part1.csv')
K3=np.array([.25,.5,.25],float)
def kg(s=3.):
 r=max(1,int(math.ceil(3*s)));x=np.arange(-r,r+1,dtype=float);k=np.exp(-.5*(x/s)**2);return k/k.sum()
KG=kg()
def peaks(v):
 ps=[i for i in range(1,255) if v[i]>=v[i-1] and v[i]>v[i+1]]
 if v[255]>v[254]: ps.append(255)
 return ps or [int(np.argmax(v))]
def valley(v,p):
 for i in range(p-1,0,-1):
  if v[i]<=v[i-1] and v[i]<v[i+1]:return i
 return 0
def pv(h,m):
 s=np.convolve(h,K3 if m=='3bin' else KG,mode='same');p=max(peaks(s));return int(p),int(valley(s,p))
def loadtf(name):
 spec=importlib.util.spec_from_file_location(name,C);tf=importlib.util.module_from_spec(spec);sys.modules[name]=tf;spec.loader.exec_module(tf);return tf
wanted_prefixes=['0020','0073','0112','0121','0153']
paths={p.name:p for g in ('before','total','after','horizon') for p in (ROOT/g).glob('*') if p.is_file()}
selected=[]
for prefix in wanted_prefixes:
 hits=[p for n,p in paths.items() if n.startswith(prefix)]
 if hits:selected.append(hits[0])
rows=[];cv2.setNumThreads(1)
for p in selected:
 im=cv2.imread(str(p));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);sc=1200/max(gray.shape);work=cv2.resize(gray,(round(gray.shape[1]*sc),round(gray.shape[0]*sc)),interpolation=cv2.INTER_AREA);h=np.bincount(work.ravel(),minlength=256).astype(float)
 row={'file':p.name}
 for m in ('3bin','sigma3'):
  pair=pv(h,m);tf=loadtf('tf_'+m+'_'+p.stem.replace(' ','_'));tf.rightmost_histogram_peak=lambda _gray,pair=pair:pair
  t=time.time();r=tf.auto_threshold_from_gray(gray);dt=time.time()-t
  row.update({f'{m}_peak':pair[0],f'{m}_left':pair[1],f'{m}_T':int(r.threshold),f'{m}_resolved':int(r.resolved),f'{m}_coarse_T':r.coarse_threshold,f'{m}_sec':round(dt,3)})
  print(p.name,m,pair,'T',r.threshold,'resolved',r.resolved,'sec',round(dt,2),flush=True)
 rows.append(row)
with OUT.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('wrote',OUT)
