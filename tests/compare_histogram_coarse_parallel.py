"""Exact coarse-topology comparison for raw, 3-bin and sigma3 histogram seeds."""
from pathlib import Path
import multiprocessing as mp, importlib.util, sys, math, csv
import cv2, numpy as np
C=Path('/mnt/data/snippets/threshold_finder_candidate_v2.py'); ROOT=Path('/mnt/data/corpus_compare'); OUT=Path('/mnt/data/tests/histogram_coarse_comparison.csv')
K3=np.array([.25,.5,.25],float)
def gauss(s=3.):
 r=max(1,int(math.ceil(3*s)));x=np.arange(-r,r+1,dtype=float);k=np.exp(-.5*(x/s)**2);return k/k.sum()
KG=gauss()
def peaks(v):
 ps=[i for i in range(1,255) if v[i]>=v[i-1] and v[i]>v[i+1]]
 if v[255]>v[254]: ps.append(255)
 return ps or [int(np.argmax(v))]
def valley(v,p):
 for i in range(p-1,0,-1):
  if v[i]<=v[i-1] and v[i]<v[i+1]: return i
 return 0
def pv(h,m):
 s=h if m=='raw' else np.convolve(h,K3 if m=='3bin' else KG,mode='same');p=max(peaks(s));return int(p),int(valley(s,p))
def loadtf():
 name='tfc_'+str(mp.current_process().pid);spec=importlib.util.spec_from_file_location(name,C);tf=importlib.util.module_from_spec(spec);sys.modules[name]=tf;spec.loader.exec_module(tf);return tf
def calc(item):
 g,sp=item; tf=loadtf(); im=cv2.imread(sp);gray=tf.to_gray(im);work=tf.resize_gray_max_dim(gray);h=np.bincount(work.ravel(),minlength=256).astype(float)
 pairs={m:pv(h,m) for m in ('raw','3bin','sigma3')};cache={}
 for pair in set(pairs.values()):
  tf.find_rightmost_histogram_peak=lambda _gray,pair=pair:pair
  try:
   c=tf.coarse_threshold_search(work); cache[pair]=(1,int(c.threshold),int(c.seed_threshold),int(c.component_area),int(np.count_nonzero(c.seed_mask)),str(c.component_bbox),str(c.seed_point))
  except tf.ThresholdResolutionError as e: cache[pair]=(0,None,None,None,0,None,None)
 row={'group':g,'file':Path(sp).name}
 for m,pair in pairs.items():
  row[f'{m}_peak'],row[f'{m}_left']=pair
  res,cT,sT,cA,sA,bbox,pt=cache[pair];row[f'{m}_resolved']=res;row[f'{m}_coarse_T']=cT;row[f'{m}_seed_T']=sT;row[f'{m}_coarse_area']=cA;row[f'{m}_seed_area']=sA;row[f'{m}_bbox']=bbox;row[f'{m}_seed_point']=pt
  row[f'{m}_peak_exact']=int(h[pair[0]]);row[f'{m}_mass3']=int(h[max(0,pair[0]-1):min(256,pair[0]+2)].sum())
 return row
if __name__=='__main__':
 items=[]
 for g in ('before','total','after','horizon'):
  items += [(g,str(p)) for p in sorted((ROOT/g).glob('*')) if p.is_file()]
 with mp.Pool(6) as pool: rows=list(pool.imap_unordered(calc,items,chunksize=1))
 rows.sort(key=lambda r:(('before','total','after','horizon').index(r['group']),r['file']))
 with OUT.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(len(rows),OUT)
