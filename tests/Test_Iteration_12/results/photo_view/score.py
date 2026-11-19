import pandas as pd, numpy as np, sys
def pts(r):
    if r.status!='match': return 0
    e,s=r.error_m,r.sigma_m
    return 10 if e<=s else 6 if e<=2*s else 3 if e<=4*s else 0 if e<=8*s else -20
def load(p):
    d=pd.read_csv(p); d['p']=d.apply(pts,axis=1); return d
def line(name,d):
    a=d[d.status=='match']; r=a.error_m/a.sigma_m
    return '%-22s score %.2f answered %d wrong %d  s/photo %.2f  median err/sigma %.2f within1 %.2f' % (name,6*d.p.mean(),len(a),(d.p==-20).sum(),d.total_seconds.mean(),r.median(),(r<=1).mean())
