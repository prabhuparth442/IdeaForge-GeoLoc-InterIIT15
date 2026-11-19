import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
D=pd.read_pickle('D.pkl')
def summ(g):
    a=g[g.ans]
    return pd.Series(dict(n=len(g),ans=len(a),pvx=6*g.p.mean(),base=6*g.pb.mean(),gain=6*g.gain.mean(),wrong=int(g.wrong.sum()),wrong_b=int(g.wrong_b.sum()),
        wr_per_ans=g.wrong.sum()/max(len(a),1),worse=(g.gain<0).mean(),better=(g.gain>0).mean(),p10=(g.p==10).mean(),med_esig=a.esig.median()))
print("== by inlier count of the answer (PV-X), pooled")
A=D[D.ans].copy(); A['ib']=pd.cut(A.inliers,[0,11,12,15,20,30,50,100,200,10000])
print(A.groupby('ib',observed=True).apply(summ).round(3))
print("== inlier bins, Rajasthan sets only (work+locked)")
R_=A[A.set.isin(['work332','locked168'])]
print(R_.groupby('ib',observed=True).apply(summ).round(3))
print("== bar counterfactual: skip answers with inliers < bar (0 points) -> score/60 per set")
for s,g in D.groupby('set'):
    out=[]
    for bar in (12,16,20,25,30,40):
        p=np.where(g.ans&(g.inliers<bar),0,g.p); out.append((bar,round(6*p.mean(),2),int((p==-20).sum()),int((g.ans&(g.inliers>=bar)).sum())))
    print(s,out)
g=D[D.set.isin(['work332','locked168'])]
for bar in (12,16,20,25,30): 
    p=np.where(g.ans&(g.inliers<bar),0,g.p); print('Rajasthan pooled bar',bar,round(6*p.mean(),2),'wrong',(p==-20).sum(),'answered',(g.ans&(g.inliers>=bar)).sum())
print("== by tilt (deg) Rajasthan+official")
T=D[D.set!='nudt300'].copy(); T['tb']=pd.cut(T.tilt,[-.1,3,5,8,11,15,25])
print(T.groupby('tb',observed=True).apply(summ).round(2))
print("== tilt x high (>=1000m) Rajasthan")
T2=D[(D.set.isin(['work332','locked168']))&(D.alt>=1000)].copy(); T2['tb']=pd.cut(T2.tilt,[-.1,4,7,10,25])
print(T2.groupby('tb',observed=True).apply(summ).round(2))
print("== pitch sign / roll sign Rajasthan")
R2=D[D.set.isin(['work332','locked168'])].copy()
R2['pr']=pd.cut(R2.pit,[-30,-8,-3,0,3,8,30]); print(R2.groupby('pr',observed=True).apply(summ).round(2))
R2['rr']=pd.cut(R2.rol,[-30,-8,-3,0,3,8,30]); print(R2.groupby('rr',observed=True).apply(summ).round(2))
print("== by coarse z of pipeline (z col) Rajasthan")
R2['zb']=pd.qcut(R2.z,5); print(R2.groupby('zb',observed=True).apply(summ).round(2))
print("== by whether coarse true_rank known / second_pass / queries")
R3=D.copy(); print(R3.groupby(['set','second_pass']).apply(summ).round(2))
