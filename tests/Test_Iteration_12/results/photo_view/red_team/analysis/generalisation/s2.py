import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
D=pd.read_pickle('D.pkl')
def summ(g):
    return pd.Series(dict(n=len(g),pvx=6*g.p.mean(),base=6*g.pb.mean(),gain=6*g.gain.mean(),
        ans=g.ans.mean(),ans_b=g.ans_b.mean(),wrong=int(g.wrong.sum()),wrong_b=int(g.wrong_b.sum()),
        pct10=(g.p==10).mean(),worse=(g.gain<0).mean(),better=(g.gain>0).mean(),
        med_esig=g.esig[g.ans].median(),med_inl=g.inliers[g.ans].median()))
D['hband']=pd.cut(D.alt,[0,150,200,250,350,600,1000,1300,1600,1800,2100],right=False)
print("== by height band, all sets pooled (dev+locked+official+nudt)")
print(D.groupby('hband',observed=True).apply(summ).round(2))
print("== by set x height band")
print(D.groupby(['set','hband'],observed=True).apply(summ).round(2))
