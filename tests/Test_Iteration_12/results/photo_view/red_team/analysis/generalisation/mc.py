import pandas as pd, numpy as np, ast
from load import *
rng=np.random.default_rng(1)
d=pd.read_csv(R+'photo_view/photo_view_full.csv'); d['null']=d.null_inl.apply(ast.literal_eval)
x=d[(d.matcher=='xoftr')&(d.input=='photo view')]
hi=x[x.height>1600]; T0=hi.true_inl.values.astype(float); N0=np.concatenate(hi.null.values).astype(float)
# also pipeline winner inliers from 1800+ for a second true pool
D=pd.read_pickle('D.pkl'); P0=D[(D.alt>1800)&D.ans&D.set.isin(['work332','locked168'])].inliers.values.astype(float)
print('true pool (diag) n',len(T0),'median',np.median(T0),' pipeline winners >1800 n',len(P0),'median',np.median(P0),' null pool n',len(N0),'mean',N0.mean())
def sim(h,slope,null_gain,Ncand,bar,pool=T0,reps=40000,nullscale_h=None):
    f=np.exp(-slope*(h-1950)/1000.)
    T=rng.poisson(rng.choice(pool,reps)*f)
    g=(1+null_gain*(h-1950)/1000.)
    nl=rng.poisson(rng.choice(N0,(reps,Ncand))*g)
    m=nl.max(1)
    right=(T>=bar)&(T>m); wrong=(m>=bar)&(m>=T); skip=~(right|wrong)
    return right.mean(),wrong.mean(),skip.mean()
print("h  slope  nullgain Ncand bar | right wrong skip")
for slope,ng in [(0.6,0.2),(1.19,0.5),(1.8,1.0)]:
  for N in (10,30):
    for bar in (12,20):
        print('--- slope %.2f null_gain/km %.1f N %d bar %d'%(slope,ng,N,bar))
        for h in (1950,2250,2500,2750,3000):
            r,w,s=sim(h,slope,ng,N,bar); print('  %4d  right %.3f wrong %.3f skip %.3f | EV pts(ignoring precision, +8/-20) %.2f'%(h,r,w,s,8*r-20*w))
