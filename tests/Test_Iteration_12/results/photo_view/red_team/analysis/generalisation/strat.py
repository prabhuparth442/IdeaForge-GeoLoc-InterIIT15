from load import *
pd.set_option('display.width',220); pd.set_option('display.max_columns',40)
PVX={'work332':'pv3_dev_x0','official500':'pv3_official_x0','nudt300':'pv3_nudt_x0','locked168':'pvx_test_t0'}
BASEF={'work332':'details_xoftr800','official500':'old_details_official_base500_earlier','nudt300':'old_details_nudt_base','locked168':'test_details_xoftr800'}
def sg(x): return np.where(x>180,x-360,x)
frames=[]
for s in PVX:
    x=run(PVX[s]); b=pd.read_csv(R+'native_fine/%s.csv'%BASEF[s]); b['pb']=b.apply(pts,axis=1)
    b=b[['filename','status','error_m','inliers','pb']].rename(columns={'status':'status_b','error_m':'err_b','inliers':'inl_b'})
    x=x.merge(b,on='filename',how='left'); frames.append(x)
D=pd.concat(frames,ignore_index=True)
D['gain']=D.p-D.pb
D['pit']=sg(D.pitch); D['rol']=sg(D.roll); D['tilt']=np.hypot(D.pit,D.rol)
D['wrong']=(D.p==-20); D['wrong_b']=(D.pb==-20)
D['ans']=D.status=='match'; D['ans_b']=D.status_b=='match'
D['esig']=D.error_m/D.sigma_m
# coarse rank
rk=[]
for s,f in [('work332','highalt_dev332'),('official500','official500'),('nudt300','nudt300')]:
    q=pd.read_csv(R+'coarse_confidence/queries_%s.csv'%f)
    g=q.groupby('name').agg(best_rank=('true_rank','min'),zq=('z1','max')).reset_index().rename(columns={'name':'filename'})
    rk.append(g)
rk=pd.concat(rk)
D=D.merge(rk,on='filename',how='left')
D.to_pickle('D.pkl')
if __name__=='__main__':
    print(D.groupby('set').agg(n=('p','size'),score=('p',lambda v:6*v.mean()),base=('pb',lambda v:6*v.mean()),wrong=('wrong','sum'),wrong_b=('wrong_b','sum')))
    print(D.tilt.describe()); print(D.groupby('set').tilt.describe())
    print(D.best_rank.describe())
