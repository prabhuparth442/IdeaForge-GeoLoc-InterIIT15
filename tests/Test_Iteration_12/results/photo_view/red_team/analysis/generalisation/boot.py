from load import *
rng=np.random.default_rng(0)
def pair(a,b):
    A=run(a).set_index('filename').p; B=run(b).set_index('filename').p
    j=pd.concat([A,B],axis=1,keys=['a','b']).dropna(); d=(j.a-j.b).values*6
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)]
    return len(d),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),(d!=0).sum(),(d>0).sum(),(d<0).sum()
rows=[('PV-X dev0 vs PV-X dev seed1','pv3_dev_x0','pvx_dev_s1'),
('PV-X nudt0 vs nudt seed1','pv3_nudt_x0','pvx_nudt_s1'),
('PV-X off0 vs off seed1','pv3_official_x0','pvx_official_s1'),
('PV-X2 vs PV-X dev s0','pvv_dev_x3s0','pv3_dev_x0'),
('PV-X2 vs PV-X nudt s0','pvv_nudt_x3s0','pv3_nudt_x0'),
('PV-X2 vs PV-X nudt s1','pvv_nudt_x3s1','pvx_nudt_s1'),
('PV-X2 vs PV-X off s0','pvv_official_x3s0','pv3_official_x0'),
('limit40 vs PV-X off s0','pvv_official_x2s0','pv3_official_x0'),
('PV-X (bar12) vs bar20+agree all-h, dev','pv3_dev_x0','pv2_dev_g0p04'),
('PV-X vs bar20 nudt','pv3_nudt_x0','pv2_nudt_g0'),
('PV-X vs bar20 off','pv3_official_x0','pvs_official_g0b20'),
('PV-X vs LoFTR<800 (b12a) dev','pv3_dev_x0','pv3_dev_b12a'),
('PV-X vs LoFTR<800 (b12a) off','pv3_official_x0','pv3_official_b12a'),
('PV-X vs LoFTR<800 nudt','pv3_nudt_x0','pv3_nudt_b12a'),
('PV-X vs gate0 prior1.0 (pvs_dev_g0) dev','pv3_dev_x0','pvs_dev_g0'),
]
print("%-42s %4s %7s %8s %8s %5s %4s %4s"%('pair','n','dscore','lo95','hi95','diff','+','-'))
for r in rows:
    n,m,lo,hi,nd,up,dn=pair(r[1],r[2]); print("%-42s %4d %+7.2f %+8.2f %+8.2f %5d %4d %4d"%(r[0],n,m,lo,hi,nd,up,dn))
# vs base bootstrap (D.pkl)
D=pd.read_pickle('D.pkl')
print()
for s,g in D.groupby('set'):
    d=(g.p-g.pb).values*6; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)]
    print(s,'PV-X minus base: %+.2f [%+.2f,%+.2f]  SE of a single score %.2f'%(d.mean(),*np.percentile(bs,[2.5,97.5]),np.std(6*g.p.values)/np.sqrt(len(g))))
g=D[D.set.isin(['work332','locked168'])]; d=(g.p-g.pb).values*6; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)]; print('Rajasthan pooled', d.mean(), np.percentile(bs,[2.5,97.5]))
g=D[(D.alt>=1000)]; d=(g.p-g.pb).values*6; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)]; print('>=1000m', len(g), d.mean(), np.percentile(bs,[2.5,97.5]))
for lo,hi in [(0,350),(350,1000),(1000,1600),(1600,2100)]:
    g=D[(D.alt>=lo)&(D.alt<hi)]; d=(g.p-g.pb).values*6; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)]; print(lo,hi,len(g),'%+.2f [%+.2f,%+.2f]'%(d.mean(),*np.percentile(bs,[2.5,97.5])))
# one wrong = ? points
for s,n in [('work332',332),('locked168',168),('nudt300',300),('official500',500)]: print(s,'one -20 -> %.2f ; one wrong replacing a +10 -> %.2f'%(6*20/n,6*30/n))
