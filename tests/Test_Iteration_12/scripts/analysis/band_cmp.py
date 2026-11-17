import pandas as pd, numpy as np, sys
sp=pd.read_csv("/home/voidness/Storage/Synced Folders/InterIIT Tech 15/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/highalt_split.csv")[['filename','band','split','alt']]
order=['100-300','300-500','500-800','800-1000','1000-1300','1300-1600','1600-2100']
def load(f):
    t=pd.read_csv(f).merge(sp,on='filename',suffixes=('_d','')); return t[t.split=='dev']
runs={'base':load('highalt_base500_with_alt.csv')}
for a in sys.argv[1:]:
    name,f=a.split('=',1); runs[name]=load(f)
rows=[]
for b in order+['ALL']:
    row={'band':b}
    for k,t in runs.items():
        s=t if b=='ALL' else t[t.band==b]
        row[k]='%.2f|%d%%|%d'%(s.points.mean(),100*(s.status=='match').mean(),(s.points<0).sum())
    rows.append(row)
print('mean points of 10 | answered | wrong')
print(pd.DataFrame(rows).to_string(index=False))
for k,t in runs.items(): print(k.ljust(22),'dev score/60 %.2f'%(t.points.mean()*6),' median inliers of right answers above 1300 m:',t[(t.alt>=1300)&(t.error_m/t.sigma_m<=2)].inliers.median(),' answered share above 1400 m %.0f%%'%(100*(t[t.alt>=1400].status=='match').mean()),' median s %.2f'%t.total_seconds.median())
