import pandas as pd, numpy as np, glob, os, sys
B="/home/voidness/Storage/Synced Folders/InterIIT Tech 15/"
R=B+"IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/"
RUNS=R+"photo_view/runs/"
sys.path.insert(0,R+"photo_view")
from score import pts
def meta():
    m3=pd.read_csv(B+"IdeaForge-GeoLoc-InterIIT15/dataset/Dataset_Official_3/d1/images/metadata.csv")
    sp=pd.read_csv(R+"highalt_split.csv")[['filename','split','band']]
    m3=m3.merge(sp,on='filename')
    m3['set']=m3.split.map({'dev':'work332','test':'locked168'})
    m1=pd.read_csv(B+"IdeaForge-GeoLoc-InterIIT15/dataset/d1/images/metadata.csv"); m1['set']='official500'
    nd=B+"Code_Testing/dataset/NUDT/Thermal-UAV/valid/changsha/"
    rows=[]
    for fd in sorted(glob.glob(nd+"*")):
        t=pd.read_csv(fd+"/Thermal_info.csv")
        for _,r in t.iterrows(): rows.append((os.path.basename(fd),r))
    rng=np.random.default_rng(1); pick=sorted(rng.choice(len(rows),size=min(300,len(rows)),replace=False))
    mn=[]
    for k,i in enumerate(pick):
        f,r=rows[i]; mn.append(dict(filename="nudt_%04d.png"%k,alt=r['rel_height'],yaw=r['yaw'],pitch=0.0,roll=0.0,flight=f,set='nudt300'))
    mn=pd.DataFrame(mn)
    allm=pd.concat([m3,m1,mn],ignore_index=True)
    return allm
M=meta()
def run(name):
    d=pd.read_csv(RUNS+name+".csv"); d['p']=d.apply(pts,axis=1); d['run']=name
    return d.merge(M,on='filename',how='left')
