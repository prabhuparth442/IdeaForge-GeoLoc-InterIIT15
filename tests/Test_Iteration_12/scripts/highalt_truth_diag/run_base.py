import sys; S=sys.argv[1]; tag=sys.argv[2]; sys.path.insert(0,S)
import hx, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
store,loftr,enc=hx.setup(S)
m=pd.read_csv(S+"/new/images/metadata.csv"); sp=pd.read_csv(hx.ROOT+"/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/highalt_split.csv")
m=m.merge(sp[["filename","split","band"]],on="filename"); m=m[m.alt>=170].reset_index(drop=True)
rows=[]
for i,r in m.iterrows():
    try:
        o=hx.diag(store,loftr,enc,S,r.drop(["split","band"]).to_dict()); o["split"]=r["split"]; rows.append(o)
    except Exception as e: print("fail",r.filename,e)
    if i%50==0: print(i,len(m),flush=True)
pd.DataFrame(rows).to_csv(S+f"/diag_{tag}.csv",index=False)
