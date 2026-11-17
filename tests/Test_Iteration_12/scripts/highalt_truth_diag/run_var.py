import sys; S=sys.argv[1]; tag=sys.argv[2]; sys.path.insert(0,S)
import hx, numpy as np, pandas as pd, warnings, cv2; warnings.filterwarnings("ignore")
from geoloc import flat_field, photo
import settings
tile={"t336":336,"t336flat":336,"t448":448,"t448flat":448}.get(tag,224)
levels=(5,6,7,8,9,10,11,12) if "l12" in tag else (5,6,7,8,9,10,11)
store,loftr,enc=hx.setup(S,tile=tile,levels=levels)
RECT=[(300,222,320,256)]
def flat(frame,pose,cam): return flat_field.mask_rects(flat_field.remove_vignette(frame,cam["cx"],cam["cy"],3),RECT)
def flatnorm(frame,pose,cam): return flat_field.local_normalise(flat(frame,pose,cam),15)
prep={"l12flatnorm":flatnorm,"lm1flatnorm":flatnorm,"flat":flat,"flatnorm":flatnorm,"t336flat":flat,"t448flat":flat,"lm1flat":flat}.get(tag)
shift={"lm1flatnorm":-1,"lm1":-1,"lm1flat":-1,"lp1":1,"lm2":-2}.get(tag,0)
if tag=="noclahe":
    def nf(frame):
        frame=frame.astype(np.float32); lo,hi=np.percentile(frame,[1,99]); return np.clip((frame-lo)/(hi-lo+1e-6),0,1)
    photo.normalise_frame=nf
m=pd.read_csv(S+"/new/images/metadata.csv"); sp=pd.read_csv(hx.ROOT+"/IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/highalt_split.csv")
m=m.merge(sp[["filename","split","band"]],on="filename"); import os
m=m[(m.alt>=800)&(m.split==os.environ.get("HX_SPLIT","dev"))].reset_index(drop=True)
rows=[]
for i,r in m.iterrows():
    try:
        o=hx.diag(store,loftr,enc,S,r.drop(["split","band"]).to_dict(),prep=prep,level_shift=shift); rows.append(o)
    except Exception as e: print("fail",r.filename,repr(e)[:200])
pd.DataFrame(rows).to_csv(S+f"/var_{tag}"+("" if os.environ.get("HX_SPLIT","dev")=="dev" else "_TEST")+".csv",index=False); print(tag,"done",len(rows),flush=True)
