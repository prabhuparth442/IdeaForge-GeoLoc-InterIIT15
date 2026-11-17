# how much of each photo falls outside the nadir-centred canvas (pure geometry, the pipeline's own camera model, lens distortion ignored)
import sys, math, numpy as np, pandas as pd; S=sys.argv[1]
sys.path.insert(0,"/home/voidness/Storage/Synced Folders/InterIIT Tech 15/Main_Pipeline_Headless")
import settings
from geoloc import photo
from geoloc.levels import level_for_height, level_gsd
m=pd.read_csv(S+"/new/images/metadata.csv"); G=settings.GEOMETRY
u,v=np.meshgrid(np.arange(0,320,4),np.arange(0,256,4))
def run(levels,tile):
    settings.LEVELS=levels; settings.TILE=tile; out=[]
    for r in m.itertuples():
        meta={k:str(getattr(r,k)) for k in m.columns}; cam=photo.read_camera(meta); pose=photo.read_pose(meta)
        L=level_for_height(pose["height"],cam); g=level_gsd(L); H=pose["height"]*G["scale"]
        ax,ay,az=photo.camera_axes(pose["yaw"]+G["yaw_offset"],G["pitch_sign"]*pose["pitch"],G["roll_sign"]*pose["roll"])
        x=(u-cam["cx"])/cam["fx"]; y=(v-cam["cy"])/cam["fy"]
        d=x[...,None]*ax+y[...,None]*ay+az   # ray in (east, north, below-ish) coordinates of the code
        t=-H/d[...,2]; e=d[...,0]*t; n=d[...,1]*t
        half=tile*g/2; inside=(np.abs(e)<half)&(np.abs(n)<half)
        cx_off=math.hypot(*( (az[:2]*(-H/az[2])) ))
        out.append(dict(alt=pose["height"],level=L,frac_in=inside.mean(),centre_off_px=cx_off/g,photo_px_per_canvas_px=g/(pose["height"]/cam["fx"])))
    d=pd.DataFrame(out); b=pd.cut(d.alt,[0,400,800,1200,1600,1950,2100])
    return d.groupby(b,observed=True).agg(n=("alt","size"),photo_inside_canvas=("frac_in","median"),worst10pct=("frac_in",lambda s:s.quantile(.1)),centre_offset_px=("centre_off_px","median"),photo_px_per_canvas_px=("photo_px_per_canvas_px","median")).round(2)
print("TILE 224, levels 3-11"); print(run(list(range(3,12)),224).to_string())
print("TILE 224, levels 3-13"); print(run(list(range(3,14)),224).to_string())
