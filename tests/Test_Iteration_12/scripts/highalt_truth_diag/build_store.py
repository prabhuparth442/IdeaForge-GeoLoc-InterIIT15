# local map store (levels 5-12) from the 1.12 m copy of tile 2684532111, using the pipeline's own level and feature code
import sys, json, os, numpy as np, cv2
sys.path.insert(0, "/home/voidness/Storage/Synced Folders/InterIIT Tech 15/Main_Pipeline_Headless")
import settings
from geoloc.map_store import equalise_level
from geoloc.edge_features import map_edge_features
from geoloc.levels import level_gsd
S=sys.argv[1]; out=S+"/store"; os.makedirs(out,exist_ok=True)
m=np.load(S+"/map_1p12.npy",mmap_mode="r"); SRC=1.12
settings.LEVELS=[5,6,7,8,9,10,11,12]
ov=m[::16,::16]; gray_max=float(np.percentile(ov[ov>0],99.5)); print("gray_max",gray_max, "p1,p50,p99",np.percentile(ov[ov>0],[1,50,99]))
for L in settings.LEVELS:
    g=level_gsd(L); h=int(m.shape[0]*SRC/g); w=int(m.shape[1]*SRC/g)
    img=np.zeros((h,w),np.uint8); step=4096
    for top in range(0,m.shape[0],step):
        part=np.asarray(m[top:top+step]).astype(np.float32)
        r0=int(round(top*SRC/g)); r1=min(int(round((top+part.shape[0])*SRC/g)),h)
        if r1>r0:
            small=cv2.resize(part,(w,r1-r0),interpolation=cv2.INTER_AREA)
            u=np.clip(small/gray_max,0,1)*254+1; u[small<=0]=0; img[r0:r1]=u.astype(np.uint8)
    pic=equalise_level(img); np.save(out+"/level_%d.npy"%L,pic); np.save(out+"/features_%d.npy"%L,map_edge_features(pic)); print("level",L,pic.shape,flush=True)
json.dump(dict(east0=735396.7371938479,north0=3014418.7462867238,gsd=0.3,crs="EPSG:32643",levels=settings.LEVELS,width=int(m.shape[1]*SRC/0.3),height=int(m.shape[0]*SRC/0.3)),open(out+"/store.json","w"))
