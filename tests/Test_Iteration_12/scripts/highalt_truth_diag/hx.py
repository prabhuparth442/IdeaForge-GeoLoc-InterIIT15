"""Truth-place diagnostic harness: the submission pipeline's own stages, run one query at a time with the true place known."""
import sys, os, math, numpy as np, pandas as pd, cv2, torch
ROOT="/home/voidness/Storage/Synced Folders/InterIIT Tech 15"
sys.path.insert(0, ROOT+"/Main_Pipeline_Headless")
import settings
from geoloc import photo, fine_matching, semantic_rerank, scoring, flat_field
from geoloc.coarse_search import score_map_for_query, pick_peaks, DEVICE
from geoloc.levels import level_for_height, level_gsd, map_piece
from geoloc.map_store import MapStore

def setup(S, tile=224, levels=(5,6,7,8,9,10,11)):
    settings.LEVELS=list(levels); settings.TILE=tile
    store=MapStore(S+"/store")
    loftr=fine_matching.load_loftr(ROOT+"/Main_Pipeline_Headless/models/"+settings.LOFTR_FILE)
    enc=semantic_rerank.load_encoder(ROOT+"/Main_Pipeline_Headless/models/"+settings.DINO_FILE)
    return store, loftr, enc

def jitter(name, radius=5000.0):
    rng=np.random.default_rng(abs(hash(name))%(2**32)); r=radius*math.sqrt(rng.random()); a=rng.random()*2*math.pi
    return r*math.cos(a), r*math.sin(a)

def stable_jitter(name, radius=5000.0):
    seed=int(name[:8],16); rng=np.random.default_rng(seed); r=radius*math.sqrt(rng.random()); a=rng.random()*2*math.pi
    return r*math.cos(a), r*math.sin(a)

def refine_px(canvas, valid, level, x, y, level_image, reach=6):
    """Pixel-step refinement of a coarse place: masked ZNCC of the edge cells for every 1 px shift of the map window, then a parabola. Returns x, y, score."""
    from geoloc.edge_features import query_edge_features, edge_features, normalise_cells
    gsd=level_gsd(level); q,cells=query_edge_features(canvas,valid); n=cells.sum()
    qz=(q-q.sum(axis=(1,2),keepdims=True)/n)*cells; qn=math.sqrt((qz**2).sum())+1e-9
    T=settings.TILE; big=map_piece(level_image,level,x,y,T+2*reach+8)
    sc=np.full((2*reach+1,2*reach+1),-1.0,np.float32)
    for i,dy in enumerate(range(-reach,reach+1)):
        for j,dx in enumerate(range(-reach,reach+1)):
            win=big[reach+4+dy:reach+4+dy+T, reach+4+dx:reach+4+dx+T]
            f=normalise_cells(edge_features(win))*cells
            fz=(f-f.sum(axis=(1,2),keepdims=True)/n)*cells
            sc[i,j]=(fz*qz).sum()/(qn*math.sqrt((fz**2).sum())+1e-9)
    i,j=np.unravel_index(sc.argmax(),sc.shape); oy=ox=0.0
    if 0<i<2*reach and 0<j<2*reach:
        a,b,c=sc[i-1,j],sc[i,j],sc[i+1,j]; oy=0.5*(a-c)/(a-2*b+c-1e-9)
        a,b,c=sc[i,j-1],sc[i,j],sc[i,j+1]; ox=0.5*(a-c)/(a-2*b+c-1e-9)
    return x+(j-reach+ox)*gsd, y+(i-reach+oy)*gsd, float(sc[i,j])

def diag(store, loftr, enc, S, row, heading=0.0, scale=1.0, level_shift=0, prep=None, keep=False):
    """One query on one photo. prep(frame, pose, camera) -> frame replaces the raw frame before the pipeline's stretch (None = as is)."""
    meta={k:str(v) for k,v in row.items()}
    cam=photo.read_camera(meta); pose=photo.read_pose(meta)
    level=int(np.clip(level_for_height(pose["height"],cam)+level_shift,min(settings.LEVELS),max(settings.LEVELS)))
    gsd=level_gsd(level); sig=scoring.sigma_m(pose["height"],cam["fx"])
    tx,ty=store.gps_to_metres(float(meta["lat_gtruth"]),float(meta["lon_gtruth"]))
    jx,jy=stable_jitter(meta["filename"]); prior=(tx+jx,ty+jy)
    frame=photo.load_photo(S+"/new/images/"+meta["filename"])
    if prep is not None: frame=prep(frame,pose,cam)
    pic=photo.normalise_frame(frame)
    blur=max(0.0,0.5*gsd/(pose["height"]/cam["fx"])-0.25)
    if blur>0.2: pic=cv2.GaussianBlur(pic,(0,0),blur)
    canvas,valid=photo.draw_canvas(pic,pose,cam,level,heading,scale)
    out=dict(filename=meta["filename"],alt=pose["height"],level=level,sigma=sig,valid_frac=float(valid.mean()),
             tilt=math.hypot(pose["pitch"],pose["roll"]))
    cache={}
    sm,geo=score_map_for_query(canvas,valid,level,prior,store.features,cache)
    xs=geo["x_m"].cpu().numpy(); ys=geo["y_m"].cpu().numpy(); sm_np=sm.cpu().numpy()
    ci=int(np.argmin(np.abs(xs-tx))); ri=int(np.argmin(np.abs(ys-ty)))
    r0,r1,c0,c1=max(ri-1,0),ri+2,max(ci-1,0),ci+2
    tscore=float(sm_np[r0:r1,c0:c1].max()); out["truth_score"]=tscore
    usable=sm_np>-1
    far=usable.copy(); g=settings.TILE//settings.CELL//2; far[max(ri-g,0):ri+g+1,max(ci-g,0):ci+g+1]=False
    vals=sm_np[far]
    out["truth_rank_cells"]=int((vals>tscore).sum()); out["n_cells"]=int(far.sum())
    out["truth_z"]=float((tscore-vals.mean())/(vals.std()+1e-9)); out["max_far_z"]=float((vals.max()-vals.mean())/(vals.std()+1e-9)) if len(vals) else np.nan
    short=pick_peaks(sm,geo,settings.K_PEAKS)
    d=np.hypot(short[:,1]-tx,short[:,2]-ty); out["truth_peak_rank"]=int(np.argmax(d<=2*settings.CELL*gsd+1e-6)) if (d<=2*settings.CELL*gsd+1e-6).any() else -1
    allv=sm_np[usable]; mu,sd=allv.mean(),allv.std()+1e-9
    out["p1z"]=float((short[0,0]-mu)/sd); out["p2z"]=float((short[1,0]-mu)/sd); out["top1_right"]=bool(d[0]<=2*settings.CELL*gsd+1e-6)
    out["p1_sharp"]=float(short[0,3]/sd)
    out["coarse_err"]=float(d[0]); out["ref_err"]=np.nan
    if os.environ.get("HX_REFINE")=="1":
        rx,ry,rs=refine_px(canvas,valid,level,short[0,1],short[0,2],store.levels[level]); out["ref_err"]=float(math.hypot(rx-tx,ry-ty))
    peaks,z=semantic_rerank.semantic_layer(enc,sm,geo,canvas,valid,level,store.levels[level])
    dM=np.hypot(peaks[:,1]-tx,peaks[:,2]-ty); out["truth_in_M"]=bool((dM<=2*settings.CELL*gsd+1e-6).any())
    inj=np.vstack([peaks,[[tscore,tx,ty,0,0]]])
    st=fine_matching.check_places(loftr,canvas,valid,level,inj,store.levels[level])
    inl=st["inliers"]; sh=st["shifts"]
    out["sh_e"]=float(sh[-1,0]); out["sh_s"]=float(sh[-1,1]); out["yaw"]=pose["yaw"]; out["pitch"]=pose["pitch"]; out["roll"]=pose["roll"]; out["inl_truth"]=float(inl[-1]); out["err_truth"]=float(math.hypot(sh[-1,0],sh[-1,1])) if inl[-1]>0 else np.nan
    ex=inj[:-1,1]+sh[:-1,0]-tx; ey=inj[:-1,2]+sh[:-1,1]-ty; e=np.hypot(ex,ey)
    wrong=e>8*sig; out["max_inl_wrong"]=float(inl[:-1][wrong].max()) if wrong.any() else 0.0
    right=e<=2*sig; out["max_inl_right_M"]=float(inl[:-1][right].max()) if right.any() else 0.0
    b=int(np.argmax(inl[:-1])); out["best_inl"]=float(inl[b]); out["best_err_sig"]=float(e[b]/sig)
    if keep: out["_canvas"]=canvas; out["_valid"]=valid; out["_map"]=map_piece(store.levels[level],level,tx,ty,settings.TILE)
    return out
