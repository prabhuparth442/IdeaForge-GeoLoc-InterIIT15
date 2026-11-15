"""Full NUDT map store, one level/strip at a time; no drone coordinates."""
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE/'Main_Pipeline_Headless'))
import settings
from geoloc.edge_features import edge_features, no_data_cells
from geoloc.map_store import equalise_level


def main():
    cv2.setNumThreads(1)
    out = HERE/'results/nudt_store_codex'
    out.mkdir(exist_ok=True)
    source = WORKSPACE/'Code_Testing/dataset/NUDT/Reference_map/changsha/ref.tif'
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024), rasterio.open(source) as ds:
        assert ds.transform.b == 0 and ds.transform.d == 0
        overview = ds.read([1,2,3], out_shape=(3, ds.height//16, ds.width//16), resampling=Resampling.average).mean(axis=0)
        gray_max = np.percentile(overview[overview>0], 99.5)
        for level in [3,4]:
            t = time.time()
            pixel = .3 * 2**(level/2)
            scale_x, scale_y = ds.res[0]/pixel, ds.res[1]/pixel
            h, w = int(ds.height*scale_y), int(ds.width*scale_x)
            image = np.lib.format.open_memmap(out/f'level_{level}.npy', mode='w+', dtype=np.uint8, shape=(h,w))
            for top in range(0,h,256):
                bottom = min(h,top+256)
                part = ds.read([1,2,3], window=Window(0, top/scale_y, w/scale_x, (bottom-top)/scale_y),
                               out_shape=(3,bottom-top,w), resampling=Resampling.average).astype(np.float32).mean(axis=0)
                grey = (np.clip(part/gray_max,0,1)*254+1).astype(np.uint8)
                grey[part<=0]=0
                image[top:bottom]=grey
            image[:] = equalise_level(image)
            image.flush()
            features = np.lib.format.open_memmap(out/f'features_{level}.npy',mode='w+',dtype=np.float32,shape=(8,h//4,w//4))
            for top in range(0,(h//4)*4,512):
                end = min((h//4)*4, top+512)
                first, last = max(0,top-8), min(h,end+8)
                part = edge_features(image[first:last].astype(np.float32)/255)
                skip=(top-first)//4
                features[:,top//4:end//4]=part[:,skip:skip+(end-top)//4]
            border = no_data_cells(image)
            # Work in feature strips to avoid advanced-indexing the complete array.
            lengths=[]
            for top in range(0,h//4,128):
                f = features[:,top:top+128]
                f[:,border[top:top+128]]=0
                length=np.sqrt((f*f).sum(axis=0))
                lengths.append(length.ravel())
            lengths=np.concatenate(lengths)
            typical=np.percentile(lengths[lengths>0][::50],30)
            for top in range(0,h//4,128):
                f=features[:,top:top+128]
                f /= np.sqrt((f*f).sum(axis=0))+typical+1e-3
            features.flush()
            print('NUDT full map level',level,'shape',image.shape,'seconds',round(time.time()-t,1),flush=True)
            del features,image,border,lengths,part
            gc.collect()
        info=dict(east0=ds.transform.c,north0=ds.transform.f,gsd=.3,crs=str(ds.crs),levels=[3,4],width=int(ds.width*ds.res[0]/.3),height=int(ds.height*ds.res[1]/.3),
                  source=str(source), preprocessing='headless equalisation and edge features; full raster, bounded memory')
        (out/'store.json').write_text(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
