"""Opt-in research adapters. Patches are process-local and always restored."""
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests/Review_20261113_codex')]
from fast_edge_codex import FastEdgeAgreementPipeline
from image_processing import geometry
from semantic_rerank import dino_rerank as dino


@torch.no_grad()
def batched_dino_scores(encoder,canvas,valid,level_images,level,peaks,batch=25):
    pooled=np.full(len(peaks),np.nan)
    dense=np.full(len(peaks),np.nan)
    real=np.flatnonzero(peaks[:,1]<dino.MISSING)
    if not len(real):return pooled,dense
    mask=np.asarray(valid,np.float32)
    query=dino.token_grids(encoder,(np.asarray(canvas,np.float32)*mask)[None])[0]
    inside=dino.valid_tokens(valid).to(query.device)
    query_unit=F.normalize(query,dim=0)
    query_mean=F.normalize(query[:,inside].mean(dim=1),dim=0)
    for start in range(0,len(real),batch):
        ids=real[start:start+batch]
        windows=np.stack([dino.map_piece(level_images,level,peaks[i,1],peaks[i,2])*mask for i in ids])
        tokens=dino.token_grids(encoder,windows)
        unit=F.normalize(tokens,dim=1)
        values=(unit*query_unit[None]).sum(dim=1)
        means=F.normalize(tokens[:,:,inside].mean(dim=2),dim=1)
        # Reduce a batch without extracting one device scalar per candidate.
        dense[ids]=values[:,inside].mean(dim=1).cpu().numpy()
        pooled[ids]=(means*query_mean[None]).sum(dim=1).cpu().numpy()
    return pooled,dense


class QueryBackbone(nn.Module):
    """One-query-only wrapper, valid for eval-mode equal-size image pairs."""
    def __init__(self,original):
        super().__init__()
        assert not original.training,'Backbone reuse requires eval mode'
        self.original=original
        self.query=None
        self.saved=None
        self.encoded_images=0

    def forward(self,x):
        assert len(x)%2==0
        n=len(x)//2
        q=x[:n]
        # Validate scope rather than silently caching a different query. This
        # correctness-first prototype incurs a GPU sync; profile it on target.
        assert torch.equal(q,q[:1].expand_as(q)),'Mixed queries in one cached batch'
        if self.query is None:
            self.query=q[:1].detach().clone()
            self.saved=self.original(q[:1])
            self.encoded_images+=1
        else:
            assert torch.equal(q[:1],self.query),'Query cache reused across different images'
        candidates=self.original(x[n:])
        self.encoded_images+=n
        return tuple(torch.cat((a.expand(n,*a.shape[1:]),b),dim=0) for a,b in zip(self.saved,candidates))


@contextmanager
def cached_query_backbone(model):
    original=model.backbone
    wrapper=QueryBackbone(original)
    model.backbone=wrapper
    try:yield wrapper
    finally:model.backbone=original


@contextmanager
def cached_photo(item):
    load,normalise=geometry.load_thermal_frame,geometry.normalise_frame
    start=time.perf_counter()
    raw=load(item)
    prepared=normalise(raw)
    seconds=time.perf_counter()-start
    def cached_load(request):
        assert request['name']==item['name'],'Per-photo cache used for another image'
        return raw
    def cached_normalise(frame):
        assert frame is raw
        return prepared
    geometry.load_thermal_frame=cached_load
    geometry.normalise_frame=cached_normalise
    try:yield seconds
    finally:geometry.load_thermal_frame,geometry.normalise_frame=load,normalise


class SpeedAgreementPipeline(FastEdgeAgreementPipeline):
    def __init__(self,*args,query_backbone=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.query_backbone_codex=query_backbone

    def candidates_for_query(self,*args,**kwargs):
        if not self.query_backbone_codex:return super().candidates_for_query(*args,**kwargs)
        with cached_query_backbone(self.matcher.loftr_model):
            return super().candidates_for_query(*args,**kwargs)

    def run(self,item,prior):
        start=time.perf_counter()
        original=dino.dino_candidate_scores
        dino.dino_candidate_scores=batched_dino_scores
        try:
            with cached_photo(item) as preparation:
                result=super().run(item,prior)
            result['seconds']['render']+=preparation
            result['total']=time.perf_counter()-start
            return result
        finally:dino.dino_candidate_scores=original
