"""Small invariance/scope checks; real network parity is in component/full runs."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
from optimisations_codex import QueryBackbone,cached_query_backbone,cached_photo,batched_dino_scores,geometry,dino


class ToyBackbone(nn.Module):
    def forward(self,x):return x*2,x*3


def test_backbone_reuses_only_identical_query():
    original=ToyBackbone().eval();wrapped=QueryBackbone(original)
    query=torch.randn(1,1,8,8);tiles=torch.randn(3,1,8,8)
    x=torch.cat((query.expand(3,-1,-1,-1),tiles))
    for expected,actual in zip(original(x),wrapped(x)):torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    wrapped(x)
    assert wrapped.encoded_images==7 # one query plus3+3 tiles
    bad=x.clone();bad[0]+=1
    try:wrapped(bad)
    except AssertionError:pass
    else:raise AssertionError('Mixed query batch was accepted')


def test_cache_restored_on_exception():
    model=nn.Module();model.backbone=ToyBackbone().eval();original=model.backbone
    try:
        with cached_query_backbone(model):raise RuntimeError('test')
    except RuntimeError:pass
    assert model.backbone is original


def test_photo_cache_does_not_survive_scope():
    raw=np.ones((16,16),np.uint16);prepared=np.ones((16,16),np.float32)
    original_load=geometry.load_thermal_frame;original_normalise=geometry.normalise_frame
    with patch.object(geometry,'load_thermal_frame',return_value=raw) as load,patch.object(geometry,'normalise_frame',return_value=prepared) as norm:
        with cached_photo(dict(name='a')):
            for _ in range(3):assert geometry.normalise_frame(geometry.load_thermal_frame(dict(name='a'))) is prepared
            try:geometry.load_thermal_frame(dict(name='b'))
            except AssertionError:pass
            else:raise AssertionError('Cross-photo cache accepted')
        assert load.call_count==1 and norm.call_count==1
    assert geometry.load_thermal_frame is original_load and geometry.normalise_frame is original_normalise


def test_empty_dino_shortlist_never_runs_encoder():
    peaks=np.zeros((3,5));peaks[:,1]=1e9
    p,d=batched_dino_scores(None,np.zeros((224,224)),np.ones((224,224),bool),{},3,peaks)
    assert np.isnan(p).all() and np.isnan(d).all()


def test_dino_vectorised_scores_match_with_missing_slots():
    rng=np.random.default_rng(15)
    canvas=rng.random((224,224),dtype=np.float32);valid=np.ones((224,224),bool);valid[:28]=False
    levels={3:(rng.random((600,600))*255).astype(np.uint8)}
    peaks=np.array([[.5,200,200,0,0],[0,1e9,1e9,0,0],[.4,240,240,0,0]])
    def fake_tokens(encoder,pictures):
        x=torch.from_numpy(pictures[:,None])
        y=torch.nn.functional.avg_pool2d(x,14)
        return torch.cat((y,y*y,torch.ones_like(y)),dim=1)
    with patch.object(dino,'token_grids',fake_tokens):
        a=dino.dino_candidate_scores(None,canvas,valid,levels,3,peaks,batch=1)
        b=batched_dino_scores(None,canvas,valid,levels,3,peaks,batch=2)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7,equal_nan=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn();print('PASS',name)
