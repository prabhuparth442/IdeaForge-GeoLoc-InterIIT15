import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from encoder_codex import gate, structural
from run_pipeline_codex import select, measure, verify_frozen


def result(points, counts, cosine):
    peaks=np.array([[1,x,y,0,0] for x,y in points],dtype=float)
    return dict(peaks=peaks,z=np.zeros(len(points)),stage2=dict(inliers=np.array(counts),shifts=np.zeros((len(points),2))),
                satellite_cosine=np.array(cosine))


def test_satellite_split_footprints_and_hashes():
    manifest=verify_frozen()
    assert manifest['provenance'].startswith('Random initialisation')
    info=json.loads((HERE/'results/map_windows_codex.json').read_text())
    width=info['source']['shape'][1]
    for w in info['windows']:
        lo,hi={'train':(0,.55),'validation':(.60,.75),'calibration':(.80,1)}[w['split']]
        assert w['x']>=int(lo*width)+256
        assert w['x']+w['side']<=int(hi*width)-256
    assert len(info['windows'])==896


def test_gate_rejects_missing_competitor_and_nonfinite():
    cal=dict(cosine=.79,margin=.065)
    assert gate(.9,.1,cal)
    assert not gate(.9,.1,cal,False)
    assert not gate(np.nan,.1,cal)
    assert not gate(.9,np.inf,cal)
    assert not gate(.9,.01,cal)


def test_polarity_invariance():
    x=np.random.default_rng(3).random((224,224),dtype=np.float32)
    np.testing.assert_allclose(structural(x),structural(1-x),atol=3e-5)


def test_distinct_location_ambiguity_abstains():
    outputs,_=select([result([(0,0),(100,0)],[12,10],[.9,.89])],8,dict(cosine=.79,margin=.065))
    assert not outputs['sat_veto']['answered']
    assert not outputs['sat_select']['answered']


def test_same_location_duplicates_do_not_erase_margin():
    outputs,_=select([result([(0,0),(1,1),(100,0)],[12,10,0],[.9,.9,.2])],8,dict(cosine=.79,margin=.065))
    assert outputs['sat_veto']['answered']
    assert outputs['sat_select']['answered']


def test_weak_candidate_requires_independent_query_agreement():
    a=result([(0,0),(100,0)],[4,0],[.9,.2])
    one,_=select([a],8,dict(cosine=.79,margin=.065))
    assert not one['sat_select']['answered']
    two,_=select([a,result([(1,1),(100,0)],[3,0],[.9,.2])],8,dict(cosine=.79,margin=.065))
    assert two['sat_select']['answered']
    assert not two['sat_veto']['answered']


def test_empty_and_failed_inputs_count_as_zero():
    outputs,records=select([],8,dict(cosine=.79,margin=.065))
    assert records==[] and not outputs['sat_select']['answered']
    assert measure({},(0,0),8)['points']==0
    assert measure(dict(answered=True,x=100,y=0),(0,0),8)['points']==-20
