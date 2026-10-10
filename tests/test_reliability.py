import copy
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import ais
import fields
import pipeline
import sar_sim
import score
from experiments import wilson_interval, compare_methods


def ping(h, x=0, y=0):
    import geo
    lon, lat = geo.xy_to_lonlat(x,y,18.95,71.85)
    return dict(t_h=h,lon=lon,lat=lat,sog_kn=10,cog_deg=0)


def inputs(pings=None):
    det=dict(centroid=[18.95,71.85],orientation_deg=0,length_km=30)
    frames=[dict(hours_back=h,usable=True,uncertainty_km=0,
                 _xy=(np.zeros(3),np.zeros(3))) for h in [1,2,3]]
    vessel=dict(mmsi='1',name='TEST',type='Crude oil tanker',flag='Test',length_m=100,
                pings=pings if pings is not None else [ping(h) for h in [0,1,2,3,4]])
    return det,dict(frames=frames),dict(vessels=[vessel])


def test_wind_is_used_for_rendering_not_just_metadata():
    low,_,lm=sar_sim.generate_scene(17,size=64,force_oil=False,mean_wind=3)
    high,_,hm=sar_sim.generate_scene(17,size=64,force_oil=False,mean_wind=10)
    assert lm['wind_ms']==3 and hm['wind_ms']==10
    assert high.mean()>low.mean()
    with pytest.raises(ValueError):sar_sim.generate_scene(1,mean_wind=float('nan'))


def test_no_track_and_long_gap_are_unknown_not_observed():
    assert ais.resample_track([], [1,2])==[None,None]
    assert ais.resample_track([ping(0),ping(3)], [1,2])==[None,None]
    sample=ais.resample_track([ping(1),ping(1.2)], [1.1])[0]
    assert sample['interpolated'] and not sample['observed']
    assert ais.resample_track([ping(1),ping(1.2)], [1])[0]['observed']


def test_truth_does_not_cross_observation_boundary():
    _,_,traffic=inputs();traffic['vessels'][0]['truth']='guilty'
    clean=ais.observation_tracks(traffic)
    assert 'truth' not in clean['vessels'][0]
    clean['vessels'][0]['pings'][0]['lat']=0
    assert traffic['vessels'][0]['pings'][0]['lat']!=0
    with pytest.raises(ValueError):
        pipeline.analyse_observation(None,np.zeros((64,64)),traffic,fields.FieldProvider(),
                                     'test','2026-01-01T00:00:00Z',18.95,71.85)
    with pytest.raises(NotImplementedError):ais.load_tracks()


def test_missing_coverage_and_exclusion_are_distinct():
    det,dr,traffic=inputs([])
    assert score.score_vessels(det,dr,traffic,18.95,71.85)[0]['status']=='INSUFFICIENT_DATA'
    det,dr,traffic=inputs([ping(h,40000,40000) for h in [0,1,2,3,4]])
    assert score.score_vessels(det,dr,traffic,18.95,71.85)[0]['status']=='EXCLUDED'


def test_score_is_label_independent_and_weights_normalised():
    det,dr,traffic=inputs()
    before=score.score_vessels(det,dr,traffic,18.95,71.85)
    traffic['vessels'][0]['truth']='innocent'
    after=score.score_vessels(det,dr,traffic,18.95,71.85)
    assert before==after and 'truth' not in after[0]
    assert after[0]['release_hour_estimate']==2
    scaled={k:10*v for k,v in score.WEIGHTS.items()}
    assert score.score_vessels(det,dr,traffic,18.95,71.85,scaled)[0]['score']==after[0]['score']
    with pytest.raises(ValueError):score.score_vessels(det,dr,traffic,18.95,71.85,{k:0 for k in scaled})


def test_baselines_abstain_without_coverage():
    det,dr,traffic=inputs([])
    methods=compare_methods(det,dr,traffic,18.95,71.85)
    assert len(methods)==6 and all(m['abstained'] for m in methods.values())
    assert wilson_interval(0,0) is None
    lo,hi=wilson_interval(0,8)
    assert lo==0 and hi>0 # zero observed errors is not proof of zero population risk


def test_browser_drift_is_repeatable_and_finite():
    node=shutil.which('node')
    if not node:pytest.skip('Node required for browser checks')
    root=Path(__file__).resolve().parents[1]
    code='''const fs=require('fs'),vm=require('vm'); const ctx={};vm.createContext(ctx);
    vm.runInContext(fs.readFileSync('web/analyse.js','utf8'),ctx);
    const m={members:[200,201,202,203]};
    const a=ctx.SNA.backwardDrift(m,32,7,290,2,42),b=ctx.SNA.backwardDrift(m,32,7,290,2,42);
    if(JSON.stringify(a)!==JSON.stringify(b))throw Error('non deterministic');
    if(a.particles!==900||a.frames.some(f=>!Number.isFinite(f.cx)))throw Error('invalid');'''
    subprocess.run([node,'-e',code],cwd=root,check=True)


def test_browser_network_matches_pytorch(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node required for browser checks')
    import torch
    root=Path(__file__).resolve().parents[1]
    model=pipeline.load_model(str(root/'models/unet_slick.pt'))
    img,_,_=sar_sim.generate_scene(123456,size=256)
    with torch.no_grad(): ref=model(torch.from_numpy(img)[None,None]).argmax(1)[0].numpy()
    inp=tmp_path/'input.json';inp.write_text(json.dumps(img.ravel().tolist()))
    code='''const fs=require('fs'),vm=require('vm');let c={setTimeout,atob};vm.createContext(c);
    vm.runInContext(fs.readFileSync('web/engine.js','utf8'),c);
    const spec=JSON.parse(fs.readFileSync('web/weights.json'));
    c.SN.forward(c.SN.loadWeights(spec),new Float32Array(JSON.parse(fs.readFileSync(process.argv[1]))),256)
    .then(r=>process.stdout.write(JSON.stringify(Array.from(r.cls))));'''
    got=json.loads(subprocess.check_output([node,'-e',code,str(inp)],cwd=root,text=True))
    disagreement=float(np.mean(np.array(got).reshape(256,256)!=ref))
    assert disagreement<0.005,disagreement


def test_random_traffic_is_reproducible_and_labelled_only_for_evaluation():
    truth=dict(valve_open_h_before=9,valve_close_h_before=7,true_release_lon=71.85,
               true_release_lat=18.95,course_deg=30,speed_kn=10)
    first=ais.build_random_scenario(truth,18.95,71.85,seed=42)
    assert first==ais.build_random_scenario(truth,18.95,71.85,seed=42)
    assert len(first['vessels'])==11
    observed=ais.observation_tracks(first)
    assert all('truth' not in v for v in observed['vessels'])
    shifted=dict(truth,true_release_lon=72.0)
    other=ais.build_random_scenario(shifted,18.95,71.85,seed=42)
    assert first['vessels'][1:]==other['vessels'][1:]


def test_gap_factor_requires_supported_matches_on_both_sides():
    det,dr,traffic=inputs([ping(h) for h in [0,1,4,5]])
    dr['frames']=[dict(hours_back=h,usable=True,uncertainty_km=0,
                      _xy=(np.zeros(3),np.zeros(3))) for h in [1,2,3,4]]
    result=score.score_vessels(det,dr,traffic,18.95,71.85)[0]
    assert result['factors']['dark_period']>0
    assert result['hours_in_corridor']==2 # missing positions not imputed
    dr['frames']=dr['frames'][:3]
    assert score.score_vessels(det,dr,traffic,18.95,71.85)[0]['factors']['dark_period']==0
