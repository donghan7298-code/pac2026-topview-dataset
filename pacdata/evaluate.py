"""Paired camera comparison using identical inventory/order and candidate budgets."""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from .io import read_jsonl,write_json
from .packing import PackingEnv,generate_candidates,immediate_score
from .teacher import features,FEATURE_NAMES


def percentile95(values):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*.95)-1)] if values else 0.


def run_policy(ep,model=None,budget=8):
    env=PackingEnv(ep);obs=env.reset();times=[];observed=[];reason=None
    for _ in range(len(ep['oracle']['arrivals'])*2+4):
        if env.terminated:break
        b=obs['current_box'];observed.append(len(obs['observed_preview']))
        if b['visual_damage_observed']:
            obs,_,_,info=env.step(dict(type='ROUTE_DAMAGE'));reason=info['reason'];continue
        if not b['measurement_valid']:
            obs,_,_,info=env.step(dict(type='REMEASURE'));reason=info['reason'];continue
        start=time.perf_counter()
        candidates=generate_candidates(obs['pallet'],obs['placed_boxes'],b,budget)
        if not candidates:
            times.append((time.perf_counter()-start)*1000);obs,_,_,info=env.step(dict(type='STOP'));reason=info['reason'];break
        if model:
            from .train_ranker import predict_model
            scores=predict_model(model,[features(obs,c) for c in candidates])
            action=candidates[int(scores.argmax())]
        else:action=max(candidates,key=lambda c:immediate_score(obs['pallet'],obs['placed_boxes'],b,c))
        times.append((time.perf_counter()-start)*1000)
        obs,_,_,info=env.step(dict(type='PLACE',candidate=action));reason=info['reason']
    p=env.state['pallet']
    return dict(episode_id=ep['episode_id'],base_group=ep['base_group'],scenario=ep['scenario'],inventory_profile=ep.get('inventory_profile','mixed_stress'),
        camera_profile=ep['camera']['name'],complete=env.state['processed']==len(ep['oracle']['arrivals']),
        input_count=len(ep['oracle']['arrivals']),processed=env.state['processed'],placed=len(env.state['placed']),routed=env.state['routed'],
        volume_ratio=sum(math.prod(b['dimensions_m']) for b in env.state['placed'])/(p['length_m']*p['width_m']*p['max_height_m']),
        decision_ms_median=statistics.median(times) if times else 0.,decision_ms_p95=percentile95(times),
        average_verified_preview=statistics.mean(observed) if observed else 0.,last_reason=reason,
        robot_feasibility='NOT_CHECKED')


def evaluate(dataset,out,model_path=None,families=12,seed=20261007):
    episodes=list(read_jsonl(Path(dataset)/'test.episodes.jsonl.gz'));by_family={}
    for ep in episodes:by_family.setdefault((ep['base_group'],ep['scenario']),[]).append(ep)
    keys=sorted(by_family);random.Random(seed).shuffle(keys);keys=keys[:families]
    policies={'greedy':None}
    if model_path:
        import numpy as np
        model=json.loads(Path(model_path).read_text())
        assert model['feature_names']==FEATURE_NAMES
        policies['neural_score']={k:np.asarray(model[k]) for k in ['mean','scale','w1','b1','w2','b2']}
    rows=[];summary={}
    for name,model in policies.items():
        for key in keys:
            family=by_family[key]
            assert len(family)==4
            for ep in family:rows.append(dict(policy=name,**run_policy(ep,model)))
        values=[r for r in rows if r['policy']==name];per_camera={}
        for camera in ('preview2','topview4','topview8','topview12'):
            subset=[r for r in values if r['camera_profile']==camera]
            per_camera[camera]=dict(episodes=len(subset),complete_rate=statistics.mean(r['complete'] for r in subset),
                mean_placed=statistics.mean(r['placed'] for r in subset),mean_volume_ratio=statistics.mean(r['volume_ratio'] for r in subset),
                mean_verified_preview=statistics.mean(r['average_verified_preview'] for r in subset),
                mean_episode_decision_p95_ms=statistics.mean(r['decision_ms_p95'] for r in subset))
        summary[name]=per_camera
        summary[name]['inventory_profiles']={profile:dict(episodes=len(subset),
            complete_rate=statistics.mean(r['complete'] for r in subset),mean_placed=statistics.mean(r['placed'] for r in subset))
            for profile in ('homogeneous_control','mixed_stress')
            if (subset:=[r for r in values if r['inventory_profile']==profile])}
    result=dict(source='SYNTHETIC_DEVELOPMENT_ONLY',seed=seed,test_families=len(keys),same_candidate_budget=8,
        constraints='single pallet; no rearrangement; no arrival reorder; no robot path model',
        summaries=summary,episodes=rows,
        interpretation='Small paired synthetic test, not an official benchmark or evidence of real system superiority')
    write_json(out,result);print(json.dumps(summary,ensure_ascii=False),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/generated');p.add_argument('--model');p.add_argument('--out',default='comparison_report.json');p.add_argument('--families',type=int,default=12);args=p.parse_args()
    evaluate(args.dataset,args.out,args.model,args.families)
