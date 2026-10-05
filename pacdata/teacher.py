"""Teacher scores use only policy observations, never oracle future order."""
import copy
import math
import random
from .packing import apply_placement, generate_candidates, greedy, mask, dims
from .io import stable_seed

FEATURE_NAMES=(['candidate_x','candidate_y','candidate_z','yaw_90','box_l','box_w','box_h','box_mass',
    'post_cog_x','post_cog_y','post_cog_z','state_volume_ratio','placed_count','verified_preview_count',
    'unordered_hint_count','unseen_count','sensor_sigma','remaining_nominal_volume_ratio']+
    [f'preview_{i}_{name}' for i in range(12) for name in ['l','w','h','mass','present']]+
    [f'post_height_{y}_{x}' for y in range(5) for x in range(5)])


def features(obs,candidate):
    p=obs['pallet'];b=obs['current_box'];d=dims(b,candidate['yaw_deg']);xyz=candidate['position_m']
    limits=[p['length_m'],p['width_m'],p['max_height_m']];volume=math.prod(limits)
    boxes=list(obs['placed_boxes'])+[dict(b,position_m=xyz,yaw_deg=candidate['yaw_deg'])]
    total=sum(box['mass_kg'] for box in boxes)
    cog=[sum(box['mass_kg']*(box['position_m'][a]+dims(box,box.get('yaw_deg',0))[a]/2) for box in boxes)/total for a in range(3)]
    nominal=sum(n*math.prod(obs['catalog'][sku]['dimensions_m']) for sku,n in obs['unseen_inventory'].items())
    v=[xyz[i]/limits[i] for i in range(3)]+[float(candidate['yaw_deg']==90)]+[d[i]/limits[i] for i in range(3)]+[b['mass_kg']/p['max_mass_kg']]
    v += [cog[i]/limits[i] for i in range(3)]+[sum(math.prod(box['dimensions_m']) for box in obs['placed_boxes'])/volume,
        len(obs['placed_boxes'])/100,len(obs['observed_preview'])/12,len(obs['unordered_visible_hints'])/12,
        sum(obs['unseen_inventory'].values())/100,obs['camera']['dimension_sigma_m']/.02,nominal/volume]
    for i in range(12):
        if i<len(obs['observed_preview']):
            t=obs['observed_preview'][i]
            v += [t['dimensions_m'][a]/limits[a] for a in range(3)]+[t['nominal_mass_kg']/p['max_mass_kg'],1.]
        else:v += [0.]*5
    for gy in range(5):
        for gx in range(5):
            x=(gx+.5)*p['length_m']/5;y=(gy+.5)*p['width_m']/5;h=0.
            for box in boxes:
                bd=dims(box,box.get('yaw_deg',0));lo=box['position_m']
                if lo[0]<=x<=lo[0]+bd[0] and lo[1]<=y<=lo[1]+bd[1]:h=max(h,lo[2]+bd[2])
            v.append(h/p['max_height_m'])
    assert len(v)==len(FEATURE_NAMES)
    return [round(value,6) for value in v]


def sampled_futures(obs,seed,draws=2,depth=6):
    futures=[]
    for draw in range(draws):
        rng=random.Random(stable_seed(seed,draw));sequence=[]
        for track in obs['observed_preview'][:depth]:
            # Draw geometric uncertainty; height and mass come from catalog.
            d=[max(.01,track['dimensions_m'][a]+rng.gauss(0,track['dimension_sigma_m'])) for a in (0,1)]
            d.append(track['dimensions_m'][2])
            sequence.append(dict(box_id=track['track_id'],sku=track['sku'],dimensions_m=d,
                mass_kg=track['nominal_mass_kg'],top_load_capacity_N=track['top_load_capacity_N'],
                allowed_yaw_deg=track['allowed_yaw_deg'],measurement_valid=True,visual_damage_observed=False))
        pool=[sku for sku,n in obs['unseen_inventory'].items() for _ in range(n)]
        rng.shuffle(pool)
        for i,sku in enumerate(pool[:max(0,depth-len(sequence))]):
            b=copy.deepcopy(obs['catalog'][sku]);b.update(box_id=f'sample_{draw}_{i}',measurement_valid=True,visual_damage_observed=False)
            b['dimensions_m']=[v*rng.uniform(.99,1.01) for v in b['dimensions_m']]
            sequence.append(b)
        futures.append(sequence)
    return futures


def teacher_score(obs,candidate,futures,max_future_candidates=8):
    p=obs['pallet'];box=obs['current_box']
    placed=apply_placement(p,obs['placed_boxes'],box,candidate)
    ratios=[];volumes=[]
    for future in futures:
        state=placed;count=0;extra_volume=0.
        for b in future:
            action=greedy(p,state,b,max_future_candidates)
            if action is None:break
            state=apply_placement(p,state,b,action);count+=1;extra_volume+=math.prod(b['dimensions_m'])
        ratios.append(count/max(1,len(future)))
        volumes.append(extra_volume/(p['length_m']*p['width_m']*p['max_height_m']))
    mean=sum(ratios)/len(ratios);low=min(ratios);fv=sum(volumes)/len(volumes)
    total=sum(b['mass_kg'] for b in placed)
    cogx=sum(b['mass_kg']*(b['position_m'][0]+dims(b,b['yaw_deg'])[0]/2) for b in placed)/total/p['length_m']
    cogy=sum(b['mass_kg']*(b['position_m'][1]+dims(b,b['yaw_deg'])[1]/2) for b in placed)/total/p['width_m']
    balance=1-min(1,abs(cogx-.5)+abs(cogy-.5))
    height=max(b['position_m'][2]+dims(b,b['yaw_deg'])[2] for b in placed)/p['max_height_m']
    score=.65*mean+.15*low+.15*fv+.05*balance-.05*height
    return dict(score=round(score,6),mean_future_fit=round(mean,6),worst_sample_fit=round(low,6),
                expected_future_volume_ratio=round(fv,6),post_balance_proxy=round(balance,6),post_height_ratio=round(height,6))


def label_observation(obs,seed,max_candidates=8,draws=2,depth=6):
    box=obs['current_box']
    if box is None or not box['measurement_valid'] or box['visual_damage_observed']:return None
    candidates=generate_candidates(obs['pallet'],obs['placed_boxes'],box,max_candidates)
    if not candidates:return None
    futures=sampled_futures(obs,seed,draws,depth)
    targets=[]
    for c in candidates:
        targets.append(dict(candidate=c,mask_status='ALLOW',features=features(obs,c),**teacher_score(obs,c,futures)))
    best=max(range(len(targets)),key=lambda i:targets[i]['score'])
    invalid=[]
    probes=[dict(candidate_id='probe_boundary',position_m=[obs['pallet']['length_m'],0,0],yaw_deg=0),
            dict(candidate_id='probe_floating',position_m=[0,0,.031],yaw_deg=0)]
    if obs['placed_boxes']:
        probes.append(dict(candidate_id='probe_overlap',position_m=obs['placed_boxes'][0]['position_m'],yaw_deg=0))
    for c in probes:
        verdict=mask(obs['pallet'],obs['placed_boxes'],box,c)
        if verdict['status']!='ALLOW':invalid.append(dict(candidate=c,**verdict))
    return dict(observation=obs,candidate_targets=targets,invalid_probes=invalid,best_candidate_index=best,
                label_kind='BOUNDED_MONTE_CARLO_GREEDY_TEACHER_NOT_OPTIMAL',teacher_seed=seed,
                teacher_config=dict(draws=draws,depth=depth,max_candidates=max_candidates,max_future_candidates=8))
