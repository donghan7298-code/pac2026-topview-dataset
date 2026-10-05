import argparse
import json
import math
from collections import Counter
from pathlib import Path
from .io import read_jsonl,file_sha256,write_json
from .packing import PackingEnv,mask,bounds,dims
from .observe import observe


def assert_inventory(ep,obs,step):
    remaining=Counter(b['sku'] for b in ep['oracle']['arrivals'][step:])
    actual=Counter(obs['unseen_inventory'])
    if obs['current_box']:actual[obs['current_box']['sku']]+=1
    for track in obs['observed_preview']:actual[track['sku']]+=1
    assert actual==remaining,(ep['episode_id'],step,actual,remaining)
    assert all(n>=0 for n in obs['unseen_inventory'].values())
    # Hints are deliberately not subtracted again.
    ranks=[t['verified_order_rank'] for t in obs['observed_preview']]
    assert ranks==list(range(1,len(ranks)+1))


def assert_allowlist(obs):
    expected={'schema_version','request_id','pallet','placed_boxes','state_verified','current_box',
        'observed_preview','unordered_visible_hints','unseen_inventory','catalog','camera','robot_feasibility'}
    assert set(obs)==expected
    serialized=json.dumps(obs)
    for key in ['"oracle"','"arrivals"','"scenario"','"split"','"base_group"','"station_fault"','"gap_after_m"']:
        assert key not in serialized,key


def validate(dataset,labels=None):
    root=Path(dataset);manifest=json.loads((root/'manifest.json').read_text())
    group_split={};ids=set();group_fingerprints={};stats={};preview_lengths=Counter();checks=0
    for split in ('train','validation','test'):
        episodes=0;arrivals=0
        path=root/f'{split}.episodes.jsonl.gz'
        assert file_sha256(path)==manifest['files'][path.name]['sha256']
        for ep in read_jsonl(path):
            episodes+=1;values=ep['oracle']['arrivals'];arrivals+=len(values)
            assert ep['split']==split;assert ep['episode_id'] not in ids;ids.add(ep['episode_id'])
            group=ep['base_group'];assert group_split.get(group,split)==split;group_split[group]=split
            fingerprint=sorted(b['box_id'] for b in values)
            assert len(fingerprint)==len(set(fingerprint))
            assert group_fingerprints.get(group,fingerprint)==fingerprint;group_fingerprints[group]=fingerprint
            assert dict(Counter(b['sku'] for b in values))==ep['initial_inventory']
            for box in values:
                assert all(math.isfinite(v) and v>0 for v in box['dimensions_m'])
                assert box['mass_kg']>0 and box['top_load_capacity_N']>0
            # All input release points, including completion; placement state not fabricated.
            for step in range(len(values)+1):
                state=dict(processed=step,pallet=ep['pallet'],placed=[],remeasured=[])
                obs=observe(ep,state);assert_inventory(ep,obs,step);assert_allowlist(obs)
                preview_lengths[len(obs['observed_preview'])]+=1;checks+=1
        assert episodes==manifest['splits'][split]['episodes'];assert arrivals==manifest['splits'][split]['arrivals']
        stats[split]=dict(episodes=episodes,arrivals=arrivals)
        print(f'{split}: episode schema, split groups and all observations passed',flush=True)
    label_stats={}
    if labels:
        folder=Path(labels);lm=json.loads((folder/'labels_manifest.json').read_text())
        for split in ('train','validation','test'):
            rows=0;states=0;path=folder/f'{split}.states.jsonl.gz'
            assert file_sha256(path)==lm['files'][path.name]['sha256']
            for record in read_jsonl(path):
                states+=1;assert record['split']==split;assert group_split[record['base_group']]==split
                obs=record['observation'];assert_allowlist(obs)
                assert all(math.isfinite(v) for c in record['candidate_targets'] for v in c['features'])
                for c in record['candidate_targets']:
                    assert len(c['features'])==len(lm['feature_names'])
                    assert mask(obs['pallet'],obs['placed_boxes'],obs['current_box'],c['candidate'])['status']=='ALLOW'
                    assert math.isfinite(c['score']);rows+=1
                for c in record['invalid_probes']:
                    assert mask(obs['pallet'],obs['placed_boxes'],obs['current_box'],c['candidate'])['status']==c['status']
                assert record['candidate_targets'][record['best_candidate_index']]['score']==max(c['score'] for c in record['candidate_targets'])
                # Rebuild each recorded placement state with the same hard rules.
                rebuilt=[]
                from .packing import apply_placement
                for b in obs['placed_boxes']:
                    rebuilt=apply_placement(obs['pallet'],rebuilt,dict(b,measurement_valid=True,visual_damage_observed=False),dict(position_m=b['position_m'],yaw_deg=b['yaw_deg']))
                assert rebuilt==obs['placed_boxes']
            assert rows==lm['splits'][split].get('candidate_targets',0);assert states==lm['splits'][split].get('states',0)
            label_stats[split]=dict(states=states,candidate_targets=rows)
            print(f'{split}: all teacher candidates and placement histories passed',flush=True)
    report=dict(status='PASS',source='SYNTHETIC_DEVELOPMENT_ONLY',groups=len(group_split),
        observation_checks=checks,episodes=len(ids),splits=stats,verified_preview_lengths=dict(sorted(preview_lengths.items())),
        labels=label_stats,robot_feasibility='NOT_CHECKED',
        limits='No real camera, physics, robot path or optimality validation')
    write_json(root/'validation_report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/generated');p.add_argument('--labels');args=p.parse_args()
    print(json.dumps(validate(args.dataset,args.labels),ensure_ascii=False),flush=True)
