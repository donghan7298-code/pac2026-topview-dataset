import argparse
import json
import random
from collections import Counter
from pathlib import Path
from .io import JsonlWriter, read_jsonl, stable_seed, write_json, file_sha256
from .packing import PackingEnv, greedy
from .teacher import FEATURE_NAMES,label_observation


def reservoir(path,limit,seed):
    rng=random.Random(seed);chosen=[]
    for i,ep in enumerate(read_jsonl(path)):
        if i<limit:chosen.append(ep)
        else:
            j=rng.randint(0,i)
            if j<limit:chosen[j]=ep
    return sorted(chosen,key=lambda e:e['episode_id'])


def build_labels(dataset,out,config):
    dataset,out=Path(dataset),Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'labels_manifest.json').exists():raise FileExistsError('Use a fresh labels output directory')
    stats={}
    for split,limit in config['episodes_per_split'].items():
        selected=reservoir(dataset/f'{split}.episodes.jsonl.gz',limit,stable_seed(config['seed'],split))
        writer=JsonlWriter(out/f'{split}.states.jsonl.gz');counts=Counter();modes=Counter();cams=Counter()
        try:
            for ep_index,ep in enumerate(selected):
                env=PackingEnv(ep);obs=env.reset();counts['episodes']+=1
                for _ in range(config['max_processed']*2+4):
                    if env.terminated or env.state['processed']>=config['max_processed']:break
                    box=obs['current_box']
                    if box['visual_damage_observed']:
                        obs,_,_,_=env.step(dict(type='ROUTE_DAMAGE'));counts['routed']+=1;continue
                    if not box['measurement_valid']:
                        obs,_,_,_=env.step(dict(type='REMEASURE'));counts['remeasured']+=1;continue
                    step=env.state['processed'];label=None
                    if step%config['stride']==0:
                        label=label_observation(obs,stable_seed(config['seed'],ep['episode_id'],step),
                            config['max_candidates'],config['draws'],config['depth'])
                    if label:
                        label.update(state_id=obs['request_id'],base_group=ep['base_group'],split=split,
                                     scenario=ep['scenario'],camera_profile=ep['camera']['name'])
                        writer.write(label);counts['states']+=1;counts['candidate_targets']+=len(label['candidate_targets'])
                        counts['invalid_probes']+=len(label['invalid_probes']);modes[ep['scenario']]+=1;cams[ep['camera']['name']]+=1
                    # Mixture of behavior policies to avoid collecting only one greedy trajectory.
                    if label and stable_seed(ep['episode_id'],step,'behavior')%2:
                        action=label['candidate_targets'][label['best_candidate_index']]['candidate']
                    else:action=greedy(obs['pallet'],obs['placed_boxes'],box,config['max_candidates'])
                    if action is None:
                        env.step(dict(type='STOP'));counts['stopped_no_candidate']+=1;break
                    obs,_,_,_=env.step(dict(type='PLACE',candidate=action))
                if (ep_index+1)%20==0:print(f'{split}: labeled {ep_index+1}/{len(selected)} episodes, {counts["candidate_targets"]} candidate targets',flush=True)
        finally:writer.close()
        stats[split]=dict(counts,scenario_states=dict(modes),camera_states=dict(cams))
    manifest=dict(source='SYNTHETIC_DEVELOPMENT_ONLY',config=config,splits=stats,feature_names=FEATURE_NAMES,
        label_kind='BOUNDED_MONTE_CARLO_GREEDY_TEACHER_NOT_OPTIMAL',
        teacher_uses='observation only; unseen SKU counts sampled without replacement',
        robot_feasibility='NOT_CHECKED',files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in out.glob('*.gz')})
    write_json(out/'labels_manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/generated');p.add_argument('--out',default='data/labels');p.add_argument('--config',default='configs/labels.json');args=p.parse_args()
    build_labels(args.dataset,args.out,json.loads(Path(args.config).read_text(encoding='utf-8')))
