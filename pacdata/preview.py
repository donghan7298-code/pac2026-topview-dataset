"""Build an offline viewer from allow-listed observations, not oracle sequences."""
import argparse
import json
from pathlib import Path
from .io import read_jsonl,write_json
from .packing import PackingEnv,greedy


def build_preview(dataset,out='examples'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    wanted=['random','large_last','occlusion_gap','measurement_recheck'];chosen={};records=[]
    for ep in read_jsonl(Path(dataset)/'train.episodes.jsonl.gz'):
        mode=ep['scenario'];key=(mode,ep['camera']['name'])
        if mode not in wanted or key in chosen or ep.get('inventory_profile')!='mixed_stress':continue
        if mode!='occlusion_gap' and ep['camera']['quality']!='clean':continue
        chosen[key]=ep
        if len(chosen)==16:break
    for mode in wanted:
        for camera in ['preview2','topview4','topview8','topview12']:
            ep=chosen[(mode,camera)];env=PackingEnv(ep);obs=env.reset();target=1 if mode=='measurement_recheck' else 3
            while not env.terminated and env.state['processed']<target:
                b=obs['current_box']
                if b['visual_damage_observed']:action=dict(type='ROUTE_DAMAGE')
                elif not b['measurement_valid']:action=dict(type='REMEASURE')
                else:
                    c=greedy(obs['pallet'],obs['placed_boxes'],b,8)
                    if c is None:break
                    action=dict(type='PLACE',candidate=c)
                obs,_,_,_=env.step(action)
            records.append(dict(scenario=mode,camera_profile=camera,processed=env.state['processed'],observation=obs))
    write_json(out/'sample_observations.json',records)
    with (out/'sample_episodes.jsonl').open('w',encoding='utf-8') as stream:
        for key,ep in list(chosen.items())[:4]:stream.write(json.dumps(ep,ensure_ascii=False)+'\n')
    template=(Path(__file__).parents[1]/'examples'/'viewer_template.html').read_text(encoding='utf-8')
    payload=json.dumps(records,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (out/'topview_viewer.html').write_text(template.replace('__OBSERVATIONS__',payload),encoding='utf-8')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/generated');p.add_argument('--out',default='examples');a=p.parse_args()
    print(f'Built {len(build_preview(a.dataset,a.out))} safe observation views')
