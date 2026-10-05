import copy
import unittest
from pacdata.generate import episodes_for_group,split_groups
from pacdata.observe import observe
from pacdata.packing import PackingEnv,mask,apply_placement,greedy,generate_candidates
from pacdata.teacher import label_observation,sampled_futures
from pacdata.validate import assert_inventory,assert_allowlist

CONFIG=dict(seed=20261004,base_groups=10,min_boxes=18,max_boxes=24)


def fixture_box(identity='b',mass=1.,cap=100.,dim=None):
    return dict(box_id=identity,sku='S',dimensions_m=dim or [.1,.1,.1],mass_kg=mass,
        top_load_capacity_N=cap,allowed_yaw_deg=[0,90],measurement_valid=True,visual_damage_observed=False)


P=dict(length_m=1.2,width_m=1.,max_height_m=1.2,max_mass_kg=200.)


class CoreTests(unittest.TestCase):
    def test_group_split_and_camera_pairing(self):
        eps=list(episodes_for_group(1,CONFIG,'train'))
        self.assertEqual(len(eps),48)
        self.assertEqual(len({e['episode_id'] for e in eps}),48)
        for i in range(0,48,4):
            a=eps[i]['oracle']['arrivals']
            for e in eps[i:i+4]:self.assertEqual(e['oracle']['arrivals'],a)
        splits=split_groups(250,CONFIG['seed'])
        self.assertEqual(list(splits.values()).count('train'),175)
        self.assertEqual(list(splits.values()).count('validation'),37)
        self.assertEqual(list(splits.values()).count('test'),38)

    def test_oracle_suffix_cannot_change_policy_input_or_labels(self):
        ep=next(episodes_for_group(2,CONFIG,'train'))
        ep['camera'].update(miss_probability=0,occlusion_probability=0,quality='clean')
        state=dict(processed=0,pallet=ep['pallet'],placed=[],remeasured=[])
        a=observe(ep,state);other=copy.deepcopy(ep)
        suffix=other['oracle']['arrivals']
        j=next(i for i in range(5,len(suffix)-1) if suffix[i]['sku']!=suffix[-1]['sku'])
        suffix[-1],suffix[j]=suffix[j],suffix[-1]
        b=observe(other,state)
        self.assertEqual(a,b);assert_allowlist(a);assert_inventory(ep,a,0)
        self.assertEqual(label_observation(a,100,max_candidates=4,draws=1,depth=3),
                         label_observation(b,100,max_candidates=4,draws=1,depth=3))

    def test_occlusion_does_not_join_a_disconnected_prefix(self):
        ep=list(episodes_for_group(1,CONFIG,'train'))[3]
        ep['camera'].update(miss_probability=0,occlusion_probability=1)
        obs=PackingEnv(ep).reset()
        self.assertFalse(obs['observed_preview']);self.assertTrue(obs['unordered_visible_hints'])
        self.assertFalse(obs['camera']['coverage_complete']);assert_inventory(ep,obs,0)

    def test_unseen_sampling_preserves_quantities(self):
        ep=next(episodes_for_group(1,CONFIG,'train'));obs=PackingEnv(ep).reset()
        from collections import Counter
        for future in sampled_futures(obs,123,draws=4,depth=100):
            actual=Counter(b['sku'] for b in future)
            expected=Counter(obs['unseen_inventory'])
            for t in obs['observed_preview']:expected[t['sku']]+=1
            self.assertEqual(actual,expected)

    def test_geometry_touch_overlap_and_floating(self):
        base=fixture_box('base',mass=6,dim=[.4,.4,.2])
        placed=apply_placement(P,[],base,dict(position_m=[0,0,0],yaw_deg=0))
        c=fixture_box('current',mass=3)
        self.assertEqual(mask(P,placed,c,dict(position_m=[.4,0,0],yaw_deg=0))['status'],'ALLOW')
        self.assertIn('OVERLAP',mask(P,placed,c,dict(position_m=[.2,.2,0],yaw_deg=0))['reasons'])
        self.assertIn('SUPPORT',mask(P,placed,c,dict(position_m=[.5,0,.03],yaw_deg=0))['reasons'])

    def test_total_mass_heavy_on_light_and_cumulative_load(self):
        base=fixture_box('base',mass=6,cap=50,dim=[.4,.4,.2])
        placed=apply_placement(P,[],base,dict(position_m=[0,0,0],yaw_deg=0))
        mid=fixture_box('mid',mass=3,dim=[.4,.4,.2])
        placed=apply_placement(P,placed,mid,dict(position_m=[0,0,.2],yaw_deg=0))
        verdict=mask(P,placed,fixture_box('top',mass=3),dict(position_m=[0,0,.4],yaw_deg=0))
        self.assertIn('TOP_LOAD',verdict['reasons'])
        heavy=mask(P,placed,fixture_box('top',mass=4),dict(position_m=[0,0,.4],yaw_deg=0))
        self.assertIn('HEAVY_ON_LIGHT',heavy['reasons'])
        smallp=dict(P,max_mass_kg=9)
        self.assertIn('PALLET_MASS',mask(smallp,placed,fixture_box('top'),dict(position_m=[.5,0,0],yaw_deg=0))['reasons'])

    def test_route_and_remeasure_do_not_count_as_placement(self):
        ep=next(episodes_for_group(1,CONFIG,'train'))
        ep['oracle']['arrivals'][0]['damaged']=True
        ep['oracle']['arrivals'][1]['station_fault']='missing_mass'
        env=PackingEnv(ep);env.reset()
        obs,_,_,_=env.step(dict(type='ROUTE_DAMAGE'))
        self.assertEqual(env.state['processed'],1);self.assertEqual(len(env.state['placed']),0)
        self.assertIsNone(obs['current_box']['mass_kg'])
        obs,_,_,_=env.step(dict(type='REMEASURE'))
        self.assertEqual(env.state['processed'],1);self.assertEqual(len(env.state['placed']),0)
        self.assertTrue(obs['current_box']['measurement_valid']);assert_inventory(ep,obs,1)

    def test_resize_state_update_and_invalid_action_no_mutation(self):
        ep=next(episodes_for_group(1,CONFIG,'train'))
        values=[]
        for i in range(14):
            b=fixture_box(str(i));b.update(damaged=False,station_fault=None,gap_after_m=.1);values.append(b)
        ep['catalog']={'S':{k:v for k,v in fixture_box().items() if k in ['sku','dimensions_m','mass_kg','top_load_capacity_N','allowed_yaw_deg']}}
        ep['oracle']=dict(arrivals=values,events=[dict(type='pallet_resize',at_processed=10,new_pallet=dict(P,length_m=1.4))]);ep['pallet']=P
        env=PackingEnv(ep);obs=env.reset();before=copy.deepcopy(env.state)
        with self.assertRaises(ValueError):env.step(dict(type='PLACE',candidate=dict(position_m=[10,0,0],yaw_deg=0)))
        self.assertEqual(before,env.state)
        for i in range(10):
            obs,_,_,_=env.step(dict(type='PLACE',candidate=greedy(obs['pallet'],obs['placed_boxes'],obs['current_box'])))
        self.assertEqual(obs['pallet']['length_m'],1.4)

    def test_future_space_example(self):
        p=dict(P,length_m=1.2,width_m=.9,max_height_m=.3)
        b=fixture_box('current',mass=3,dim=[.3,.6,.3]);n=fixture_box('next',mass=8,dim=[.9,.9,.3])
        left=apply_placement(p,[],b,dict(position_m=[0,0,0],yaw_deg=0))
        center=apply_placement(p,[],b,dict(position_m=[.45,0,0],yaw_deg=0))
        self.assertTrue(generate_candidates(p,left,n))
        self.assertFalse(generate_candidates(p,center,n))

    def test_easy_control_can_complete(self):
        e=next(episodes_for_group(0,dict(CONFIG,min_boxes=48,max_boxes=80),'train'))
        env=PackingEnv(e);obs=env.reset()
        while not env.terminated:
            c=greedy(obs['pallet'],obs['placed_boxes'],obs['current_box'],8)
            self.assertIsNotNone(c)
            obs,_,_,_=env.step(dict(type='PLACE',candidate=c))
        self.assertEqual(len(env.state['placed']),len(e['oracle']['arrivals']))


if __name__=='__main__':unittest.main()
