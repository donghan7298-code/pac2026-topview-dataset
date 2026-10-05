"""Small NumPy neural score model; optional dependency for learning only."""
import argparse
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from .io import read_jsonl,write_json


def load_states(path):
    states=list(read_jsonl(path));features=[];scores=[];groups=[]
    for state in states:
        start=len(features)
        for c in state['candidate_targets']:
            features.append(c['features']);scores.append(c['score'])
        groups.append((start,len(features)))
    return states,features,scores,groups


def predict_model(model,features):
    import numpy as np
    x=(np.asarray(features,dtype=np.float64)-model['mean'])/model['scale']
    hidden=np.tanh(x@model['w1']+model['b1'])
    return (hidden@model['w2']+model['b2']).reshape(-1)


def metrics(model,features,scores,groups):
    import numpy as np
    pred=predict_model(model,features);target=np.asarray(scores)
    hits=0;regrets=[]
    for start,end in groups:
        chosen=start+int(np.argmax(pred[start:end]));best=max(target[start:end])
        regret=float(best-target[chosen]);regrets.append(regret);hits+=regret<=1e-6
    return dict(candidate_mse=float(np.mean((pred-target)**2)),states=len(groups),candidates=len(scores),
        teacher_top1_including_ties=hits/max(1,len(groups)),mean_teacher_regret=sum(regrets)/max(1,len(regrets)))


def train(labels,out,epochs=20,seed=20261006):
    import numpy as np
    labels,out=Path(labels),Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'ranker_model.json').exists():raise FileExistsError('Use a fresh model output directory')
    train_states,fx,fy,fg=load_states(labels/'train.states.jsonl.gz')
    val_states,vx,vy,vg=load_states(labels/'validation.states.jsonl.gz')
    if not fx or not vx:raise ValueError('Both train and validation must contain candidate labels')
    x=np.asarray(fx,dtype=np.float64);y=np.asarray(fy,dtype=np.float64).reshape(-1,1)
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-6]=1.
    x=(x-mean)/scale;rng=np.random.default_rng(seed)
    w1=rng.normal(0,.05,(x.shape[1],32));b1=np.zeros((1,32));w2=rng.normal(0,.05,(32,1));b2=np.asarray([[y.mean()]])
    params=[w1,b1,w2,b2];mom=[np.zeros_like(p) for p in params];vel=[np.zeros_like(p) for p in params]
    iteration=0;best_model=None;best_val=float('inf');history=[]
    for epoch in range(epochs):
        order=rng.permutation(len(x))
        for start in range(0,len(order),256):
            ids=order[start:start+256];a=x[ids];target=y[ids]
            hidden=np.tanh(a@w1+b1);prediction=hidden@w2+b2
            residual=2*(prediction-target)/len(ids)
            gh=(residual@w2.T)*(1-hidden**2)
            gradients=[a.T@gh+1e-5*w1,gh.sum(axis=0,keepdims=True),hidden.T@residual+1e-5*w2,residual.sum(axis=0,keepdims=True)]
            iteration+=1
            for i,(param,gradient) in enumerate(zip(params,gradients)):
                mom[i]=.9*mom[i]+.1*gradient;vel[i]=.999*vel[i]+.001*gradient**2
                param-=.001*(mom[i]/(1-.9**iteration))/(np.sqrt(vel[i]/(1-.999**iteration))+1e-8)
        model=dict(mean=mean,scale=scale,w1=w1,b1=b1,w2=w2,b2=b2)
        vm=metrics(model,vx,vy,vg);history.append(dict(epoch=epoch+1,validation=vm))
        if vm['candidate_mse']<best_val:
            best_val=vm['candidate_mse'];best_model={k:v.copy() for k,v in model.items()}
    # Test is read only once, after epoch selection with validation.
    _,tx,ty,tg=load_states(labels/'test.states.jsonl.gz')
    report=dict(source='SYNTHETIC_DEVELOPMENT_ONLY',model='103-feature 32-hidden tanh neural score model',
        seed=seed,epochs=epochs,selected_by='validation_candidate_mse',
        train=metrics(best_model,fx,fy,fg),validation=metrics(best_model,vx,vy,vg),
        test=metrics(best_model,tx,ty,tg) if tx else None,history=history,
        interpretation='Teacher imitation only; no claim of better real packing, robot feasibility or physical safety')
    serialized={k:v.tolist() for k,v in best_model.items()}
    serialized.update(schema_version='2.0',feature_count=x.shape[1],feature_names=json.loads((labels/'labels_manifest.json').read_text())['feature_names'],
                      model_kind='synthetic_teacher_score_regression',source='SYNTHETIC_DEVELOPMENT_ONLY')
    write_json(out/'ranker_model.json',serialized);write_json(out/'training_report.json',report)
    print(json.dumps({key:report[key] for key in ['train','validation','test']},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',default='data/labels');p.add_argument('--out',default='data/model');p.add_argument('--epochs',type=int,default=20);args=p.parse_args()
    train(args.labels,args.out,args.epochs)
