"""Fit the frozen-reference Softplus readout for SI H.3."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
from pathlib import Path
import argparse,json,time,hashlib,sys
import numpy as np,pandas as pd,torch
from scipy.spatial.distance import cdist
from reference import *
from weighted_fit import fit_original,arms,independent_reference
from data import draw_compounds

def caps_for(frame):
    labels=frame.groupby('ck').labels.first();counts=frame.ck.value_counts()
    return np.array([20/(2*(labels==y).sum()*counts[c])for c,y in zip(frame.ck,frame.labels)])

def specs():
    base=next(a for a in arms() if a['name']=='softplus_original')
    return [{**base,'name':PRIMARY+'__softplus_original','recipe':PRIMARY}]

def identity():
    files=[BASE/'code'/name for name in ['reference.py','weighted_fit.py','run_study.py']]+[BASE/'protocol.json',BASE/'assets/frozen_reference.json']
    files += [ROOT/f'05_reconstruction/data/{n}_{p}.csv'for n in ['mhedili','dilimap']for p in ['metadata','embeddings','pathways']]
    files += [ROOT/'05_reconstruction/code/reconstructed_features.py',ROOT/'01_code/10_extension/data.py']
    record={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()for f in files}
    return record,hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()

def run(output,counts,seeds,steps=1000):
    torch.set_num_threads(4);output.mkdir(parents=True,exist_ok=True)
    config=json.loads((BASE/'assets/frozen_reference.json').read_text());domains,O,masks,refs=load();s=domains['source'];train=s.loc[masks['source']]
    labs=train.groupby('ck').labels.first();spec=specs()
    if steps>1000:spec=[a for a in spec if a['recipe']==PRIMARY]
    names=[a['name']for a in spec]
    Bs={r:{d:project(f,O[d],a)for d,f in domains.items()}for r,a in config['recipes'].items()}
    Gs={d:(f[GF].to_numpy(float)-config['embedding_mean'])/config['embedding_scale']for d,f in domains.items()}
    hashes,run_id=identity();meta={'identity':run_id,'input_hashes':hashes,'specs':spec,'steps':steps,'requested_counts':counts,'requested_seeds':seeds}
    existing=output/'config.json'
    if existing.exists():assert json.loads(existing.read_text())['identity']==run_id,'Changed inputs: use new output'
    existing.write_text(json.dumps(meta,indent=2))
    for n in counts:
        for seed in (seeds if n<len(labs)else[0]):
            path=output/f'n{n}_seed{seed}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['identity']==run_id and old['steps']==steps;continue
            t=time.time();selected=draw_compounds(labs,n,seed);mask=masks['source']&s.ck.isin(selected).to_numpy();f=s.loc[mask];y=f.labels.to_numpy();cap=caps_for(f)
            G=Gs['source'][mask];K=np.exp(-cdist(G,G,'sqeuclidean')/1536);Ps=[]
            for a in spec:
                p=Bs[a['recipe']]['source'][mask].copy()
                if a.get('component')=='magnitude':p[:,1]=0
                if a.get('component')=='direction':p[:,0]=0
                Ps.append(p)
            P=np.array(Ps);fit=fit_original(K,P,y,spec,steps=steps,caps=cap)
            errors=[]
            for j,a in enumerate(spec):
                p=np.zeros_like(P[j])if a['kind']=='kernel'else P[j];v=K@fit['a'][:,j]+p@fit['w'][j]+fit['b'][j]
                obj=.5*fit['a'][:,j]@K@fit['a'][:,j]+.5*a['lambda']*(fit['w'][j]@fit['w'][j])+cap@np.maximum(0,1-(2*y-1)*v)
                errors.append(abs(obj-fit['objective'][j]))
            assert max(errors)<1e-7
            np.savez_compressed(output/f'n{n}_seed{seed}_state.npz',a=fit['a'],b=fit['b'],w=fit['w'],raw=fit['raw_parameters'],caps=cap,train_ids=f.sample_id.to_numpy(str),arm_names=np.array(names))
            rec={'identity':run_id,'n':n,'seed':seed,'steps':steps,'train_compounds':selected.tolist(),'train_profiles':len(f),'objective':fit['objective'].tolist(),'updates':fit['updates'].tolist(),'best_update':fit['best_update'].tolist(),'maximum_objective_residual':max(errors),'device':fit['device']}
            for d,ff in domains.items():
                if d=='source':continue
                take=masks[d];out=ff.loc[take];ke=np.exp(-cdist(Gs[d][take],G,'sqeuclidean')/1536);kernel=ke@fit['a'];linear=[]
                for j,a in enumerate(spec):
                    p=Bs[a['recipe']][d][take].copy()
                    if a['kind']=='kernel':p*=0
                    if a.get('component')=='magnitude':p[:,1]=0
                    if a.get('component')=='direction':p[:,0]=0
                    linear.append(p@fit['w'][j])
                linear=np.array(linear).T;full=kernel+linear+fit['b']
                np.savez_compressed(output/f'n{n}_seed{seed}_{d}.npz',full=full,linear=linear,kernel=kernel,arm_names=np.array(names),sample_id=out.sample_id.to_numpy(str),compound=out.ck.to_numpy(str),label=out.labels.to_numpy())
            rec['elapsed_seconds']=time.time()-t;path.write_text(json.dumps(rec,indent=2))
            print(f'n={n} seed={seed}: {len(spec)} early-stopping approximate solutions; {time.time()-t:.1f}s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--counts',nargs='+',type=int,default=[4,20,60,102]);p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)));p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',type=Path,default=BASE/'results/fits');a=p.parse_args();run(a.output,a.counts,a.seeds,a.steps)
