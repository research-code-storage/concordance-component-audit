"""Fit the reported predictors from the supplied feature matrices."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,json,time,hashlib,sys
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from reconstructed_features import *
sys.path.insert(0,str(ROOT/'01_code/11_legacy_review'))
from original_fit import fit_original,arms
from data import aggregate
from core import rbf,objective,rank_audit

def input_identity():
    files=[BASE/'protocol.json',BASE/'assets/pathways_54.gmt',Path(__file__),
           BASE/'code/reconstructed_features.py',ROOT/'01_code/11_legacy_review/original_fit.py',
           ROOT/'01_code/10_extension/core.py',ROOT/'01_code/10_extension/data.py']
    files += [BASE/f'data/{name}_{part}.csv'for name in ['mhedili','dilimap']for part in ['metadata','pathways','embeddings']]
    hashes={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()for f in files}
    return hashes,hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def run(output,training_counts=(60,),seeds=tuple(range(10))):
    torch.set_num_threads(4);output.mkdir(parents=True,exist_ok=True);domains,overlap,held=load();s=domains['source'];labs=s[~s.control].groupby('ck').labels.first().drop(held)
    designs=['historical_centred','matched_centred','revised_origin'];origin={};reference={}
    for design in designs:
        origin[design]={};reference[design]={}
        for name,f in domains.items():origin[design][name],reference[design][name]=origins(f,matched=design!='historical_centred')
    masks={name:(~f.control.values)&(~f.ck.isin(overlap).values if name!='source'else f.ck.isin(held).values)for name,f in domains.items()}
    # Exactly the same evaluable profile set for all designs.
    for name in masks:
        for design in designs:masks[name]&=np.isfinite(origin[design][name]).all(1)
    config={'protocol':json.loads((BASE/'protocol.json').read_text()),'held_compounds':held.tolist(),'references':reference,'source_overlap_excluded':{},'evaluation':{}}
    for name,f in domains.items():
        ff=f.loc[masks[name]];config['evaluation'][name]={'profiles':len(ff),'compounds':int(ff.ck.nunique()),'positive_compounds':int(ff.groupby('ck').labels.first().sum()),'sample_ids':ff.sample_id.tolist()};config['source_overlap_excluded'][name]=sorted(set(f.loc[~f.control,'ck'])&overlap)if name!='source'else[]
    ph=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest();config['protocol_sha256']=ph
    definitions=[a for a in arms() if a['name'] in ['softplus_original','kernel_original']];specs=[{**a,'name':design+'__'+a['name'],'design':design}for design in designs for a in definitions];names=[a['name']for a in specs]
    hashes,identity=input_identity();config.update({'input_hashes':hashes,'run_identity':identity,'arms':specs,'optimizer_seed':0,'torch_version':torch.__version__})
    if (output/'config.json').exists():
        old=json.loads((output/'config.json').read_text());assert old['run_identity']==identity,'Inputs or fitting code changed: use a separate output directory'
    (output/'config.json').write_text(json.dumps(config,indent=2))
    all_results=[]
    for n in training_counts:
        for seed in seeds:
            path=output/f'n{n}_seed{seed}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['run_identity']==identity;all_results.append(old);continue
            start=time.time();selected=draw_compounds(labs,n,seed);mask=s.ck.isin(selected).values&~s.control.values
            for design in designs:mask&=np.isfinite(origin[design]['source']).all(1)
            if set(s.loc[mask,'ck'])!=set(selected):raise ValueError('Training compound dropped for missing controls')
            y=s.labels.values[mask];sg=StandardScaler().fit(s.loc[mask,GF]);G=sg.transform(s.loc[mask,GF]);K=rbf(G,G,1/1536)
            assets={d:fit_coordinates(s,origin[d]['source'],selected,d)for d in designs};B={d:{name:transform(f,origin[d][name],assets[d])for name,f in domains.items()}for d in designs}
            scalers={d:StandardScaler().fit(B[d]['source'][mask])for d in designs};P=np.array([scalers[a['design']].transform(B[a['design']]['source'][mask])for a in specs])
            fit=fit_original(K,P,y,specs);np.savez_compressed(output/f'n{n}_seed{seed}_state.npz',a=fit['a'],b=fit['b'],w=fit['w'],raw_parameters=fit['raw_parameters'],embedding_mean=sg.mean_,embedding_scale=sg.scale_,train_ids=s.sample_id[mask].to_numpy(str),arm_names=np.array(names))
            rec={'n_training_compounds':n,'seed':seed,'protocol_sha256':ph,'run_identity':identity,'fitting_device':fit['device'],'train_compounds':selected.tolist(),'n_training_profiles':int(mask.sum()),'coordinate_assets':assets,'coordinate_scalers':{d:{'mean':scalers[d].mean_.tolist(),'scale':scalers[d].scale_.tolist()}for d in designs},'objective':fit['objective'].tolist(),'updates':fit['updates'].tolist(),'best_updates':fit['best_update'].tolist(),'domains':{}}
            for name,f in domains.items():
                ff=f.loc[masks[name]];ke=rbf(sg.transform(ff[GF]),G,1/1536);ker=ke@fit['a'];lin=[]
                for j,a in enumerate(specs):
                    pe=scalers[a['design']].transform(B[a['design']][name][masks[name]])
                    if a['kind']=='kernel':pe=np.zeros_like(pe)
                    lin.append(pe@fit['w'][j])
                lin=np.array(lin).T;full=ker+lin+fit['b'];diagnostic=np.column_stack([scalers[d].transform(B[d][name][masks[name]]).sum(1)for d in designs])
                np.savez_compressed(output/f'n{n}_seed{seed}_{name}_scores.npz',full=full,linear=lin,kernel=ker,diagnostic=diagnostic,diagnostic_names=np.array(designs),arm_names=np.array(names),sample_id=ff.sample_id.to_numpy(str),compound=ff.ck.to_numpy(str),label=ff.labels.values)
                rec['domains'][name]={}
                for j,a in enumerate(specs):
                    c=aggregate(ff.labels,ff.ck,lin[:,j]);k=aggregate(ff.labels,ff.ck,ker[:,j]);rec['domains'][name][a['name']]=rank_audit(c.label.to_numpy(),c.score.to_numpy(),k.score.to_numpy())
            errors=[]
            for j,a in enumerate(specs):
                p=np.zeros_like(P[j])if a['kind']=='kernel'else P[j];errors.append(abs(objective(K,p,y,fit['a'][:,j],fit['w'][j],fit['b'][j],lam=a['lambda'])-fit['objective'][j]))
            assert max(errors)<1e-7;rec['maximum_objective_residual']=max(errors);rec['elapsed_seconds']=time.time()-start;path.write_text(json.dumps(rec,indent=2));all_results.append(rec)
            print(f'Reconstruction n={n}, draw={seed}: {len(specs)} early-stopping approximate solutions, {time.time()-start:.1f}s',flush=True)
    expected=[output/f'n{n}_seed{seed}.json'for n in [60]for seed in range(10)]
    if all(p.exists()for p in expected):
        all_results=[json.loads(p.read_text())for p in expected];assert all(r['run_identity']==identity for r in all_results)
        (output/'all_runs.json').write_text(json.dumps({'config':config,'runs':all_results},indent=2));print('All reconstructed comparisons completed.',flush=True)
    else:print('Requested fits saved; full summary waits for all ten source draws.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'results/experiments');p.add_argument('--training-counts',type=int,nargs='+',choices=[60],default=[60]);p.add_argument('--seeds',type=int,nargs='+',choices=list(range(10)),default=list(range(10)));a=p.parse_args();run(a.output,a.training_counts,a.seeds)
