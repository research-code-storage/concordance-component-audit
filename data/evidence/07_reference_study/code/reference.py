"""Frozen source reference construction; no target-treated or outcome fitting."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'07_reference_study'
sys.path.insert(0,str(ROOT/'05_reconstruction/code'))
import reconstructed_features as old
PC,GF=old.PC,old.GF
PRIMARY='balanced_origin_all'

def load():
    domains,overlap,_=old.load()
    domains={k:f.reset_index(drop=True)for k,f in domains.items()}
    refs={};origins={};masks={}
    for name,f in domains.items():
        origins[name],refs[name]=old.origins(f,True)
        masks[name]=~f.control.to_numpy()&np.isfinite(origins[name]).all(1)
        if name!='source':masks[name]&=~f.ck.isin(overlap).to_numpy()
    return domains,origins,masks,refs

def weights(compounds,kind='balanced',multiplicity=None):
    c=np.asarray(compounds,str);unique,inv,count=np.unique(c,return_inverse=True,return_counts=True)
    w=np.ones(len(c))if kind=='profiles'else 1/count[inv]
    if multiplicity is not None:w*=np.array([multiplicity.get(k,0)for k in unique])[inv]
    if not np.isfinite(w).all()or w.sum()<=0:raise ValueError('Invalid reference weights')
    return w/w.sum()

def moments(x,w,centred=True):
    mean=w@x;var=w@((x-mean)**2);scale=np.sqrt(np.maximum(var,0));scale[scale<1e-12]=1
    return mean if centred else np.zeros(x.shape[1]),scale

def quantile(x,w,p):
    order=np.argsort(x,kind='stable');cum=np.cumsum(w[order]);cum/=cum[-1]
    return float(x[order[min(np.searchsorted(cum,p),len(x)-1)]])

def fit_asset(source,origin,recipe,*,multiplicity=None):
    eligible=~source.control.to_numpy()&np.isfinite(origin).all(1)
    if multiplicity is not None:eligible&=source.ck.map(multiplicity).fillna(0).to_numpy()>0
    ix=np.flatnonzero(eligible);frame=source.iloc[ix];X=frame[PC].to_numpy(float)-origin[ix]
    kind=recipe['weighting'];w=weights(frame.ck,kind,multiplicity);drops=[]
    if recipe['pool']=='thinned':
        _,pre_scale=moments(X,w,False);U=normalize(X/pre_scale);kept=[]
        keys=['ck','source_file','cell','time','dose','baseline_type','media','assay_batch']
        keys=frame[keys].astype(str).agg('|'.join,axis=1).to_numpy()
        for key in sorted(set(keys)):
            positions=np.flatnonzero(keys==key)
            positions=positions[np.argsort(frame.iloc[positions].sample_id.to_numpy(str),kind='stable')]
            sim=U[positions]@U[positions].T;drop=set()
            for i,pos in enumerate(positions):
                if i in drop:continue
                kept.append(pos)
                for j in range(i+1,len(positions)):
                    if j not in drop and sim[i,j]>=.99:
                        drop.add(j);drops.append([str(frame.iloc[pos].sample_id),str(frame.iloc[positions[j]].sample_id),float(sim[i,j])])
        kept=np.array(kept);ix=ix[kept];frame=source.iloc[ix];X=source.iloc[ix][PC].to_numpy(float)-origin[ix];w=weights(frame.ck,kind,multiplicity)
    centre,scale=moments(X,w,recipe['origin']=='centred');Z=(X-centre)/scale;radius=np.linalg.norm(Z,axis=1)
    q=recipe.get('tail_fraction',.15);lo,hi=quantile(radius,w,q),quantile(radius,w,1-q)
    tails=[radius>=hi,radius<=lo];cent=[];mi=[i for i,c in enumerate(PC)if any(k in c for k in old.KEYWORDS)]
    for k,mask in enumerate(tails):
        mask&=radius>1e-12
        if mask.sum()<2:raise ValueError('Insufficient nonzero tail profiles')
        fit=KMeans(n_clusters=2,n_init=10,random_state=recipe.get('seed',42)).fit(normalize(Z[mask]),sample_weight=w[mask])
        C=normalize(fit.cluster_centers_)
        swap=C[0,mi].mean()<C[1,mi].mean()if k==0 else C[0,mi].mean()>C[1,mi].mean()
        cent.extend(C[[1,0]]if swap else C)
    result={'recipe':recipe,'centre':centre.tolist(),'scale':scale.tolist(),'centroids':np.array(cent).tolist(),
        'retained_ids':source.iloc[ix].sample_id.tolist(),'weights':w.tolist(),'q15_or_low':lo,'q85_or_high':hi,
        'low_count':int(tails[1].sum()),'high_count':int(tails[0].sum()),'thinning_pairs':drops,
        'source_compounds':sorted(frame.ck.unique()),'metabolic_indices':mi,'zero_radius_count':int((radius<=1e-12).sum())}
    # The final 2D scale always refers to the full eligible source-treated pool,
    # using the weighting rule of this reference recipe, never target subjects.
    all_ix=np.flatnonzero(eligible);B=project(source.iloc[all_ix],origin[all_ix],result,scaled=False)
    bw=weights(source.iloc[all_ix].ck,kind,multiplicity);bm,bs=moments(B,bw)
    result['coordinate_mean']=bm.tolist();result['coordinate_scale']=bs.tolist()
    return result

def project(frame,origin,asset,scaled=True):
    z=(frame[PC].to_numpy(float)-origin-asset['centre'])/asset['scale'];r=np.linalg.norm(z,axis=1)
    u=np.divide(z,r[:,None],out=np.zeros_like(z),where=r[:,None]!=0);M=np.asarray(asset['centroids'])
    b=np.column_stack([r,u@(M[1]-M[3])])
    return (b-asset['coordinate_mean'])/asset['coordinate_scale']if scaled else b

def recipes():
    return {PRIMARY:{'name':PRIMARY,'weighting':'balanced','origin':'origin','pool':'all','tail_fraction':.15,'seed':42}}

def build():
    domains,O,masks,refs=load();s=domains['source'];assets={}
    for name,r in recipes().items():assets[name]=fit_asset(s,O['source'],r)
    w=weights(s.loc[masks['source'],'ck']);G=s.loc[masks['source'],GF].to_numpy(float);gm,gs=moments(G,w)
    config={'protocol_sha256':hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),'recipes':assets,
            'embedding_mean':gm.tolist(),'embedding_scale':gs.tolist(),'references':refs,'panels':{}}
    for name,f in domains.items():
        ff=f.loc[masks[name]];config['panels'][name]={'profiles':len(ff),'compounds':int(ff.ck.nunique()),
            'positive_compounds':int(ff.groupby('ck').labels.first().sum()),'sample_ids':ff.sample_id.tolist()}
        b=project(ff,O[name][masks[name]],assets[PRIMARY],False)
        table=ff[['sample_id','chemical','ck','labels','cell']+GF].copy();table['S_Intensity']=b[:,0];table['S_tox_raw']=b[:,1]
        table.to_csv(BASE/f'data/{name}_770_features.csv',index=False)
    (BASE/'assets/frozen_reference.json').write_text(json.dumps(config,indent=2))
    print(json.dumps({'panels':{k:{a:b for a,b in v.items()if a!='sample_ids'}for k,v in config['panels'].items()},
        'reference_pools':{k:len(v['retained_ids'])for k,v in assets.items()}}),flush=True)
    return config

if __name__=='__main__':build()
