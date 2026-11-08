"""Frozen-reference labelling study (SI H.3), recomputed from saved profile scores."""
from pathlib import Path
import argparse,json
import numpy as np
MAIN='balanced_origin_all__softplus_original'
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'generated');a=p.parse_args()
    folder=a.study/'07_reference_study/results/fits';cells={};component=[]
    for n in [4,20,60,102]:
        for dom in ['external','PHH']:
            draws=[];identity=None
            for seed in (range(10) if n<102 else [0]):
                z=np.load(folder/f'n{n}_seed{seed}_{dom}.npz',allow_pickle=False);names=z['arm_names'].tolist()
                assert names==[MAIN] and len(np.unique(z['sample_id']))==len(z['sample_id'])
                ids,inv,count=np.unique(z['compound'],return_inverse=True,return_counts=True)
                labels=np.array([z['label'][inv==i][0] for i in range(len(ids))])
                assert all(np.all(z['label'][inv==i]==labels[i]) for i in range(len(ids))),'Conflicting compound labels'
                key=(ids.tolist(),labels.tolist())
                if identity is not None:assert key==identity,'Panel differs across draws'
                identity=key;x=z['full'];assert np.isfinite(x).all()
                v=np.zeros((len(ids),x.shape[1]));np.add.at(v,inv,x);v/=count[:,None]
                d=v[labels==1][:,None,:]-v[labels==0][None,:,:];draws.append(((d>0)+.5*(d==0)).mean((0,1)))
                if n==60:
                    parts=[]
                    for name in ('linear','kernel'):
                        w=np.zeros((len(ids),1));np.add.at(w,inv,z[name]);w/=count[:,None];parts.append(w[:,0])
                    ell,g=parts;D=ell[labels==1,None]-ell[None,labels==0];E=g[labels==1,None]-g[None,labels==0]
                    np.testing.assert_allclose(d[:,:,0],D+E,rtol=1e-10,atol=1e-10)
                    assert np.all(d[:,:,0]!=0) and np.all(D!=0), 'Changed-pair counts require an explicit tie treatment'
                    hf=(d[:,:,0]>0)+.5*(d[:,:,0]==0);hl=(D>0)+.5*(D==0)
                    rho=np.where(D==0,.5*(E!=0),np.where(abs(E)>abs(D),1.,.5*(abs(E)==abs(D))))
                    assert abs((hf-hl).mean())<=abs(hf-hl).mean()+1e-12<=rho.mean()+2e-12
                    component.append({'domain':dom,'seed':seed,'Delta':float((hf-hl).mean()),'A':float(abs(hf-hl).mean()),'B':float(rho.mean())})
            m=np.mean(draws,axis=0);cells[n,dom,MAIN]=float(m[0])
    out={dom:{str(n):cells[n,dom,MAIN] for n in [4,20,60,102]} for dom in ['external','PHH']}
    out['spread_over_training_sizes']={k:max(out[k].values())-min(out[k].values()) for k in ['external','PHH']}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'frozen_reference.json').write_text(json.dumps(out,indent=2))
    (a.out/'frozen_component_audit.json').write_text(json.dumps({'cells':component},indent=2))
    print('Frozen-reference compound AUCs from 31 saved draws per panel')
    for n in ['4','20','60','102']:print(n, f'DILImap {out["external"][n]:.4f}; PHH {out["PHH"][n]:.4f}')
if __name__=='__main__':main()
