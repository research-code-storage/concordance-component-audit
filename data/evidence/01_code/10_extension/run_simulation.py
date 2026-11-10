"""Fitted-model simulation: 200 training draws, one fixed evaluation sample, three displacements and two coordinate-weight initializations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,time
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from data import ROOT
from core import rbf,fit_batch,predict_batch,rank_audit

def draw(rng,n,v,disp=0):
    y=rng.binomial(1,.5,n);u=np.abs(rng.randn(n))*(2*y-1)
    z=.9*rng.randn(n,len(v))+np.outer(u,v)+disp
    info=np.stack([np.abs(u)+.6*rng.randn(n),u+.6*rng.randn(n)],1)
    noise=rng.randn(n,2)
    return z,info,noise,y

def run(R=200):
    torch.set_num_threads(2);out=ROOT/'04_results/extension_double';rng=np.random.RandomState(7777)
    v=rng.randn(30);v/=np.linalg.norm(v);w=rng.randn(30);w/=np.linalg.norm(w)
    ze,ie,ne,ye=draw(rng,300,v);shifts=[0,6,16];records=[]
    design={'R':R,'D':30,'n_train':300,'n_test':300,'shifts':shifts,'gamma':1/60,'C':20,'lambda':.001,
        'training_seeds':list(range(1000,1000+R)),'test_seed':7777,'common_training_draws_across_cells':True,
        'evaluation':'fixed observations; only covariate displacement changes',
        'initial_values':[0,1.3132616877555847],'optimizer':'documented independent batched Adam updates',
        'device':'cpu','dtype':'float64','steps':1000,'patience':80,'lr':.001,'eta_min':1e-5,'minimum_improvement':.0001,'clip_norm':1.}
    (out/'simulation_design.json').write_text(json.dumps(design,indent=2))
    for r in range(R):
        z,i,n,y=draw(np.random.RandomState(1000+r),300,v);sg=StandardScaler().fit(z);si=StandardScaler().fit(i);sn=StandardScaler().fit(n)
        G=sg.transform(z);P1=si.transform(i);P0=sn.transform(n);K=rbf(G,G,1/60)
        Ps=np.array([P1,P1,P0,P0]);initial=np.array([[1.3132616877555847]*2,[0,0]]*2)
        fit=fit_batch(K,Ps,y,[1]*4,initial,device='cpu')
        scores=[]
        for sh in shifts:
            Kt=rbf(sg.transform(ze+sh*w),G,1/60);Pt=np.array([si.transform(ie)]*2+[sn.transform(ne)]*2)
            pred=predict_batch(fit,Kt,Pt);scores.append(pred)
            for j in range(4):
                ra=rank_audit(ye,pred['linear'][:,j],pred['kernel'][:,j])
                records.append({'replicate':r,'shift':sh,'informative':j<2,'initial':float(initial[j,0]),
                    'auc':roc_auc_score(ye,pred['full'][:,j]),'auc_sd_component':ra['kernel_sd'],
                    'kernel_similarity':float(Kt.mean()),'share':ra['sd_share'],
                    'auc_difference':ra['auc_difference'],'changed_fraction':ra['changed_fraction'],
                    'bound':ra['absolute_auc_bound'],'objective':float(fit['objective'][j])})
        np.savez_compressed(out/f'simulation_rep{r:03}.npz',alpha=fit['a'],weights=fit['w'],bias=fit['b'],
            full=np.stack([p['full']for p in scores]),kernel=np.stack([p['kernel']for p in scores]),linear=np.stack([p['linear']for p in scores]),test_labels=ye,train_labels=y)
        if (r+1)%10==0:
            pd.DataFrame(records).to_csv(out/'simulation_replicates.csv',index=False)
            print(f'simulation {r+1}/{R}',flush=True)
    df=pd.DataFrame(records);df.to_csv(out/'simulation_replicates.csv',index=False);summaries=[]
    for (sh,inf,init),g in df.groupby(['shift','informative','initial'],sort=True):
        summary={'shift':int(sh),'informative':bool(inf),'initial':float(init),'R':len(g)}
        for col in ['auc','kernel_similarity','share','auc_difference','changed_fraction','bound']:
            summary[col+'_mean']=float(g[col].mean());summary[col+'_sd']=float(g[col].std(ddof=1));summary[col+'_mcse']=float(g[col].std(ddof=1)/np.sqrt(R))
        summaries.append(summary)
    (out/'simulation_summary.json').write_text(json.dumps({'design':design,'cells':summaries},indent=2))
    print('simulation complete',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=200);run(ap.parse_args().replicates)
