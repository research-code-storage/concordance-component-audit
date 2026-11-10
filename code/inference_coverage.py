"""Evaluate coverage and standard-error calibration for fixed-predictor inference.

The design is specified in inference_design.json. Binary reference values use
numerical integration. Survival reference values use replicated scrambled Sobol
integration with event and censoring times integrated analytically.
"""
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_v]='1'
from pathlib import Path
import argparse,json,concurrent.futures,sys,time,hashlib
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
KEYS=('Delta','A','B'); Z975=1.959963984540054
MASTER=20261009; BOOT_MASTER=20261010

def H(x): return (x>0).astype(float)+.5*(x==0)
def pair_kernels(D,E):
    delta=H(D+E)-H(D)
    rho=np.where(D==0,.5*(E!=0),np.where(np.abs(E)>np.abs(D),1.,.5*(np.abs(E)==np.abs(D))))
    return np.stack((delta,np.abs(delta),rho))

def binary_matrices(rng,n,a,geometry,distribution):
    m=n//2;y=np.r_[np.ones(m),np.zeros(m)]
    err=rng.standard_t(3,n)/np.sqrt(3) if distribution=='t3' else rng.normal(size=n)
    ell=.6*y+err
    if distribution=='rounded_normal':ell=np.rint(ell)
    g=a*(rng.normal(size=n)+.6*geometry*y)
    return pair_kernels(ell[:m,None]-ell[None,m:],g[:m,None]-g[None,m:])

def binary_estimate(K):
    m,n=K.shape[1:];T=K.mean(axis=(1,2));v=K.mean(2);w=K.mean(1)
    # Proposition 4 applied to the ratio U-statistic. In a fixed-case/control
    # design, the identical asymptotic projection is conditioned on class counts.
    V=(m+n)/(m+n-1)*(np.sum((v-T[:,None])**2,axis=1)/m**2+
                     np.sum((w-T[:,None])**2,axis=1)/n**2)
    return T,np.sqrt(V)

def survival_matrices(rng,n,a,censor_rate):
    x,z=rng.normal(size=(2,n));rate=np.exp(x+.6*z)
    event=rng.exponential(1/rate);censor=rng.exponential(1/censor_rate,n)
    t=np.minimum(event,censor);status=event<=censor
    c=status[:,None] & (t[:,None]<t[None,:])
    K=pair_kernels(x[:,None]-x[None,:],a*(z[:,None]-z[None,:]))
    return c.astype(float),K,float(1-status.mean())

def survival_estimate(c,K):
    n=len(c);M=c.sum()
    if M==0:return np.full(3,np.nan),np.full(3,np.nan)
    ck=K*c;T=ck.sum(axis=(1,2))/M
    S=ck.sum(1)+ck.sum(2);S0=c.sum(0)+c.sum(1)
    V=n/(n-1)/M**2*np.sum((S-T[:,None]*S0)**2,axis=1)
    return T,np.sqrt(V)

def bootstrap_binary(K,rng,B):
    m,n=K.shape[1:];p=rng.multinomial(m,np.full(m,1/m),B).astype(float)
    q=rng.multinomial(n,np.full(n,1/n),B).astype(float)
    vals=np.stack([np.einsum('bi,bi->b',p@k,q)/(m*n) for k in K],axis=1)
    return np.quantile(vals,[.025,.975],axis=0),B

def bootstrap_survival(c,K,rng,B):
    n=len(c);w=rng.multinomial(n,np.full(n,1/n),B).astype(float)
    den=np.einsum('bi,bi->b',w@c,w);ok=den>0
    vals=np.stack([np.einsum('bi,bi->b',w@(c*k),w)[ok]/den[ok] for k in K],axis=1)
    return (np.quantile(vals,[.025,.975],axis=0) if ok.any() else np.full((2,3),np.nan)),int(ok.sum())

def binary_truth(a,s,dist):
    mu=.6*a*s;sd=np.sqrt(2)*a
    def v(d):
        delta=ndtr((d+mu)/sd)-(1. if d>0 else .5 if d==0 else 0.)
        A=.5 if d==0 else ndtr((-abs(d)-np.sign(d)*mu)/sd)
        B=.5 if d==0 else ndtr((-abs(d)-mu)/sd)+ndtr((mu-abs(d))/sd)
        return np.array([delta,A,B])
    if dist=='rounded_normal':
        j=np.arange(-12,13);pp=ndtr(j+.5-.6)-ndtr(j-.5-.6);pn=ndtr(j+.5)-ndtr(j-.5)
        out=sum(pp[i]*pn[k]*v(float(x-y)) for i,x in enumerate(j) for k,y in enumerate(j))
        return {'value':out.tolist(),'integration_error_bound':float(1-pp.sum()*pn.sum()),'method':'finite sum of rounded-normal cell probabilities; omitted mass < 1e-28'}
    if dist=='normal':
        density=lambda d: np.exp(-.25*(d-.6)**2)/(2*np.sqrt(np.pi))
    else:
        # The standardized t3 density is 2/[pi(1+x^2)^2]. Its difference
        # has characteristic function (1+|u|)^2 exp(-2|u|).
        density=lambda d:4*(20+(d-.6)**2)/(np.pi*(4+(d-.6)**2)**3)
    vals=[];errs=[]
    for k in range(3):
        pieces=[quad(lambda d:v(d)[k]*density(d),lo,hi,epsabs=2e-12,epsrel=2e-10,limit=250)
                for lo,hi in [(-np.inf,-1),(-1,0),(0,1),(1,np.inf)]]
        vals.append(sum(p[0] for p in pieces));errs.append(sum(p[1] for p in pieces))
    if dist=='normal':
        analytic=ndtr((.6+mu)/np.sqrt(2+sd**2))-ndtr(.6/np.sqrt(2))
        assert abs(vals[0]-analytic)<2e-10
    return {'value':vals,'integration_error_bound':max(errs),'method':'adaptive quadrature of independent pair differences'}

def censor_rate(target):
    def frac(logc):
        c=np.exp(logc)
        return quad(lambda z: c/(np.exp(np.sqrt(1.36)*z)+c)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-11)[0]
    r=brentq(lambda z:frac(z)-target,-8,8,xtol=1e-12)
    return float(np.exp(r))

def survival_truth(a,cr,blocks,power):
    estimates=[]
    for b in range(blocks):
        sob=qmc.Sobol(4,scramble=True,seed=20261011+b)
        total=np.zeros(4)
        for _ in range(2**max(0,power-17)):
            u=sob.random_base2(power if power<=17 else 17) if _==0 else sob.random(2**17)
            x1,z1,x2,z2=ndtri(np.clip(u,1e-15,1-1e-15)).T
            r1=np.exp(x1+.6*z1);r2=np.exp(x2+.6*z2)
            weight=r1/(r1+r2+2*cr)
            k=pair_kernels(x1-x2,a*(z1-z2))
            total[:3]+=(k*weight).sum(1);total[3]+=weight.sum()
        estimates.append(total[:3]/total[3])
    e=np.array(estimates)
    return {'value':e.mean(0).tolist(),'reference_se':(e.std(0,ddof=1)/np.sqrt(blocks)).tolist(),
            'block_values':e.tolist(),'blocks':blocks,'pairs_per_block':2**power,
            'method':'scrambled Sobol quadrature; event and independent censoring times analytically integrated'}

def scenarios():
    out=[]
    for n in (50,200):
        for a in (.01,.1,.5,1.):
            for s,geo in [(1,'aligned'),(0,'independent'),(-1,'opposed')]:
                for d in ['normal','t3','rounded_normal']:
                    out.append({'kind':'binary','n':n,'a':a,'s':s,'geometry':geo,'distribution':d,
                                'bootstrap':s==0 and a==.01 and d in ['normal','rounded_normal']})
    for n in (50,200):
        for a in (.1,.5):
            for c in (.3,.6):
                out.append({'kind':'survival','n':n,'a':a,'censor_fraction':c,'censor_rate':censor_rate(c),'bootstrap':c==.6})
    for i,o in enumerate(out):o['id']=i
    return out

def one_case(job):
    cfg,truth,R,BR,B,out=job;sid=cfg['id'];path=Path(out)/f'case_{sid:02}.npz'
    est=np.empty((R,3));se=np.empty_like(est);lo=np.full_like(est,np.nan);hi=lo.copy()
    bv=np.zeros(R,int);censor=np.full(R,np.nan)
    for r in range(R):
        rng=np.random.default_rng(np.random.SeedSequence([MASTER,sid,r]))
        if cfg['kind']=='binary':
            K=binary_matrices(rng,cfg['n'],cfg['a'],cfg['s'],cfg['distribution']);est[r],se[r]=binary_estimate(K)
        else:
            c,K,censor[r]=survival_matrices(rng,cfg['n'],cfg['a'],cfg['censor_rate']);est[r],se[r]=survival_estimate(c,K)
        assert abs(est[r,0])<=est[r,1]+1e-12 and est[r,1]<=est[r,2]+1e-12
        if cfg['bootstrap'] and r<BR:
            rngb=np.random.default_rng(np.random.SeedSequence([BOOT_MASTER,sid,r]))
            ci,valid=bootstrap_binary(K,rngb,B) if cfg['kind']=='binary' else bootstrap_survival(c,K,rngb,B)
            lo[r],hi[r]=ci;bv[r]=valid
    np.savez_compressed(path,estimate=est,se=se,bootstrap_lower=lo,bootstrap_upper=hi,
                        bootstrap_valid=bv,censor_fraction=censor,truth=np.asarray(truth['value']))
    return {'id':sid,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def summarize(cfg,truth,path):
    with np.load(path) as z:est=z['estimate'];se=z['se'];lo=z['bootstrap_lower'];hi=z['bootstrap_upper'];bv=z['bootstrap_valid'];cf=z['censor_fraction']
    out={'scenario':cfg,'truth':truth,'replicates':len(est),'metrics':{}}
    tr=np.array(truth['value'])
    for j,key in enumerate(KEYS):
        valid=np.isfinite(se[:,j])&(se[:,j]>0);cov=valid&(est[:,j]-Z975*se[:,j]<=tr[j])&(est[:,j]+Z975*se[:,j]>=tr[j])
        cp=float(cov.mean());sd=float(np.std(est[:,j],ddof=1))
        q={'estimate_mean':float(np.mean(est[:,j])),'empirical_sd':sd,'mean_se':float(np.mean(se[:,j])),
           'mean_se_over_sd':float(np.mean(se[:,j])/sd),'bias':float(np.mean(est[:,j])-tr[j]),
           'normal_coverage':cp,'coverage_mcse':float(np.sqrt(cp*(1-cp)/len(est))),
           'zero_se_count':int((se[:,j]==0).sum()),'unavailable_count':int((~valid).sum()),
           'coverage_given_positive_se':float(cov.sum()/valid.sum()) if valid.any() else None}
        use=bv>0
        if use.any():
            bcover=(lo[use,j]<=tr[j])&(hi[use,j]>=tr[j]);p=float(bcover.mean())
            q['bootstrap']={'replicates':int(use.sum()),'resamples_requested':int(bv.max()),
               'minimum_valid_resamples':int(bv[use].min()),'percentile_coverage':p,
               'coverage_mcse':float(np.sqrt(p*(1-p)/use.sum())),
               'normal_coverage_same_replicates':float(cov[use].mean()),
               'zero_width_count':int((lo[use,j]==hi[use,j]).sum()),
               'normal_unavailable_count':int((~valid[use]).sum())}
        out['metrics'][key]=q
    if cfg['kind']=='survival':out['realized_censor_fraction']=float(np.mean(cf))
    return out

def validate():
    # Independent scalar enumeration and explicit subject-resample indices.
    from lung_ustat import closed_form,projection
    rng=np.random.default_rng(991)
    for n in (8,14):
        K=binary_matrices(rng,n,.5,0,'rounded_normal');T,se=binary_estimate(K)
        m=n//2;c=np.zeros((n,n));c[:m,m:]=1
        for j in range(3):
            kk=np.zeros_like(c);kk[:m,m:]=K[j]
            q,v=closed_form(c,kk);p,pv=projection(c,kk)
            assert np.allclose([q,v],[T[j],se[j]**2],rtol=1e-12,atol=1e-14)
            assert np.allclose([q,v],[p,pv],rtol=1e-12,atol=1e-14)
        c,k,_=survival_matrices(rng,n,.5,1.)
        t,s=survival_estimate(c,k)
        for j in range(3):
            q,v=closed_form(c,k[j]);assert np.allclose([q,v],[t[j],s[j]**2],rtol=1e-12)
            idx=rng.integers(0,n,n);w=np.bincount(idx,minlength=n)
            direct=(c*k[j])[np.ix_(idx,idx)].sum()/c[np.ix_(idx,idx)].sum()
            weighted=(w@(c*k[j])@w)/(w@c@w)
            assert np.allclose(direct,weighted,atol=1e-14)
    return 'PASS: ratio projection, fixed-class reduction and bootstrap count algebra'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'data/evidence/08_inference')
    ap.add_argument('--replicates',type=int,default=2000);ap.add_argument('--bootstrap-replicates',type=int,default=1000)
    ap.add_argument('--bootstrap-resamples',type=int,default=999);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--truth-power',type=int,default=20);ap.add_argument('--truth-blocks',type=int,default=16)
    ap.add_argument('--verify',action='store_true');args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.verify:
        design=json.loads((out/'inference_design.json').read_text());truths=json.loads((out/'reference_values.json').read_text())
        cells=[summarize(c,truths[str(c['id'])],out/f"case_{c['id']:02}.npz") for c in design['scenarios']]
        summary=json.loads((out/'summary.json').read_text());assert cells==summary['cells'];print(validate());print('All 80 coverage cells and saved replicate summaries verified');return
    configs=scenarios();design={'replicates':args.replicates,'bootstrap_evaluation_replicates':args.bootstrap_replicates,
        'bootstrap_resamples':args.bootstrap_resamples,'master_seed':MASTER,'bootstrap_master_seed':BOOT_MASTER,
        'bootstrap_selection':'first 1000 replicates of each of eight design-defined scenarios, selected before outcomes',
        'zero_se_policy':'normal interval unavailable; counts as noncoverage in unconditional coverage; also report conditional coverage',
        'binary_sampling':'equal fixed case/control counts; bootstrap separately within classes',
        'survival_sampling':'iid subjects; ell=X, g=a Z; X,Z iid standard normal; event rate exp(X+0.6 Z); independent exponential censoring; observed-pair estimand',
        'survival_bootstrap':'resample subjects; reconstruct observed comparable pairs; no refitting or IPCW estimation',
        'truth_sobol_blocks':args.truth_blocks,'truth_sobol_power':args.truth_power,'scenarios':configs}
    (out/'inference_design.json').write_text(json.dumps(design,indent=2),encoding='utf8')
    print(validate(),flush=True);truths={};cache={}
    for cfg in configs:
        key=(cfg['kind'],cfg['a'],cfg.get('s'),cfg.get('distribution'),cfg.get('censor_rate'))
        if key not in cache:
            cache[key]=binary_truth(cfg['a'],cfg['s'],cfg['distribution']) if cfg['kind']=='binary' else survival_truth(cfg['a'],cfg['censor_rate'],args.truth_blocks,args.truth_power)
            print('reference',key,cache[key]['value'],flush=True)
        truths[str(cfg['id'])]=cache[key]
    (out/'reference_values.json').write_text(json.dumps(truths,indent=2),encoding='utf8')
    print('Reference values complete; starting 80 coverage cells',flush=True)
    jobs=[(c,truths[str(c['id'])],args.replicates,args.bootstrap_replicates,args.bootstrap_resamples,out) for c in configs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(one_case,j):j[0]['id'] for j in jobs}
        for f in concurrent.futures.as_completed(pending):r=f.result();print('complete',r['id'],flush=True)
    cells=[summarize(c,truths[str(c['id'])],out/f"case_{c['id']:02}.npz") for c in configs]
    summary={'design_file':'inference_design.json','validation':validate(),'cells':cells}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Coverage experiment complete',flush=True)

if __name__=='__main__':main()
