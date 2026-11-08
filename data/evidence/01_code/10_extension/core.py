"""Kernel operations, projected nonnegative model fitting and pairwise concordance calculations."""
import math
import numpy as np
from scipy.spatial.distance import cdist

def rbf(x,y,gamma):
    return np.exp(-gamma*cdist(np.asarray(x,float),np.asarray(y,float),'sqeuclidean'))

def class_caps(y,C=20.):
    y=np.asarray(y)
    if set(np.unique(y))!={0,1}:raise ValueError('Both binary classes required')
    return np.where(y==1,C/(2*np.sum(y==1)),C/(2*np.sum(y==0)))

def objective(K,P,y,a,w,b,C=20.,lam=.001):
    f=K@a+P@w+b
    return float(.5*a@K@a+.5*lam*(w@w)+class_caps(y,C)@np.maximum(0,1-(2*y-1)*f))



def rank_audit(y,linear,kernel):
    """Finite-sample AUC attribution at the supplied observation unit, including ties.

    A pair can change its ordering only if |linear difference| <= |kernel
    difference|. The resulting bound is deterministic, not a confidence interval.
    """
    y=np.asarray(y,int);linear=np.asarray(linear,float);kernel=np.asarray(kernel,float)
    dl=linear[y==1,None]-linear[y==0];dk=kernel[y==1,None]-kernel[y==0]
    h=lambda d:(d>0).astype(float)+.5*(d==0)
    changed=np.sign(dl+dk)!=np.sign(dl)
    vulnerable=np.abs(dl)<=np.abs(dk)
    delta=float((h(dl+dk)-h(dl)).mean())
    bound=float(vulnerable.mean())
    if abs(delta)>bound+1e-12:raise AssertionError('Pairwise AUC bound violated')
    return {'auc_linear':float(h(dl).mean()),'auc_full':float(h(dl+dk).mean()),
            'auc_difference':delta,'pairs':int(dl.size),'changed_pairs':int(changed.sum()),
            'changed_fraction':float(changed.mean()),'vulnerable_pairs':int(vulnerable.sum()),
            'absolute_auc_bound':bound,'kernel_sd':float(np.std(kernel)),
            'linear_sd':float(np.std(linear)),
            'sd_share':float(np.std(kernel)/(np.std(kernel)+np.std(linear)))if np.std(kernel)+np.std(linear)>0 else 0.}

def fit_batch(K,Ps,y,cones,w_initial,*,steps=1000,patience=80,lr=.001,
              eta_min=1e-5,C=20.,lam=.001,min_improvement=1e-4,clip=1.,seed=0,device=None,dtype='float64'):
    """Independent Adam fits sharing K; post-update objective and snapshot agree.

    Ps shape (models,n,coordinates); no cross-model gradient averaging/clipping.
    A padded zero coordinate has zero initial weight and stays zero. Independent
    best snapshots and early-stopping counters are retained for every model.
    AdamW with weight_decay=0 equals Adam here; no unstated regularisation.
    """
    import torch
    dev=device or ('cuda'if torch.cuda.is_available()else'cpu')
    tensor_dtype=getattr(torch,dtype)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(seed)
    Kt=torch.as_tensor(K,dtype=tensor_dtype,device=dev)
    P=torch.as_tensor(Ps,dtype=tensor_dtype,device=dev)
    m,n,d=P.shape
    if Kt.shape!=(n,n):raise ValueError('Kernel/feature shape mismatch')
    cones=torch.as_tensor(cones,device=dev)
    if not bool(torch.all((cones==0)|(cones==1)|(cones==-1))):raise ValueError('Invalid cone')
    a=(torch.randn(n,1,device=dev)*.01).to(tensor_dtype).repeat(1,m)
    b=torch.zeros(m,device=dev,dtype=tensor_dtype);w=torch.as_tensor(w_initial,dtype=tensor_dtype,device=dev).clone()
    if w.shape!=(m,d):raise ValueError('Initial weights shape mismatch')
    w=torch.where((cones==1)[:,None],torch.clamp(w,min=0),torch.where((cones==-1)[:,None],torch.clamp(w,max=0),w))
    labels=np.asarray(y)
    if labels.ndim==1:
        ys=(2*labels-1)[:,None];cs=class_caps(labels,C)[:,None]
    elif labels.shape==(m,n):
        ys=(2*labels-1).T;cs=np.stack([class_caps(row,C)for row in labels]).T
    else:raise ValueError('Labels must be length n or a models-by-n array')
    ysign=torch.as_tensor(ys,dtype=tensor_dtype,device=dev)
    caps=torch.as_tensor(cs,dtype=tensor_dtype,device=dev)
    def evaluate(a,b,w):
        ka=Kt@a;pr=ka+torch.einsum('mnd,md->nm',P,w)+b
        values=.5*(a*ka).sum(0)+.5*lam*(w*w).sum(1)+(caps*torch.clamp(1-ysign*pr,min=0)).sum(0)
        return values,pr,ka
    with torch.no_grad():
        loss,pr,ka=evaluate(a,b,w);best=loss.clone();ba=a.clone();bb=b.clone();bw=w.clone()
        bad=torch.zeros(m,dtype=torch.long,device=dev);active=torch.ones(m,dtype=torch.bool,device=dev)
        stopped=torch.full((m,),steps,dtype=torch.long,device=dev);best_step=torch.zeros(m,dtype=torch.long,device=dev)
        ma=torch.zeros_like(a);va=torch.zeros_like(a);mb=torch.zeros_like(b);vb=torch.zeros_like(b);mw=torch.zeros_like(w);vw=torch.zeros_like(w)
        trace=[]
        for step in range(steps):
            signed=caps*ysign*(ysign*pr<=1)
            ga=ka-Kt@signed;gb=-signed.sum(0);gw=lam*w-torch.einsum('mnd,nm->md',P,signed)
            norm=torch.sqrt((ga*ga).sum(0)+gb*gb+(gw*gw).sum(1))
            scale=torch.clamp(clip/(norm+1e-6),max=1.)*active
            ga*=scale;gb*=scale;gw*=scale[:,None]
            rate=eta_min+.5*(lr-eta_min)*(1+math.cos(math.pi*step/steps))
            for par,g,mean,var in ((a,ga,ma,va),(b,gb,mb,vb),(w,gw,mw,vw)):
                mean.mul_(.9).add_(g,alpha=.1);var.mul_(.999).addcmul_(g,g,value=.001)
                upd=(mean/(1-.9**(step+1)))/(torch.sqrt(var/(1-.999**(step+1)))+1e-8)
                mask=active[None,:]if par is a else active[:,None]if par is w else active
                par.add_(upd*mask,alpha=-rate)
            w=torch.where((cones==1)[:,None],torch.clamp(w,min=0),torch.where((cones==-1)[:,None],torch.clamp(w,max=0),w))
            loss,pr,ka=evaluate(a,b,w)
            improve=active&(loss<best-min_improvement)
            ba[:,improve]=a[:,improve];bb[improve]=b[improve];bw[improve]=w[improve];best[improve]=loss[improve];best_step[improve]=step+1
            bad=torch.where(improve,0,bad+active)
            done=active&(bad>=patience);stopped[done]=step+1;active&=~done
            if step+1 in (1,5,20,100,300,600,1000):trace.append({'updates':step+1,'objective':loss.cpu().numpy().tolist(),'w':w.cpu().numpy().tolist()})
            if not bool(active.any()):break
        final,_,_=evaluate(ba,bb,bw)
        if not torch.allclose(final,best,rtol=2e-5,atol=2e-5):raise AssertionError('Snapshot/objective mismatch')
    return {'a':ba.cpu().numpy(),'b':bb.cpu().numpy(),'w':bw.cpu().numpy(),
            'objective':final.cpu().numpy(),'updates':stopped.cpu().numpy(),
            'best_update':best_step.cpu().numpy(),'trace':trace,'device':str(dev)}

def predict_batch(fit,Kte,Pte):
    kernel=np.asarray(Kte)@fit['a'];linear=np.einsum('mnd,md->nm',Pte,fit['w'])
    return {'full':kernel+linear+fit['b'],'kernel':kernel,'linear':linear}

