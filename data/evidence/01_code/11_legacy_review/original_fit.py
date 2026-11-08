"""Softplus and validation model fitting with independent AdamW updates, post-update checkpoints and per-draw scaling."""
import math
import numpy as np,torch

def arms():
    return [{'name': 'projected_wd0', 'kind': 'hard', 'lambda': 0.001, 'wd': 0.0, 'init': 'positive'}, {'name': 'softplus_original', 'kind': 'softplus', 'lambda': 0.001, 'wd': 0.0001, 'init': 'positive'}, {'name': 'softplus_reverse', 'kind': 'reverse', 'lambda': 0.001, 'wd': 0.0001, 'init': 'positive'}, {'name': 'free_matched_positive_start', 'kind': 'free', 'lambda': 0.001, 'wd': 0.0001, 'init': 'positive'}, {'name': 'kernel_original', 'kind': 'kernel', 'lambda': 0.001, 'wd': 0.0001, 'init': 'zero'}]


def fit_original(K,P,y,specs=None,*,steps=1000,device=None,seed=0):
    specs=specs or arms();dev=device or ('cuda'if torch.cuda.is_available()else'cpu');dtype=torch.float64
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(seed)
    n=len(y);m=len(specs);Kt=torch.as_tensor(K,dtype=dtype,device=dev)
    supplied=np.asarray(P)
    matrices=[supplied[j]if supplied.ndim==3 else supplied for j in range(m)]
    Ps=np.stack([np.zeros_like(p)if a['kind']=='kernel'else p for a,p in zip(specs,matrices)]);Pt=torch.as_tensor(Ps,dtype=dtype,device=dev)
    a=(torch.randn(n,1,device=dev)*.01).to(dtype).repeat(1,m);random_w=(torch.randn(2,device=dev)*.05).to(dtype)
    v=torch.stack([torch.ones(2,dtype=dtype,device=dev)if s['kind']in ['softplus','reverse']else
       random_w.clone()if s['init']=='random'else torch.zeros(2,dtype=dtype,device=dev)if s['init']=='zero'else
       torch.full((2,),1.3132616877555847,dtype=dtype,device=dev)for s in specs])
    b=torch.zeros(m,dtype=dtype,device=dev)
    soft=torch.tensor([s['kind']in ['softplus','reverse']for s in specs],device=dev);hard=torch.tensor([s['kind']=='hard'for s in specs],device=dev)
    signs=torch.tensor([-1. if s['kind']=='reverse'else 1. for s in specs],dtype=dtype,device=dev)
    lam=torch.tensor([s['lambda']for s in specs],dtype=dtype,device=dev);wd=torch.tensor([s['wd']for s in specs],dtype=dtype,device=dev)
    ys=torch.as_tensor((2*np.asarray(y)-1)[:,None],dtype=dtype,device=dev);cap=torch.as_tensor(np.where(y==1,20/(2*(y==1).sum()),20/(2*(y==0).sum()))[:,None],dtype=dtype,device=dev)
    def weights(v):return torch.where(soft[:,None],signs[:,None]*torch.nn.functional.softplus(v),v)
    def evaluate(a,b,v):
        w=weights(v);ka=Kt@a;pred=ka+torch.einsum('mnd,md->nm',Pt,w)+b
        objective=.5*(a*ka).sum(0)+.5*lam*(w*w).sum(1)+(cap*torch.clamp(1-ys*pred,min=0)).sum(0)
        return objective,pred,ka,w
    with torch.no_grad():
        obj,pred,ka,w=evaluate(a,b,v);best=obj.clone();ba=a.clone();bb=b.clone();bv=v.clone()
        active=torch.ones(m,dtype=torch.bool,device=dev);bad=torch.zeros(m,dtype=torch.long,device=dev);stop=torch.full((m,),steps,dtype=torch.long,device=dev);beststep=torch.zeros(m,dtype=torch.long,device=dev)
        moments=[(torch.zeros_like(x),torch.zeros_like(x))for x in [a,b,v]]
        for step in range(steps):
            signed=cap*ys*(ys*pred<=1);ga=ka-Kt@signed;gb=-signed.sum(0)
            gw=lam[:,None]*w-torch.einsum('mnd,nm->md',Pt,signed)
            derivative=torch.where(soft[:,None],signs[:,None]*torch.sigmoid(v),torch.ones_like(v));gv=gw*derivative
            norm=torch.sqrt((ga*ga).sum(0)+gb*gb+(gv*gv).sum(1));scale=torch.clamp(1/(norm+1e-6),max=1.)*active
            ga*=scale;gb*=scale;gv*=scale[:,None];lr=1e-5+.5*(.001-1e-5)*(1+math.cos(math.pi*step/steps))
            for par,g,(mu,var)in zip([a,b,v],[ga,gb,gv],moments):
                mu.mul_(.9).add_(g,alpha=.1);var.mul_(.999).addcmul_(g,g,value=.001)
                update=(mu/(1-.9**(step+1)))/(torch.sqrt(var/(1-.999**(step+1)))+1e-8)
                mask=active[None,:]if par is a else active[:,None]if par is v else active
                decay=wd[None,:]if par is a else wd[:,None]if par is v else wd
                par.mul_(1-lr*decay*mask);par.add_(update*mask,alpha=-lr)
            v=torch.where(hard[:,None],torch.clamp(v,min=0),v)
            obj,pred,ka,w=evaluate(a,b,v);improve=active&(obj<best-1e-4)
            ba[:,improve]=a[:,improve];bb[improve]=b[improve];bv[improve]=v[improve];best[improve]=obj[improve];beststep[improve]=step+1
            bad=torch.where(improve,0,bad+active);done=active&(bad>=80);stop[done]=step+1;active&=~done
            if not active.any():break
        final,_,_,w=evaluate(ba,bb,bv)
        if not torch.allclose(best,final,rtol=1e-10,atol=1e-10):raise AssertionError('Checkpoint mismatch')
    return {'a':ba.cpu().numpy(),'b':bb.cpu().numpy(),'w':w.cpu().numpy(),'raw_parameters':bv.cpu().numpy(),'objective':final.cpu().numpy(),'updates':stop.cpu().numpy(),'best_update':beststep.cpu().numpy(),'device':str(dev)}

def independent_reference(K,P,y,spec,*,steps=100,seed=0,device='cpu'):
    """Independent PyTorch autograd + built-in AdamW validation implementation."""
    torch.manual_seed(seed);n=len(y);dtype=torch.float64
    K=torch.tensor(K,dtype=dtype,device=device);P=torch.tensor(np.zeros_like(P)if spec['kind']=='kernel'else P,dtype=dtype,device=device)
    alpha=torch.nn.Parameter((torch.randn(n,1,device=device)*.01).to(dtype));rw=(torch.randn(2,device=device)*.05).to(dtype)
    initial=torch.ones(2,dtype=dtype,device=device)if spec['kind']in ['softplus','reverse']else rw if spec['init']=='random'else torch.zeros(2,dtype=dtype,device=device)if spec['init']=='zero'else torch.full((2,),1.3132616877555847,dtype=dtype,device=device)
    v=torch.nn.Parameter(initial);b=torch.nn.Parameter(torch.zeros(1,dtype=dtype,device=device));params=[alpha,b,v]
    sign=torch.tensor(2*y-1,dtype=dtype,device=device);caps=torch.tensor(np.where(y==1,20/(2*(y==1).sum()),20/(2*(y==0).sum())),dtype=dtype,device=device)
    def getw():return (-1 if spec['kind']=='reverse'else 1)*torch.nn.functional.softplus(v)if spec['kind']in ['softplus','reverse']else v
    def objective():
        w=getw();a=alpha[:,0];pred=K@a+P@w+b
        return .5*a@(K@a)+.5*spec['lambda']*(w@w)+caps@torch.clamp(1-sign*pred,min=0)
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=spec['wd']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=1e-5)
    best=float(objective().detach());snap=[x.detach().clone()for x in params];bad=0
    for i in range(steps):
        opt.zero_grad();objective().backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();sch.step()
        with torch.no_grad():
            if spec['kind']=='hard':v.clamp_(min=0)
            obj=float(objective())
            if obj<best-1e-4:best=obj;snap=[x.detach().clone()for x in params];bad=0
            else:bad+=1
        if bad>=80:break
    with torch.no_grad():
        for x,z in zip(params,snap):x.copy_(z)
        return {'a':alpha[:,0].cpu().numpy(),'b':float(b),'w':getw().cpu().numpy(),'objective':float(objective())}

if __name__=='__main__':
    from scipy.spatial.distance import cdist
    torch.set_num_threads(4);rng=np.random.RandomState(442);G=rng.normal(size=(36,4));K=np.exp(-.2*cdist(G,G,'sqeuclidean'));P=rng.normal(size=(36,2));y=np.tile([0,1],18)
    definitions=arms();result=fit_original(K,P,y,definitions,steps=150,device='cpu')
    for j,s in enumerate(definitions):
        other=independent_reference(K,P,y,s,steps=150)
        for key in ['a','b','w','objective']:
            x=result[key][:,j]if key=='a'else result[key][j]
            np.testing.assert_allclose(x,other[key],rtol=1e-9,atol=1e-9,err_msg=s['name']+' '+key)
    print('PASS: all ten variants match independent PyTorch autograd and built-in AdamW.')
