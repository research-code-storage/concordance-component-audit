"""Recompute component scores and objectives from the supplied feature matrices and fitted parameters."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/'data/evidence'
sys.path.insert(0,str(EV/'05_reconstruction/code'))
import reconstructed_features as rf
sys.path.insert(0,str(EV/'07_reference_study/code'))
import reference as fr

def check(x,y,label,errors):
 x=np.asarray(x);y=np.asarray(y)
 np.testing.assert_allclose(x,y,rtol=1e-8,atol=1e-8,err_msg=label)
 errors.append(float(np.max(np.abs(x-y))))

def main():
 errors=[];objectives=[];count=0;score_files=0
 domains,overlap,held=rf.load();source=domains['source']
 origins={d:{p:rf.origins(f,d!='historical_centred')[0] for p,f in domains.items()}
          for d in ['historical_centred','matched_centred','revised_origin']}
 for folder in [EV/'05_reconstruction/results/experiments',EV/'05_reconstruction/results/fit_validation']:
  cfg=json.loads((folder/'config.json').read_text());specs=cfg['arms']
  for path in sorted(folder.glob('*_state.npz')):
   z=np.load(path);prefix=path.name[:-10];rec=json.loads((folder/(prefix+'.json')).read_text())
   assert z['arm_names'].tolist()==[a['name'] for a in specs]
   train=source.sample_id.isin(z['train_ids']).to_numpy();assert source.sample_id[train].tolist()==z['train_ids'].tolist()
   G=(source.loc[train,rf.GF].to_numpy()-z['embedding_mean'])/z['embedding_scale']
   K=np.exp(-cdist(G,G,'sqeuclidean')/1536);P={}
   for d,asset in rec['coordinate_assets'].items():
    s=rec['coordinate_scalers'][d]
    P[d]={p:(rf.transform(f,origins[d][p],asset)-s['mean'])/s['scale'] for p,f in domains.items()}
   y=source.labels.to_numpy()[train];caps=np.where(y==1,20/(2*(y==1).sum()),20/(2*(y==0).sum()))
   kernel=K@z['a']
   for k,a in enumerate(specs):
    pp=P[a['design']]['source'][train].copy()
    if a['kind']=='kernel':pp[:]=0
    w=z['w'][k];pred=kernel[:,k]+pp@w+z['b'][k]
    obj=.5*z['a'][:,k]@kernel[:,k]+.5*a['lambda']*(w@w)+caps@np.maximum(0,1-(2*y-1)*pred)
    check(obj,rec['objective'][k],prefix+' objective',objectives)
    if a['kind']=='softplus':check(w,np.logaddexp(0,z['raw_parameters'][k]),prefix+' Softplus',errors)
   count+=len(specs)
   for panel,f in domains.items():
    score=folder/f'{prefix}_{panel}_scores.npz'
    if not score.exists():continue
    q=np.load(score);mask=f.sample_id.isin(q['sample_id']).to_numpy()
    assert f.sample_id[mask].tolist()==q['sample_id'].tolist()
    GE=(f.loc[mask,rf.GF].to_numpy()-z['embedding_mean'])/z['embedding_scale']
    ker=np.exp(-cdist(GE,G,'sqeuclidean')/1536)@z['a']
    lin=np.column_stack([np.zeros(mask.sum()) if a['kind']=='kernel' else P[a['design']][panel][mask]@z['w'][k] for k,a in enumerate(specs)])
    for key,x in [('kernel',ker),('linear',lin),('full',ker+lin+z['b'])]:check(x,q[key],prefix+'/'+panel+'/'+key,errors)
    diag=np.column_stack([P[d][panel][mask].sum(1) for d in q['diagnostic_names']])
    check(diag,q['diagnostic'],prefix+' coordinate sum',errors);score_files+=1
   print('State parameters:',prefix,len(specs),'models',flush=True)

 folder=EV/'07_reference_study/results/fits';assets=json.loads((EV/'07_reference_study/assets/frozen_reference.json').read_text())
 fd,O,masks,_=fr.load();s=fd['source'];asset=assets['recipes']['balanced_origin_all']
 Gs={p:(f[rf.GF].to_numpy()-assets['embedding_mean'])/assets['embedding_scale'] for p,f in fd.items()}
 Ps={p:fr.project(f,O[p],asset) for p,f in fd.items()}
 for path in sorted(folder.glob('*_state.npz')):
  z=np.load(path);prefix=path.name[:-10];rec=json.loads((folder/(prefix+'.json')).read_text())
  mask=s.sample_id.isin(z['train_ids']).to_numpy();assert s.sample_id[mask].tolist()==z['train_ids'].tolist()
  G=Gs['source'][mask];K=np.exp(-cdist(G,G,'sqeuclidean')/1536);y=s.labels.to_numpy()[mask]
  f=s.loc[mask];labels=f.groupby('ck').labels.first();counts=f.ck.value_counts()
  caps=np.array([20/(2*(labels==v).sum()*counts[c]) for c,v in zip(f.ck,f.labels)])
  check(caps,z['caps'],prefix+' compound weights',errors)
  a=z['a'][:,0];w=z['w'][0];ka=K@a;pred=ka+Ps['source'][mask]@w+z['b'][0]
  obj=.5*a@ka+.0005*(w@w)+caps@np.maximum(0,1-(2*y-1)*pred)
  check(obj,rec['objective'][0],prefix+' frozen objective',objectives)
  check(w,np.logaddexp(0,z['raw'][0]),prefix+' frozen Softplus',errors)
  for panel in ['PHH','external']:
   q=np.load(folder/f'{prefix}_{panel}.npz');take=fd[panel].sample_id.isin(q['sample_id']).to_numpy()
   assert fd[panel].sample_id[take].tolist()==q['sample_id'].tolist()
   ker=np.exp(-cdist(Gs[panel][take],G,'sqeuclidean')/1536)@a;lin=Ps[panel][take]@w
   for key,x in [('kernel',ker),('linear',lin),('full',ker+lin+z['b'][0])]:check(x,q[key][:,0],prefix+'/'+panel+'/'+key,errors)
   score_files+=1
  count+=1;print('Frozen state:',prefix,flush=True)
 assert count==94 and score_files==92
 report={'status':'PASS','models':count,'score_archives':score_files,'maximum_score_or_parameter_error':max(errors),
         'maximum_objective_error':max(objectives),'method':'feature and parameter replay without refitting'}
 (ROOT/'generated').mkdir(exist_ok=True);(ROOT/'generated/state_replay.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 with threadpool_limits(limits=2):main()
