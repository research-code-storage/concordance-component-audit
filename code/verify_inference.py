"""Replay seeded samples and independently check the coverage reference integrals."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.integrate import quad
import inference_coverage as ic
ROOT=ic.ROOT;EV=ROOT/'data/evidence/08_inference'

def verify(replay=None):
 design=json.loads((EV/'inference_design.json').read_text());truth=json.loads((EV/'reference_values.json').read_text())
 summary=json.loads((EV/'summary.json').read_text())
 assert [ic.summarize(c,truth[str(c['id'])],EV/f"case_{c['id']:02}.npz") for c in design['scenarios']]==summary['cells']
 checks=[]
 for c in design['scenarios']:
  sid=c['id'];z=np.load(EV/f'case_{sid:02}.npz')
  assert z['estimate'].shape==(2000,3)
  for r in (0,999,1999):
   rng=np.random.default_rng(np.random.SeedSequence([ic.MASTER,sid,r]))
   if c['kind']=='binary':
    K=ic.binary_matrices(rng,c['n'],c['a'],c['s'],c['distribution']);t,se=ic.binary_estimate(K)
   else:
    w,K,cf=ic.survival_matrices(rng,c['n'],c['a'],c['censor_rate']);t,se=ic.survival_estimate(w,K)
    assert cf==z['censor_fraction'][r]
   np.testing.assert_allclose(t,z['estimate'][r],rtol=0,atol=2e-14)
   np.testing.assert_allclose(se,z['se'][r],rtol=0,atol=2e-14)
   if c['bootstrap'] and r==0:
    rngb=np.random.default_rng(np.random.SeedSequence([ic.BOOT_MASTER,sid,r]))
    ci,bv=ic.bootstrap_binary(K,rngb,999) if c['kind']=='binary' else ic.bootstrap_survival(w,K,rngb,999)
    np.testing.assert_allclose(ci,np.stack((z['bootstrap_lower'][r],z['bootstrap_upper'][r])),rtol=0,atol=2e-14)
    assert bv==z['bootstrap_valid'][r]
  if replay:
   other=np.load(Path(replay)/f'case_{sid:02}.npz');assert set(z.files)==set(other.files)
   for k in z.files:np.testing.assert_allclose(z[k],other[k],rtol=0,atol=2e-14,equal_nan=True)
  if c['kind']=='binary':
   ref=ic.binary_truth(c['a'],c['s'],c['distribution'])
   np.testing.assert_allclose(ref['value'],truth[str(sid)]['value'],rtol=0,atol=2e-12)
  checks.append({'scenario':sid,'seeded_replicates':[0,999,1999],'full_replay':bool(replay)})
 # Independent convolution verifies the analytic standardized-t3 pair density.
 density=lambda x:2/(np.pi*(1+x*x)**2)
 for d in (-3.,-.2,0.,1.,4.):
  numeric=quad(lambda x:density(x)*density(x-d),-np.inf,np.inf,epsabs=1e-12)[0]
  exact=4*(20+d*d)/(np.pi*(4+d*d)**3)
  assert abs(numeric-exact)<1e-11
 # Independent pseudo-random pair integration checks all 36 binary truths and four
 # survival reference cases; comparisons use their own sampling SE.
 rng=np.random.default_rng(20261012);M=1000000;reference_checks=[];seen=set()
 for c in design['scenarios']:
  key=(c['kind'],c['a'],c.get('s'),c.get('distribution'),c.get('censor_rate'))
  if key in seen:continue
  seen.add(key)
  if c['kind']=='binary':
   err=rng.standard_t(3,(2,M))/np.sqrt(3) if c['distribution']=='t3' else rng.normal(size=(2,M))
   l1=.6+err[0];l0=err[1]
   if c['distribution']=='rounded_normal':l1=np.rint(l1);l0=np.rint(l0)
   e=c['a']*(rng.normal(size=M)-rng.normal(size=M)+.6*c['s'])
   val=ic.pair_kernels(l1-l0,e);est=val.mean(1);se=val.std(1,ddof=1)/np.sqrt(M)
  else:
   x1,z1,x2,z2=rng.normal(size=(4,M));r1=np.exp(x1+.6*z1);r2=np.exp(x2+.6*z2)
   w=r1/(r1+r2+2*c['censor_rate']);val=ic.pair_kernels(x1-x2,c['a']*(z1-z2))
   est=(val*w).mean(1)/w.mean();se=((val-est[:,None])*w/w.mean()).std(1,ddof=1)/np.sqrt(M)
  ref=np.array(truth[str(c['id'])]['value']);score=abs(est-ref)/np.maximum(se,1e-12)
  assert np.max(score)<5,(c,score)
  reference_checks.append({'scenario':c['id'],'independent_pairs':M,'standardized_error':score.tolist()})
 # Survival integration uncertainty: coverage changes under +/- 3 reference SE.
 sensitivities=[]
 for cell in summary['cells']:
  c=cell['scenario']
  if c['kind']!='survival':continue
  z=np.load(EV/f"case_{c['id']:02}.npz");t=z['estimate'];se=z['se'];ref=np.array(cell['truth']['value']);rs=np.array(cell['truth']['reference_se'])
  covers=[]
  for v in (ref-3*rs,ref,ref+3*rs):covers.append(((t-ic.Z975*se<=v)&(t+ic.Z975*se>=v)).mean(0).tolist())
  sensitivities.append({'scenario':c['id'],'coverage_minus_center_plus_3_reference_se':covers})
 out={'status':'PASS','seed_replay':checks,'reference_checks':reference_checks,'survival_reference_sensitivity':sensitivities,'projection_validation':ic.validate()}
 (ROOT/'generated').mkdir(exist_ok=True);(ROOT/'generated/inference_verification.json').write_text(json.dumps(out,indent=2))
 print('80 coverage cells, 240 seeded datasets, eight bootstrap samples and 40 independent reference checks passed')
 if replay:print('All 160000 replicate estimates/SEs and all 8000 bootstrap intervals reproduced')
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--full-replay',type=Path);a=p.parse_args();verify(a.full_replay)
