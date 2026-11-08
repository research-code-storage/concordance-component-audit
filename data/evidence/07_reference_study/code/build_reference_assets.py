"""Recompute the frozen coordinate assets from the source data and compare them with assets/frozen_reference.json."""
import argparse,json
from threadpoolctl import threadpool_limits
from reference import *

def check(a,b):
    if isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in b:check(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):check(x,y)
    elif isinstance(b,float):assert np.isclose(a,b,rtol=1e-10,atol=1e-12)
    else:assert a==b

def main(out):
    domains,O,masks,refs=load();stored=json.loads((BASE/'assets/frozen_reference.json').read_text());new={}
    for name,r in recipes().items():
        new[name]=fit_asset(domains['source'],O['source'],r);check(new[name],stored['recipes'][name])
    w=weights(domains['source'].loc[masks['source'],'ck']);m,s=moments(domains['source'].loc[masks['source'],GF].to_numpy(),w)
    np.testing.assert_allclose(m,stored['embedding_mean'],rtol=1e-10,atol=1e-12);np.testing.assert_allclose(s,stored['embedding_scale'],rtol=1e-10,atol=1e-12)
    out.mkdir(parents=True,exist_ok=True);(out/'reference_assets.json').write_text(json.dumps(new,indent=2))
    print('PASS: coordinate assets and embedding scaler recomputed and equal to the saved assets.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.output)
