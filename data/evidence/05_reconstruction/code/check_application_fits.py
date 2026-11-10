"""Independent autograd refits of the three Softplus fits of one saved source draw (SI Section G.4)."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from scipy.spatial.distance import cdist
from reconstructed_features import ROOT,BASE,GF,load,origins,transform
sys.path.insert(0,str(ROOT/'01_code/model_fitting'))
from additive_fit import independent_reference


def main(folder,n,seed,device,output):
    torch.set_num_threads(4);rec=json.loads((folder/f'n{n}_seed{seed}.json').read_text());config=json.loads((folder/'config.json').read_text());state=np.load(folder/f'n{n}_seed{seed}_state.npz',allow_pickle=False)
    domains,_,_=load();s=domains['source'];mask=s.sample_id.isin(state['train_ids']).values;assert s.sample_id[mask].tolist()==state['train_ids'].tolist()
    G=(s.loc[mask,GF].values-state['embedding_mean'])/state['embedding_scale'];K=np.exp(-cdist(G,G,'sqeuclidean')/1536);records=[]
    for j,a in enumerate(config['arms']):
        if not a['name'].endswith('__softplus_original'):continue
        d=a['design'];o,_=origins(s,d!='historical_centred');raw=transform(s,o,rec['coordinate_assets'][d])[mask];sc=rec['coordinate_scalers'][d];P=(raw-sc['mean'])/sc['scale']
        ref=independent_reference(K,P,s.labels.values[mask],a,steps=1000,device=device);errors={}
        for key in ['a','b','w','objective']:
            saved=state['a'][:,j]if key=='a'else rec['objective'][j]if key=='objective'else state[key][j]
            np.testing.assert_allclose(saved,ref[key],atol=1e-8,rtol=1e-8,err_msg=a['name']+' '+key);errors[key]=float(np.max(np.abs(saved-ref[key])))
        records.append({'arm':a['name'],'errors':errors});print('PASS independent refit:',a['name'],flush=True)
    report={'status':'PASS','n_training_compounds':n,'seed':seed,'training_profiles':int(mask.sum()),'device':device,'comparisons':records}
    output.mkdir(parents=True,exist_ok=True);(output/'application_fit_checks.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=BASE/'results/fit_validation');p.add_argument('--training-count',type=int,default=4);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda',help='the saved fits were made on CUDA; other devices need not reproduce the optimizer path');p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.results,a.training_count,a.seed,a.device,a.output)
