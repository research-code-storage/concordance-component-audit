"""Independent autograd check of 15 heterogeneous small-problem fits and of the grouped AUC summaries (SI Section E.4)."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'2')
from pathlib import Path
import json,tempfile,sys
import numpy as np,pandas as pd,torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from reconstructed_features import ROOT,test as test_geometry
from analyse_reconstructed import analyse
sys.path.insert(0,str(ROOT/'01_code/model_fitting'))
from additive_fit import arms,fit_additive,independent_reference


def main():
    torch.set_num_threads(2);test_geometry();rng=np.random.RandomState(671)
    by_name={a['name']:a for a in arms()}
    specs=[by_name[n] for n in ['projected_wd0','softplus_original','free_matched_positive_start','kernel_original','softplus_reverse']]*3
    g=rng.normal(size=(36,5));K=np.exp(-.2*cdist(g,g,'sqeuclidean'));P=rng.normal(size=(15,36,2));y=np.tile([0,1],18)
    fit=fit_additive(K,P,y,specs,steps=150,device='cpu')
    errors=[]
    for j,spec in enumerate(specs):
        ref=independent_reference(K,P[j],y,spec,steps=150)
        for key in ['a','b','w','objective']:
            value=fit[key][:,j]if key=='a'else fit[key][j];np.testing.assert_allclose(value,ref[key],atol=1e-9,rtol=1e-9);errors.append(float(np.max(np.abs(value-ref[key]))))
    designs=['historical_centred','matched_centred','revised_origin'];names=[d+'__'+s['name']for d in designs for s in arms() if s['name'] in ['softplus_original','kernel_original']]
    expected={};runs=[]
    target=Path(os.environ.get('CONCORDANCE_VERIFY_OUTPUT',str(Path(__file__).resolve().parents[4]/'generated/small_fit_checks')));target.mkdir(parents=True,exist_ok=True)
    folder=target/'auc_fixture';folder.mkdir(exist_ok=True)
    for n in [60]:
        for seed in range(10):
            full=rng.normal(size=(20,6));lin=full-rng.normal(size=(20,6))*.1;diag=rng.normal(size=(20,3));comp=np.repeat([f'c{i}'for i in range(10)],2);labels=np.repeat([0]*4+[1]*6,2)
            for domain in ['source','external','PHH']:
                np.savez_compressed(folder/f'n{n}_seed{seed}_{domain}_scores.npz',full=full,linear=lin,diagnostic=diag,arm_names=np.array(names),diagnostic_names=np.array(designs),compound=comp,label=labels)
            allscore=np.column_stack([full,diag]).reshape(10,2,9).mean(1);expected[n,seed]=[roc_auc_score(labels[::2],allscore[:,j])for j in range(9)]
            runs.append({'n_training_compounds':n,'seed':seed})
    (folder/'all_runs.json').write_text(json.dumps({'runs':runs,'config':{'protocol_sha256':'synthetic test fixture'}}))
    out=analyse(folder,write=False);assert len(out['cells'])==27
    for row in out['cells']:
        j=(names+[d+'__fixed_sum'for d in designs]).index(row['arm']);reference=[expected[row['training_compounds'],seed][j]for seed in range(10)]
        np.testing.assert_allclose(reference,row['draw_auc'],atol=1e-14)
    # Deleting one planned draw must fail instead of silently dropping its results.
    (folder/'all_runs.json').write_text(json.dumps({'runs':runs[:-1],'config':{'protocol_sha256':'synthetic test fixture'}}))
    try:analyse(folder,write=False)
    except ValueError as err:assert 'Incomplete comparison'in str(err)
    else:raise AssertionError('An incomplete comparison was accepted')
    report={'status':'PASS','independent_heterogeneous_fits':15,'maximum_parameter_or_objective_error':max(errors)}
    (target/'small_fit_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    print('Small-problem fits agree with the independent implementation',flush=True)

if __name__=='__main__':main()
