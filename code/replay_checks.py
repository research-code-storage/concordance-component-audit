"""Recompute released results into generated/ and compare them with the saved records."""
from pathlib import Path
import argparse,importlib.util,json,sys,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/'data/evidence';OUT=ROOT/'generated'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def dump(name,value):
    OUT.mkdir(exist_ok=True);(OUT/(name+'.json')).write_text(json.dumps(value,indent=2))
def compare(a,b,path='root',tol=1e-10):
    if isinstance(b,dict):
        assert a.keys()==b.keys(),path
        for k in b:compare(a[k],b[k],path+'/'+k,tol)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),tol)
    elif isinstance(b,(int,float)) and not isinstance(b,bool):np.testing.assert_allclose(a,b,rtol=tol,atol=tol,err_msg=path)
    else:assert a==b,path
def main_intervals():
    m=module(EV/'05_reconstruction/code/analyse_reconstructed.py','main_analysis')
    folder=EV/'05_reconstruction/results/experiments';out=OUT/'main_inference';out.mkdir(parents=True,exist_ok=True)
    result=m.analyse(folder,write=True,output=out);compare(result,json.loads((folder/'summary.json').read_text()))
    x=np.load(out/'bootstrap.npz');y=np.load(folder/'bootstrap.npz');err=0.;assert set(x.files)==set(y.files)
    for k in x.files:
        np.testing.assert_allclose(x[k],y[k],rtol=0,atol=1e-12);err=max(err,float(np.max(abs(x[k]-y[k]))))
    dump('main_interval_audit',{'cells':27,'intervals':81,'max_bootstrap_error':err});print('AUC and paired intervals regenerated for 27 cells:',err)

def study(part):
    sub={'survival':'survival_illustration.py','simulation':'simulate_bounds.py'}[part]
    out=OUT/(part+'_replay');subprocess.run([sys.executable,str(EV/'07_reference_study/code'/sub),'--output',str(out)],check=True)
    names=['all_14400_replicates.csv'] if part=='simulation' else ['heldout_scores.csv','complete_case_inputs.csv','comparable_pairs.csv']
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(EV/'07_reference_study/results'/part/name),check_exact=False,atol=1e-10,rtol=1e-10)
    dump(part+'_replay_audit',{'status':'PASS','source':'Generator rerun; every saved row compared'})
    print(part,'generation replay agrees with saved rows')

def fitted_sim():
    folder=EV/'04_results/extension_double';df=pd.read_csv(folder/'simulation_replicates.csv');assert len(df)==2400
    maxerr=0.;pair_rows=[]
    for i in range(200):
        z=np.load(folder/f'simulation_rep{i:03}.npz');pos=z['test_labels']==1;neg=~pos
        H=lambda v:(v>0).astype(float)+.5*(v==0)
        hf=H(z['full'][:,pos,None,:]-z['full'][:,None,neg,:]);hl=H(z['linear'][:,pos,None,:]-z['linear'][:,None,neg,:])
        D=z['linear'][:,pos,None,:]-z['linear'][:,None,neg,:]
        E=z['kernel'][:,pos,None,:]-z['kernel'][:,None,neg,:]
        # The saved intercept is separate; it cancels from every pair margin.
        np.testing.assert_allclose(np.ptp(z['full']-z['linear']-z['kernel'],axis=1),0,atol=1e-12,rtol=0)
        np.testing.assert_allclose(z['full'][:,pos,None,:]-z['full'][:,None,neg,:],D+E,atol=1e-12,rtol=0)
        rho=np.where(D==0,.5*(E!=0),np.where(abs(E)>abs(D),1.,.5*(abs(E)==abs(D))))
        for k,shift in enumerate([0,6,16]):
            for j in range(4):
                rows=df[(df.replicate==i)&(df['shift']==shift)&(df.informative==(j<2))&(abs(df.initial-(1.3132616877555847 if j%2==0 else 0))<1e-10)];assert len(rows)==1
                vals=[hf[k,:,:,j].mean(),(hf-hl)[k,:,:,j].mean()];ref=rows.iloc[0][['auc','auc_difference']].to_numpy(float)
                maxerr=max(maxerr,float(max(abs(np.array(vals)-ref))));np.testing.assert_allclose(vals,ref,atol=1e-12,rtol=0)
                A=float(abs(hf-hl)[k,:,:,j].mean());B=float(rho[k,:,:,j].mean())
                assert abs(vals[1])<=A+1e-12 and A<=B+1e-12
                pair_rows.append(dict(replicate=i,shift=shift,informative=j<2,initial=1.3132616877555847 if j%2==0 else 0.,A=A,B=B))
    saved=json.loads((folder/'simulation_summary.json').read_text())['cells']
    for r in saved:
        g=df[(df['shift']==r['shift'])&(df.informative==r['informative'])&(abs(df.initial-r['initial'])<1e-10)];assert len(g)==200
        for name in ['auc','auc_difference']:
            np.testing.assert_allclose([g[name].mean(),g[name].std(ddof=1)/np.sqrt(200)],[r[name+'_mean'],r[name+'_mcse']],atol=1e-12,rtol=0)
    pairs=pd.DataFrame(pair_rows);pairs.to_csv(OUT/'fitted_component_replicates.csv',index=False)
    for r in saved:
        g=pairs[(pairs['shift']==r['shift'])&(pairs.informative==r['informative'])&(abs(pairs.initial-r['initial'])<1e-10)]
        for name in ('A','B'):
            r[name+'_mean']=float(g[name].mean());r[name+'_mcse']=float(g[name].std(ddof=1)/np.sqrt(200))
    dump('fitted_component_audit',{'source':'data/evidence/04_results/extension_double/simulation_rep000.npz through simulation_rep199.npz','cells':saved,'pair_rows':len(pairs),'chain_violations':0})
    dump('fitted_simulation_audit',{'status':'PASS','training_draws':200,'evaluation_rows':2400,'maximum_auc_difference':maxerr})
    print('Fitted simulation: 2400 saved-score evaluations checked; maximum difference',maxerr)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['main','survival','simulation','fitted_sim']);a=p.parse_args();OUT.mkdir(exist_ok=True)
    {'main':main_intervals,'survival':lambda:study('survival'),'simulation':lambda:study('simulation'),'fitted_sim':fitted_sim}[a.part]()
