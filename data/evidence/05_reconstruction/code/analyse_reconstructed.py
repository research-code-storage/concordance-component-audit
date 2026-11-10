"""Paired compound resampling of every reported comparison (SI Table S4 and Section H.3)."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1]

def analyse(folder,write=True,output=None):
    record=json.loads((folder/'all_runs.json').read_text());runs=record['runs'];expected={(n,s)for n in [60]for s in range(10)}
    if {(r['n_training_compounds'],r['seed'])for r in runs}!=expected:raise ValueError('Incomplete comparison; refuse selective summaries')
    cells=[];all_boot={}
    for n in [60]:
        for domain in ['source','external','PHH']:
            scores=[];linear=[];previous=None
            for seed in range(10):
                d=np.load(folder/f'n{n}_seed{seed}_{domain}_scores.npz',allow_pickle=False);names=d['arm_names'].tolist()+[str(x)+'__fixed_sum'for x in d['diagnostic_names']]
                full=np.column_stack([d['full'],d['diagnostic']]);lin=np.column_stack([d['linear'],d['diagnostic']]);meta=pd.DataFrame({'compound':d['compound'],'label':d['label']})
                if (meta.groupby('compound').label.nunique()!=1).any():raise ValueError('Conflicting evaluation labels')
                f=pd.DataFrame(full,columns=names);f['compound']=meta.compound;f=f.groupby('compound',sort=True)[names].mean()
                l=pd.DataFrame(lin,columns=names);l['compound']=meta.compound;l=l.groupby('compound',sort=True)[names].mean();labels=meta.groupby('compound',sort=True).label.first().to_numpy()
                identity=(names,f.index.tolist(),labels.tolist())
                if previous is not None and identity!=previous:raise ValueError('Comparison panels differ across source draws')
                previous=identity;scores.append(f.values);linear.append(l.values)
            scores=np.array(scores);linear=np.array(linear);pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
            if len(pos)<2 or len(neg)<2:raise ValueError('Insufficient compounds per class for this uncertainty summary')
            def contribution(x):
                diff=x[:,pos,None,:]-x[:,None,neg,:];return (diff>0).astype(float)+.5*(diff==0)
            h=contribution(scores);hl=contribution(linear);draw=h.mean((1,2));averaged=h.mean(0).transpose(2,0,1);deleted=hl.mean(0).transpose(2,0,1)
            rng=np.random.RandomState(20260922);bp=rng.multinomial(len(pos),np.full(len(pos),1/len(pos)),size=20000);bn=rng.multinomial(len(neg),np.full(len(neg),1/len(neg)),size=20000)
            boot=np.einsum('bi,mij,bj->bm',bp,averaged,bn,optimize=True)/(len(pos)*len(neg));delboot=np.einsum('bi,mij,bj->bm',bp,averaged-deleted,bn,optimize=True)/(len(pos)*len(neg));all_boot[f'n{n}_{domain}']=boot;all_boot[f'n{n}_{domain}_deletion']=delboot
            baseline='historical_centred__softplus_original';b=names.index(baseline);ci=lambda v:np.quantile(v,[.025,.975]).tolist()
            for j,name in enumerate(names):
                cells.append({'training_compounds':n,'domain':domain,'arm':name,'auc_mean':float(draw[:,j].mean()),'draw_sd':float(draw[:,j].std(ddof=1)),'draw_auc':draw[:,j].tolist(),'compound_bootstrap_ci':ci(boot[:,j]),'paired_baseline':baseline,'paired_auc_difference':float((draw[:,j]-draw[:,b]).mean()),'paired_difference_ci':ci(boot[:,j]-boot[:,b]),'mean_kernel_deletion_difference':float((averaged[j]-deleted[j]).mean()),'kernel_deletion_ci':ci(delboot[:,j]),'changed_auc_pairs':[int(np.sum(h[k,:,:,j]!=hl[k,:,:,j]))for k in range(10)],'compounds':len(labels),'positive':len(pos),'negative':len(neg)})
    result={'fitted_models':60,'fixed_diagnostic_scores':30,'summary_cells':len(cells),'bootstrap_replicates':20000,'bootstrap_seed':20260922,'estimand':'mean of ten draw-specific compound AUCs','cells':cells}
    if write:
        dest=Path(output)if output is not None else folder;dest.mkdir(parents=True,exist_ok=True)
        (dest/'summary.json').write_text(json.dumps(result,indent=2));np.savez_compressed(dest/'bootstrap.npz',**all_boot)
        pd.DataFrame(cells).to_csv(dest/'all_results.csv',index=False)
        fmt=lambda x:f'{x:.4f}';interval=lambda x:'['+', '.join(map(fmt,x))+']'
        print('Completed all',len(cells),'reported cells.',flush=True)
        for r in cells:
            if r['training_compounds']==60 and r['domain']=='external':print(r['arm'],fmt(r['auc_mean']),interval(r['compound_bootstrap_ci']),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=BASE/'results/experiments');p.add_argument('--output-dir',type=Path);a=p.parse_args();analyse(a.results,output=a.output_dir)
