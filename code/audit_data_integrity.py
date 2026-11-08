"""Read-only audit of released matrices, metadata and every main train/test split.

Missing provenance fields are inventoried separately from missing model inputs.
Outputs go to generated/; no source matrix, label or fitted state is changed.
"""
from pathlib import Path
from itertools import combinations
import ast
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/evidence/05_reconstruction'

def main():
    tree=ast.parse((BASE/'code/reconstructed_features.py').read_text(encoding='utf-8'))
    aliases=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='ALIASES' for t in n.targets))
    def canonical(values):
        return pd.Series(values,dtype=str).str.strip().str.lower().replace(aliases).to_numpy()
    report={'scope':'Released feature-level matrices and saved main splits; no raw expression/reads reconstruction',
            'matrices':{},'metadata':{},'splits':[]}
    metas={};matrices={}
    pathways=[x.split('\t')[0] for x in (BASE/'assets/pathways_54.gmt').read_text().splitlines()]
    for coll,expected in [('mhedili',7818),('dilimap',763)]:
        m=pd.read_csv(BASE/f'data/{coll}_metadata.csv',keep_default_na=False)
        assert len(m)==expected and m.sample_id.is_unique,(coll,'metadata identities')
        assert not (m[['sample_id','chemical','condition','labels','tier','cell']].astype(str).apply(lambda x:x.str.strip())=='').any().any()
        m['ck']=canonical(m.chemical);m['labels']=pd.to_numeric(m.labels)
        assert set(m.labels)<={0,1},(coll,'nonbinary labels')
        treated=m[m.condition.eq('Treated')]
        conflicts=treated.groupby('ck').labels.nunique().loc[lambda x:x>1]
        assert conflicts.empty,(coll,'conflicting labels',conflicts.to_dict())
        missing={c:int(m[c].astype(str).str.strip().str.lower().isin(['','not_recorded','unknown','nan','na','n/a']).sum()) for c in m.columns}
        report['metadata'][coll]={'rows':len(m),'missing_or_unrecorded_fields':{k:v for k,v in missing.items() if v},'treated_label_conflicts':0}
        metas[coll]=m.set_index('sample_id',drop=False)
        for kind,cols in [('embeddings',[f'emb_{i}' for i in range(768)]),('pathways',pathways)]:
            f=pd.read_csv(BASE/f'data/{coll}_{kind}.csv')
            assert f.columns.tolist()==['sample_id']+cols,(coll,kind,'column names/order')
            assert f.sample_id.tolist()==m.sample_id.tolist(),(coll,kind,'metadata order or missing row')
            x=f[cols].to_numpy(dtype=float)
            assert np.isfinite(x).all(),(coll,kind,'nonfinite features')
            exact=int(pd.DataFrame(x).duplicated().sum())
            rounded=int(pd.DataFrame(np.round(x,10)).duplicated().sum())
            assert exact==rounded==0,(coll,kind,'duplicate features')
            report['matrices'][f'{coll}_{kind}']={'rows':len(f),'features':len(cols),'nonfinite':0,'duplicate_ids':0,
                                                   'exact_duplicate_rows':exact,'rounded_10_decimal_duplicate_rows':rounded,
                                                   'metadata_order_matches':True,'column_order_matches':True}
            matrices[coll,kind]=x
    report['cross_collection']={}
    for kind in ['embeddings','pathways']:
        source={row.tobytes() for row in matrices['mhedili',kind]}
        common=sum(row.tobytes() in source for row in matrices['dilimap',kind])
        assert common==0,(kind,'shared cross-collection vector')
        report['cross_collection'][kind+'_shared_vectors']=common
    common=set(metas['mhedili'].index)&set(metas['dilimap'].index)
    assert not common,('shared sample IDs',sorted(common))
    report['cross_collection']['shared_sample_ids']=len(common)
    ex=BASE/'results/experiments';panels={}
    states=sorted(ex.glob('n*_seed*_state.npz'));assert len(states)==10
    for p in states:
        z=np.load(p,allow_pickle=False);train_ids=z['train_ids'].astype(str)
        assert len(train_ids)==len(set(train_ids))
        train=metas['mhedili'].loc[train_ids]
        assert train.tier.eq(3).all() and train.condition.eq('Treated').all()
        train_comp=set(train.ck)
        expected_n=int(p.name.split('_')[0][1:]);assert len(train_comp)==expected_n
        for panel in ['source','external','PHH']:
            q=np.load(ex/(p.name.replace('_state.npz',f'_{panel}_scores.npz')),allow_pickle=False)
            ids=q['sample_id'].astype(str);comp=canonical(q['compound']);labels=q['label']
            assert len(ids)==len(set(ids)) and set(labels)<={0,1}
            meta=metas['dilimap' if panel=='external' else 'mhedili'].loc[ids]
            assert np.array_equal(comp,meta.ck.to_numpy()) and np.array_equal(labels,meta.labels.to_numpy()),(p.name,panel,'metadata/score mismatch')
            assert meta.condition.eq('Treated').all()
            for key in ['full','linear','kernel','diagnostic']:
                assert q[key].shape[0]==len(ids) and np.isfinite(q[key]).all()
            shared_ids=set(train_ids)&set(ids);shared_comp=train_comp&set(comp)
            assert not shared_ids and not shared_comp,(p.name,panel,'train/test overlap')
            report['splits'].append({'state':p.name,'panel':panel,'shared_ids':len(shared_ids),'shared_compounds':len(shared_comp)})
            identity=(tuple(ids),tuple(comp),tuple(labels))
            if panel in panels:assert identity==panels[panel]
            else:panels[panel]=identity
    source=metas['mhedili'];source_comp=set(source.loc[source.tier.eq(3)&source.condition.eq('Treated'),'ck'])
    report['evaluation_panels']={k:{'profiles':len(v[0]),'compounds':len(set(v[1]))} for k,v in panels.items()}
    # The frozen-reference study reuses the final metadata annotations too.
    frozen=ROOT/'data/evidence/07_reference_study/results/fits'
    report['frozen_score_label_checks']=[]
    for panel,collection in [('external','dilimap'),('PHH','mhedili')]:
        files=sorted(frozen.glob(f'n*_seed*_{panel}.npz'));assert len(files)==31
        for p in files:
            q=np.load(p,allow_pickle=False);ids=q['sample_id'].astype(str)
            meta=metas[collection].loc[ids]
            assert len(ids)==len(set(ids)) and meta.condition.eq('Treated').all()
            assert np.array_equal(canonical(q['compound']),meta.ck.to_numpy()),(p.name,'compound mismatch')
            assert np.array_equal(q['label'],meta.labels.to_numpy()),(p.name,'label mismatch')
            report['frozen_score_label_checks'].append({'file':p.name,'profiles':len(ids),'status':'PASS'})
    report['cross_panel']={}
    for panel in ['external','PHH']:
        assert not source_comp&set(panels[panel][1]),(panel,'overlap with full source collection')
    for a,b in combinations(['external','PHH'],2):
        ni=len(set(panels[a][0])&set(panels[b][0]));nc=len(set(panels[a][1])&set(panels[b][1]))
        assert ni==nc==0,(a,b,'overlapping panels')
        report['cross_panel'][a+' vs '+b]={'shared_ids':ni,'shared_compounds':nc}
    report['status']='PASS'
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    (out/'data_integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Four feature matrices: finite, unique and aligned to metadata and expected columns')
    print('All 10 main training draws x 3 reported evaluation panels: no sample/compound overlap or score-label mismatch')
    print('All 31 frozen-reference draws x 2 evaluation panels: labels and compounds agree with final metadata')
    print('Missing provenance/batch fields inventoried separately; see generated/data_integrity_audit.json')

if __name__=='__main__':main()
