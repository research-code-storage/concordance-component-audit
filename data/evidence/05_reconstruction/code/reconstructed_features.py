"""Strict cohort joins and fitted perturbation coordinates for the reconstruction."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.cluster import KMeans
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'05_reconstruction'
sys.path.insert(0,str(ROOT/'01_code/10_extension'))
from data import draw_compounds
PC=[line.split('\t')[0]for line in (BASE/'assets/pathways_54.gmt').read_text().splitlines()]
GF=[f'emb_{i}'for i in range(768)]
KEYWORDS=['TCA','OXIDATIVE','RESPIRATORY','METABOLISM','GLYCOLYSIS','LIPID','CITRIC','PHOSPHORYLATION','MITOCHONDRIA','ENERGY']
ALIASES={'paracetamol':'acetaminophen','cyclosporine a':'cyclosporin a','cyclosporine':'cyclosporin a','ciclosporin':'cyclosporin a','d-galactosamine':'galactosamine','glibenclamide':'glyburide'}

def join(a,b):
    if not a.sample_id.is_unique or not b.sample_id.is_unique:raise ValueError('Nonunique join ID')
    if not set(a.sample_id)<=set(b.sample_id):raise ValueError('Missing feature ID')
    c=a.merge(b,on='sample_id',how='left',validate='one_to_one',sort=False)
    if c.sample_id.tolist()!=a.sample_id.tolist():raise ValueError('Join order changed')
    return c

def load(with_embeddings=True):
    frames={}
    for name in ['mhedili','dilimap']:
        m=pd.read_csv(BASE/f'data/{name}_metadata.csv',keep_default_na=False);p=pd.read_csv(BASE/f'data/{name}_pathways.csv');m=join(m,p)
        if with_embeddings:m=join(m,pd.read_csv(BASE/f'data/{name}_embeddings.csv')[['sample_id']+GF])
        m['ck']=m.chemical.astype(str).str.strip().str.lower().replace(ALIASES)
        m['control']=m.condition.eq('Control');m['labels']=pd.to_numeric(m.labels).astype(int)
        if not np.isfinite(m[PC+(GF if with_embeddings else[])].values).all():raise ValueError('Nonfinite feature')
        if (m[~m.control].groupby('ck').labels.nunique()>1).any():raise ValueError('Conflicting compound labels')
        frames[name]=m
    s=frames['mhedili'].query('tier == 3').copy();source_comp=set(s.loc[~s.control,'ck']);held=draw_compounds(s[~s.control].groupby('ck').labels.first(),20,12345)
    domains={'source':s,'external':frames['dilimap']}
    for cell in ['PHH']:domains[cell]=frames['mhedili'][frames['mhedili'].cell.eq(cell)].copy()
    return domains,source_comp,held

def origins(frame,matched):
    X=frame[PC].values.astype(float);O=np.full_like(X,np.nan);records=[]
    keys=['cell']if not matched else ['source_file','cell','time','baseline_type','media','assay_batch']
    if matched and frame.cell.eq('DILImap').all():keys=['plate','time']
    groups=frame[keys].astype(str).agg('|'.join,axis=1).to_numpy()
    for key in sorted(set(groups)):
        rows=groups==key;controls=rows&frame.control.values
        if matched and frame.cell.eq('DILImap').all():controls&=frame.chemical.eq('DMSO').values
        n=int(controls.sum());record={'group':key,'n_controls':n,'reference_ids':frame.sample_id[controls].tolist(),'covered':n>=3}
        if n>=3:O[rows]=np.median(X[controls],axis=0)
        record['uncovered_treated_ids']=frame.sample_id[rows&~frame.control.values].tolist()if n<3 else[];records.append(record)
    return O,records

def fit_coordinates(source,origins_source,training_compounds,design):
    eligible=(source.ck.isin(training_compounds)|source.control).to_numpy()&np.isfinite(origins_source).all(1)
    positions=np.flatnonzero(eligible);raw=source[PC].to_numpy(float);delta=raw-origins_source
    if design=='historical_centred':
        # Similarity thinning is done after a source-training-only first z-score.
        # Historical whole-cohort z-scoring is not reused across held-out compounds.
        first=StandardScaler().fit(raw[eligible]);U=normalize(first.transform(raw[eligible]));keep=[]
        for chemical in sorted(source.iloc[positions].chemical.unique()):
            ix=np.flatnonzero(source.iloc[positions].chemical.to_numpy()==chemical);sim=U[ix]@U[ix].T;drop=set()
            for j in range(len(ix)):
                if j in drop:continue
                keep.append(positions[ix[j]]);drop.update(k for k in range(j+1,len(ix))if sim[j,k]>=.99)
        positions=np.array(keep,dtype=int)
    if len(positions)<8:raise ValueError('Insufficient source coordinate pool')
    scaler=StandardScaler(with_mean=design!='revised_origin').fit(delta[positions])
    centre=np.zeros(len(PC))if design=='revised_origin'else scaler.mean_;scale=scaler.scale_
    Z=(delta[positions]-centre)/scale;r=np.linalg.norm(Z,axis=1);lo,hi=np.quantile(r,[.15,.85]);mi=[i for i,c in enumerate(PC)if any(k in c for k in KEYWORDS)]
    low=normalize(KMeans(n_clusters=2,n_init=10,random_state=42).fit(normalize(Z[r<=lo])).cluster_centers_)
    high=normalize(KMeans(n_clusters=2,n_init=10,random_state=42).fit(normalize(Z[r>=hi])).cluster_centers_)
    if high[0,mi].mean()<high[1,mi].mean():high=high[[1,0]]
    if low[0,mi].mean()>low[1,mi].mean():low=low[[1,0]]
    record={'design':design,'centre':centre.tolist(),'scale':scale.tolist(),'centroids':np.vstack([high,low]).tolist(),'retained_ids':source.iloc[positions].sample_id.tolist(),'training_compounds':list(training_compounds),'reference_ids':source.loc[source.control,'sample_id'].tolist(),'q15':float(lo),'q85':float(hi),'n_init':10}
    assert set(source.iloc[positions].loc[lambda x:~x.control,'ck'])<=set(training_compounds)
    return record

def transform(frame,origin,assets):
    z=(frame[PC].values-origin-np.array(assets['centre']))/np.array(assets['scale']);r=np.linalg.norm(z,axis=1);u=z/np.where(r>0,r,1)[:,None];c=np.array(assets['centroids'])
    return np.column_stack([r,u@(c[1]-c[3])])

def test():
    rng=np.random.RandomState(61);x=rng.normal(size=(20,len(PC)));o=rng.normal(size=x.shape);scale=np.exp(rng.normal(size=len(PC)));centre=rng.normal(size=len(PC));shift=rng.normal(size=len(PC))
    single=(x-o-centre)/scale;translated=(x+shift-(o+shift)-centre)/scale
    np.testing.assert_allclose(single,translated,atol=1e-13)
    np.testing.assert_allclose((o-o)/scale,0,atol=0)
    double=(x-2*o)/scale;assert not np.allclose(single,double)
    assert np.linalg.norm((-centre)/scale)>0
    print('PASS: single reference subtraction is translation invariant; origin-preserving coordinates map a reference to zero; centring differs from subtracting a control twice.')

if __name__=='__main__':test()
