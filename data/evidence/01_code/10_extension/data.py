"""Single data route: names, joins, control IDs and exact frozen transformations."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
CTRL={'dmso','untreated','water'}
MISSING={'REACTOME_CELLULAR_RESPONSES_TO_STIMULI','REACTOME_TRANSPORT_OF_SMALL_MOLECULES'}






def draw_compounds(labels,n,seed):
    rng=np.random.RandomState(seed);pos=labels[labels==1].index.values;neg=labels[labels==0].index.values
    k=round(n*len(pos)/len(labels))
    if k>len(pos)or n-k>len(neg):raise ValueError('Not enough compounds')
    return np.r_[rng.permutation(pos)[:k],rng.permutation(neg)[:n-k]]

def aggregate(labels,compounds,scores,method='mean'):
    df=pd.DataFrame({'label':labels,'compound':compounds,'score':scores})
    if (df.groupby('compound').label.nunique()!=1).any():raise ValueError('Inconsistent labels')
    return df.groupby('compound',sort=True).agg(label=('label','first'),score=('score',method))
