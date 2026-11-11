"""Independently verify folded-perturbation witnesses, pairwise envelopes and SI Table S8."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def scalar_H(x):
    if x > 0:
        return 1.0
    if x < 0:
        return 0.0
    return 0.5


def independent(ell, g, y, t, h):
    changes, budgets, endpoint_plus, endpoint_minus = [], [], [], []
    for i in range(len(y)):
        if y[i] != 1:
            continue
        for j in range(len(y)):
            if y[j] != 0:
                continue
            D, E = ell[i] - ell[j], t * (g[i] - g[j])
            q = scalar_H(D)
            p = abs(scalar_H(D + E) - q)
            m = abs(scalar_H(D - E) - q)
            budgets.append(max(p, m))
            endpoint_plus.append(p)
            endpoint_minus.append(m)
            changes.append(scalar_H(D + (h[i] - h[j])) - q)
    n = len(changes)
    return {'A': math.fsum(map(abs, changes))/n, 'Delta': math.fsum(changes)/n,
            'B': math.fsum(budgets)/n, 'plus': math.fsum(endpoint_plus)/n,
            'minus': math.fsum(endpoint_minus)/n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=REPO/'generated/nonlinear')
    a = p.parse_args()
    data = json.loads((a.results/'results.json').read_text(encoding='utf-8'))
    cfg = json.loads((HERE/'parameters.json').read_text(encoding='utf-8'))
    tol = cfg['validation_tolerance']
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(HERE/'parameters.json') == data['parameters_sha256']
    for src in data['provenance']:
        assert digest(REPO/src['path']) == src['sha256']
    snapshots = json.loads((a.results/'compound_scores.json').read_text(encoding='utf-8'))
    by_cell = {(x['panel'], x['seed']): x['subjects'] for x in snapshots}
    # Re-aggregate all raw profile scores with scalar summation, independently of run.py.
    for panel, seed in by_cell:
        path = REPO/f'data/evidence/05_reconstruction/results/experiments/n60_seed{seed}_{panel}_scores.npz'
        with np.load(path, allow_pickle=False) as z:
            col = list(z['arm_names']).index(cfg['arm'])
            for subject in by_cell[panel, seed]:
                ids = [i for i, c in enumerate(z['compound']) if c == subject['compound']]
                assert len(ids) == subject['profiles']
                assert all(int(z['label'][i]) == subject['label'] for i in ids)
                for name in ['linear','kernel','full']:
                    mean = math.fsum(float(z[name][i, col]) for i in ids)/len(ids)
                    assert abs(mean-subject[name]) < tol
    # Validate every endpoint and selected worst-movement witness by nested scalar pairs.
    assert len(data['records']) == 120
    for r in data['records']:
        rows = by_cell[r['panel'], r['seed']]
        ell = [x['linear'] for x in rows]
        g = [x['kernel'] for x in rows]
        y = [x['label'] for x in rows]
        witness = r['grid_max_A_witness']
        h = [witness['sign']*r['t']*abs(x-witness['c']) for x in g]
        check = independent(ell, g, y, r['t'], h)
        for key, expected in [('A',r['F']), ('Delta',witness['Delta']), ('B',r['B']),
                              ('plus',r['plus']['A']), ('minus',r['minus']['A'])]:
            assert abs(check[key]-expected) < tol, (r['panel'],r['seed'],r['t'],key)
        for i in range(len(g)):
            for j in range(len(g)):
                assert abs(h[i]-h[j]) <= r['t']*abs(g[i]-g[j])+tol
    # Exhaustive scalar tie cases: compare sum-minus-correction with the max definition.
    for D in range(-4,5):
        for E in range(-4,5):
            q=scalar_H(D)
            ap, am = abs(scalar_H(D+E)-q), abs(scalar_H(D-E)-q)
            T=.5 if D==0 and E!=0 else 0
            assert ap+am-T == max(ap,am)
    with (a.results/'grid_results.csv').open(encoding='utf-8',newline='') as f:
        grid=list(csv.DictReader(f))
    assert len(grid)==24240
    indexed={}
    for row in grid:
        indexed.setdefault((row['panel'],int(row['seed']),int(row['t'])),[]).append(row)
    for r in data['records']:
        rows=indexed[r['panel'],r['seed'],r['t']]
        assert len(rows)==202
        assert max(float(x['A']) for x in rows)==r['F']
        assert max(abs(float(x['Delta'])) for x in rows)==r['grid_max_abs_Delta']
        assert all(float(x['A'])<=r['B']+tol for x in rows)
    primary={x['panel']:x for x in data['summary'] if x['t']==1}
    assert [primary[p]['F_exceeds_M'] for p in ['PHH','external','source']]==[0,2,4]
    assert [primary[p]['abs_Delta_exceeds_M'] for p in ['PHH','external','source']]==[0,0,1]
    source=next(r for r in data['records'] if (r['panel'],r['seed'],r['t'])==('source',9,1))
    external=next(r for r in data['records'] if (r['panel'],r['seed'],r['t'])==('external',0,1))
    assert source['plus']['A']==source['minus']['A']==4/96
    assert source['F']==6/96 and source['grid_max_A_witness']['Delta']==-6/96 and source['B']==8/96
    assert external['F']==external['B']==3/528 and external['M']==2/528
    assert external['grid_max_A_witness']['Delta']==-1/528
    toy=cfg['toy'];ell=toy['positive_ell']+toy['negative_ell'];g=toy['positive_g']+toy['negative_g']
    check=independent(ell,g,[1]*10+[0]*10,1,[-abs(x) for x in g])
    assert check=={'A':.04,'Delta':-.04,'B':.04,'plus':.02,'minus':.02}
    from run import table_tex
    expected=table_tex(data['summary'])
    assert (a.results/'table_S8.tex').read_text(encoding='utf-8')==expected
    import re
    numeric_rows=[]
    for line in expected.splitlines():
        if '&' in line and line.rstrip().endswith(chr(92)*2):
            values=re.findall(r'[-+]?\d+(?:\.\d+)?',line.replace('$-$','-'))
            if values:numeric_rows.append(values)
    published=json.loads((REPO/'reported/table_numeric_rows.json').read_text())['S8']
    assert numeric_rows==published
    report={'status':'PASS','source_hashes':30,'raw_compound_aggregations':len(snapshots),
            'independent_endpoint_and_witness_cells':120,'scalar_tie_cases':81,
            'grid_records_checked':24240,'published_table_numeric_rows_match':True,
            'toy_exact_expected_values':True,'primary_counts_and_examples_match':True}
    (a.results/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
