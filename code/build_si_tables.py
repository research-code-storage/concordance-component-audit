"""Render final SI tables directly from identified analysis records."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from decimal_reporting import rounded
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/'data/evidence'
PANELS={'source':'Within-source holdout','external':'DILImap','PHH':'PHH'}
DES={'historical_centred':'Pooled, centred','matched_centred':'Matched, centred','revised_origin':'Matched, origin preserving'}
ARM={'softplus_original':'Full predictor','fixed_sum':'Fixed sum','kernel_original':'Kernel only'}

def longtable(number,caption,columns,header,rows):
 head=header+r' \\'
 return '\n'.join([r'\ReserveTableSpace',r'\noindent\textbf{Table S'+str(number)+':} '+caption,r'\begin{small}',r'\setlength{\tabcolsep}{'+('2' if number in (4,12) else '3')+'pt}',r'\begin{longtable}{'+columns+'}',r'\toprule',head,r'\midrule',r'\endfirsthead',r'\toprule',head,r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot',*rows,r'\end{longtable}',r'\end{small}'])

def application_table():
 records=json.loads((EV/'05_reconstruction/results/experiments/summary.json').read_text())['cells'];rows=[]
 for dom,name in PANELS.items():
  for d,dname in DES.items():
   for arm,aname in ARM.items():
    hits=[r for r in records if r['training_compounds']==60 and r['domain']==dom and r['arm']==d+'__'+arm];assert len(hits)==1;r=hits[0]
    lo,hi=r['compound_bootstrap_ci'];rows.append(f"{name} & {dname} & {aname} & {r['auc_mean']:.4f} & [{lo:.4f}, {hi:.4f}]"+r' \\')
 assert len(rows)==27
 return longtable(4,'Discrimination of the prediction scores used in the component analysis, with 60 training compounds. All evaluation panels and reference designs are shown; intervals are conditional compound-bootstrap 95\\% intervals.',r'L{.15\linewidth}L{.27\linewidth}L{.17\linewidth}C{.11\linewidth}C{.21\linewidth}',r'Panel & Coordinates & Score & Mean AUC & 95\% CI',rows)

def rank_table():
 records=list(csv.DictReader((ROOT/'generated/theory_replay/complete_rank_diagnostics.csv').open()));rows=[]
 for dom,name in PANELS.items():
  fixed=[int(r['fixed_sum_all_pair_disagreements']) for r in records if r['domain']==dom];assert len(fixed)==10;counts=[]
  for seed in range(10):
   z=np.load(EV/f'05_reconstruction/results/experiments/n60_seed{seed}_{dom}_scores.npz');j=list(z['arm_names']).index('matched_centred__softplus_original')
   ids,inv,n=np.unique(z['compound'],return_inverse=True,return_counts=True);full=np.bincount(inv,z['full'][:,j])/n;lin=np.bincount(inv,z['linear'][:,j])/n
   a,b=np.triu_indices(len(ids),1);counts.append(int(np.sum(np.sign(full[a]-full[b])!=np.sign(lin[a]-lin[b]))))
  rows.append(f'{name} & {len(a)} & {min(fixed)}--{max(fixed)} & {min(counts)}--{max(counts)}'+r' \\')
 return longtable(7,'Complete-rank differences from the matched-centred full predictor, over all unordered compound pairs and ten source draws. Fixed sum uses the unfitted coordinate sum; kernel deletion retains the fitted linear coefficients.','lccc','Panel & All pairs & Fixed sum & Kernel deletion',rows)

def generated_rows(name):
 """Data rows of a generated publication table (generated/<name>)."""
 lines=[x.strip() for x in (ROOT/'generated'/name).read_text(encoding='utf8').splitlines()]
 a=lines.index(r'\midrule');b=lines.index(r'\bottomrule')
 rows=[r'\addlinespace' if x==r'\midrule' else x for x in lines[a+1:b]]
 assert rows and all(x.endswith(r'\\') or x==r'\addlinespace' for x in rows),name
 return rows

def certificate_tables():
 text=longtable(5,'Kernel-deletion bounds for the original full-predictor fits with 60 training compounds, for every evaluation panel and coordinate design. Ranges span ten source draws. Flagged-pair counts use the inclusive screen defining $V_\\omega$; the maximum bound uses the tie-aware form $B_\\omega$ of main-text Proposition~1 on the observed pairs.','llcccc','Panel & Coordinate design & Pairs & Actual changes & Flagged pairs & Maximum $B_\omega$',generated_rows('table2.tex'))
 text+='\n\n'+longtable(6,'Component-score permutation and the deletion bound on the same fitted states, for the original full predictor with 60 training compounds. $B_\\omega$ and \\emph{changed} are the tie-aware bound of main-text Proposition~1 and the pairs whose contribution actually changes on deletion, both maxima over the ten source draws. \\emph{Deletion} is the mean net concordance change $\\Delta$, calculated as full minus retained concordance. \\emph{Permutation} is the mean drop in concordance when the kernel component is permuted across compounds, over 1000 permutations per draw, and the last column is the fraction of permutations that change concordance at all.','llccccc','Panel & Coordinate design & $B_\omega$ & Changed & Deletion & Permutation & Nonzero',generated_rows('table7.tex'))
 return text

def refit_tables():
 r=json.loads((ROOT/'generated/survival_refit.json').read_text());sv=json.loads((ROOT/'generated/survival_example.json').read_text())
 assert r['n_comparable_pairs']==sv['n_comparable_pairs']==1653
 sgn=lambda x:('$-$' if x<0 else '')+f'{abs(x):.3f}'
 rows=[]
 for key,label in (('delete_age_sex','Age and sex'),('delete_ecog','ECOG')):
  d=r[key];s=sv[key];names=d['kept_features']
  for k in ('concordance','net_change','movement'):
   assert abs(d[k+'_same_fit']-d[k+'_refit'])<1e-12 if d['same_fit_versus_refit_changed_pairs']==0 else True
  assert abs(d['net_change_same_fit']-s['net_change'])<1e-12 and abs(d['movement_same_fit']-s['mean_absolute_pair_change'])<1e-12
  coef=lambda c:', '.join(('ECOG' if n=='ph.ecog' else n)+' '+sgn(v) for n,v in zip(names,c))
  if rows:rows.append(r'\addlinespace')
  rows.append(f"{label} & Same fit & {coef(d['same_fit_coefficients'])} & {d['concordance_same_fit']:.4f} & {d['net_change_same_fit']:.4f} & {d['movement_same_fit']:.4f} & {s['strict_reversals']} & {s['tie_transitions']} & -- "+r'\\')
  rows.append(f" & Refit & {coef(d['refit_coefficients'])} & {d['concordance_refit']:.4f} & {d['net_change_refit']:.4f} & {d['movement_refit']:.4f} & {d['strict_reversals_refit']} & {d['tie_transitions_refit']} & {d['same_fit_versus_refit_changed_pairs']} "+r'\\')
 text=longtable(12,'Same-fit and refitted reduced Cox models: 1653 comparable pairs from 69 held-out patients. Coefficients use training-standardized covariates. $C$ is reduced-score concordance; $\\Delta$ and $A_\\omega$ compare it with the full score. Reversals and ties count strict sign changes and tie transitions. \\emph{Changed} compares the two reduced-score orderings. $B_\\omega$ is reported for the same-fit decomposition.',r'llL{.22\linewidth}rrrrrr',r'Deleted & Reduced model & Coefficients & $C$ & $\Delta$ & $A_\omega$ & Reversals & Ties & Changed',rows)
 b=r['bootstrap'];iv=lambda o:'['+sgn(o['lower'])+', '+sgn(o['upper'])+']'
 rows=[]
 for key,label in (('delete_age_sex','Age and sex'),('delete_ecog','ECOG')):
  t=r[key]['training_bootstrap']
  if rows:rows.append(r'\addlinespace')
  rows.append(f"{label} & $\\Delta$ & {iv(t['net_change_same_fit'])} & {iv(t['net_change_refit'])} "+r'\\')
  rows.append(f" & $A_\\omega$ & {iv(t['movement_same_fit'])} & {iv(t['movement_refit'])} "+r'\\')
  rows.append(f" & $B_\\omega$ & {iv(t['budget_same_fit'])} & -- "+r'\\')
  rows.append(f" & $A_\\omega-|\\Delta|$ & {iv(t['movement_minus_absolute_net_same_fit'])} & -- "+r'\\')
 text+='\n\n'+longtable(13,f"Training-sample bootstrap for the lung-cancer illustration: {b['replicates']} resamples of the 158 training patients (seed {b['seed']}), refitting the full and both reduced Cox models in every replicate and evaluating on the fixed 1653 held-out comparable pairs. Entries are 95\\% percentile intervals. The same-fit column deletes the block from each refitted full model; the refit column compares each refitted full model with the reduced model refitted on the same resample. The intervals describe model-fitting variability conditional on the held-out patients.",'llcc',r'Deleted & Quantity & Same fit & Refit',rows)
 return text

def fitted_table():
 cells=json.loads((ROOT/'generated/fitted_component_audit.json').read_text())['cells'];rows=[]
 for c in cells:
  v=['Informative' if c['informative'] else 'Noise',str(c['shift']),f"{c['initial']:.3f}"]+[f"{c[k+'_mean']:.4f} ({c[k+'_mcse']:.4f})" for k in ('auc','auc_difference','A','B')]
  rows.append(' & '.join(v)+r' \\')
 return longtable(3,r'All fitted-model simulation cells. Entries are means (Monte Carlo standard errors) over 200 independent training draws, conditional on one evaluation sample. Displacement cells reuse fits. $\Delta$, $A_\omega$ and $B_\omega$ all refer to same-fit kernel deletion.','lccrrrr',r'Coordinates & Shift & Start & AUC & $\Delta$ & $A_\omega$ & $B_\omega$',rows)

def coverage_tables():
 cells=json.loads((EV/'08_inference/summary.json').read_text())['cells'];rows=[]
 for c in cells[:72]:
  q=c['scenario'];m=c['metrics'];v=[str(q['n']),f"{q['a']:g}",str(q['s']),{'normal':'N','t3':r'$t_3$','rounded_normal':'R'}[q['distribution']]]
  for k in ('Delta','A','B'):v +=[rounded(m[k]['normal_coverage']),f"{m[k]['mean_se_over_sd']:.3f}"]
  v +=['/'.join(str(m[k]['unavailable_count']) for k in ('Delta','A','B'))];rows.append(' & '.join(v)+r' \\')
 text=longtable(9,r'Binary coverage study: 2000 datasets per row. C is unconditional coverage of nominal 95\% normal intervals; RSE is mean estimated standard error divided by empirical SD. U counts unavailable intervals ($\Delta/A/B$) out of 2000; these count as noncoverage. N, $t_3$ and R denote normal, standardized $t_3$ and rounded-normal scores. Reference values and replicate estimates are in the reproducibility archive.','rrrrrrrrrrr',r'$n$ & $a$ & $s$ & Dist. & C$_\Delta$ & RSE$_\Delta$ & C$_A$ & RSE$_A$ & C$_B$ & RSE$_B$ & U',rows)
 rows=[]
 for c in cells[72:]:
  q=c['scenario']
  for j,k in enumerate(('Delta','A','B')):
   m=c['metrics'][k];sym=r'$\Delta$' if k=='Delta' else '$'+k+'$'
   rows.append(f"{q['n']} & {q['a']:g} & {100*q['censor_fraction']:.0f}\\% & {sym} & {c['truth']['value'][j]:.5f} & {rounded(m['normal_coverage'])} & {m['mean_se_over_sd']:.3f}"+r' \\')
 text+='\n\n'+longtable(10,r'Survival coverage study: 2000 datasets per scenario. Censoring is the expected proportion. Each reference is the observed-pair estimand, including the censoring distribution. All normal intervals were available.','rrrrrrr',r'$n$ & $a$ & Censoring & Quantity & Reference & Coverage & RSE',rows)
 rows=[]
 for c in cells:
  q=c['scenario']
  if not q['bootstrap']:continue
  model=({'normal':'Binary N','rounded_normal':'Binary R'}[q['distribution']] if q['kind']=='binary' else 'Survival')
  for k in ('Delta','A','B'):
   m=c['metrics'][k]['bootstrap'];sym=r'$\Delta$' if k=='Delta' else '$'+k+'$'
   rows.append(f"{model} & {q['n']} & {q['a']:g} & {sym} & {rounded(m['normal_coverage_same_replicates'])} & {rounded(m['percentile_coverage'])} & {m['normal_unavailable_count']} & {m['zero_width_count']}"+r' \\')
 text+='\n\n'+longtable(11,r'Paired interval comparison on the same first 1000 datasets of eight design-defined scenarios, with 999 resamples each. Binary scenarios have $s=0$; survival scenarios have 60\% expected censoring. Normal and percentile columns report coverage. U counts unavailable normal intervals; Z counts zero-width percentile intervals, both out of 1000.','lrrrrrrr',r'Model & $n$ & $a$ & Quantity & Normal & Percentile & U & Z',rows)
 return text



if __name__=='__main__':
    out=ROOT/'generated/tables';out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('S3',fitted_table),('S4',application_table),('S5_S6',certificate_tables),('S7',rank_table),('S9_S10_S11',coverage_tables),('S12_S13',refit_tables)]:
        (out/(name+'.tex')).write_text(fn(),encoding='utf-8')
    print('Numerical SI tables written to generated/tables/.')
