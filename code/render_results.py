"""Write numerical table rows, reported metrics and plotting inputs from reproduced analyses."""
from pathlib import Path
import csv,json,re,shutil
import numpy as np
import build_si_tables as tables
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'generated';EV=ROOT/'data/evidence'

def numeric_rows(text):
 out=[]
 for line in text.splitlines():
  if '&' not in line or not line.rstrip().endswith(r'\\'):continue
  if any(s in line for s in ['\\caption','\\multicolumn','Strength','Coordinates &','Coordinate design','Mean AUC','Coverage','Reference','Quantity','Reversals','All pairs','Dist.','Geometry','RSE']):continue
  v=re.findall(r'[-+]?\d+(?:\.\d+)?',line.replace('$-$','-'))
  if v:out.append(v)
 return out

def j(name):return json.loads((G/name).read_text())
def main():
 out=G/'tables';out.mkdir(exist_ok=True)
 res=j('results.json');lines=[r'\begin{tabular}{llcccc}',r'\toprule','Panel & Design & Pairs & Changed & Flagged & B '+r'\\',r'\midrule']
 for p in ['PHH','external','source']:
  label=tables.PANELS[p]
  for d,dlabel in tables.DES.items():
   r=res[f'60|{p}|{d}__softplus_original']
   lines.append(f"{label} & {dlabel} & {r['n_pairs']} & {r['changed_pairs_min']}--{r['changed_pairs_max']} & {r['vulnerable_min']}--{r['vulnerable_max']} & {r['budget_max']:.4f}"+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}'];(G/'table2.tex').write_text('\n'.join(lines))
 allrows={}
 blocks=re.findall(r'\\begin\{longtable\}[\s\S]*?\\end\{longtable\}',(G/'si_simulation_tables.tex').read_text())
 # S1/S2 are emitted as tabular blocks by the simulation generator.
 if not blocks:blocks=re.findall(r'\\begin\{tabular\}[\s\S]*?\\end\{tabular\}',(G/'si_simulation_tables.tex').read_text())
 assert len(blocks)==2
 for i,b in enumerate(blocks,1):allrows[f'S{i}']=numeric_rows(b)
 for nums,fn in [([3],tables.fitted_table),([4],tables.application_table),([5,6],tables.certificate_tables),([7],tables.rank_table),([9,10,11],tables.coverage_tables),([12,13],tables.refit_tables)]:
  text=fn();bs=re.findall(r'\\begin\{longtable\}[\s\S]*?\\end\{longtable\}',text);assert len(bs)==len(nums)
  for n,b in zip(nums,bs):
   allrows[f'S{n}']=numeric_rows(b.split(r'\endfoot')[-1]);(out/f'Table_S{n}.tex').write_text(b)
 allrows['S8']=numeric_rows((G/'nonlinear/table_S8.tex').read_text())
 allrows['Main2']=numeric_rows((G/'table6.tex').read_text())
 allrows['Main3']=numeric_rows((G/'clinical_summary.tex').read_text())
 cfg=json.loads((EV/'05_reconstruction/results/experiments/config.json').read_text())
 allrows['Main4']=[]
 for domain in ['source','external','PHH']:
  q=cfg['evaluation'][domain];r=res[f'60|{domain}|matched_centred__softplus_original']
  allrows['Main4'].append(list(map(str,[q['profiles'],r['n_compounds'],r['n_positive'],r['n_negative'],r['n_pairs']])))
 for name,rows in allrows.items():
  with (out/f'Table_{name}.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (G/'table_numeric_rows.json').write_text(json.dumps(allrows,indent=2))
 shutil.copy2(G/'clinical_summary.tex',out/'Table_3.tex');shutil.copy2(G/'table6.tex',out/'Table_2.tex')
 shutil.copy2(G/'si_simulation_tables.tex',out/'Tables_S1_S2.tex');shutil.copy2(G/'nonlinear/table_S8.tex',out/'Table_S8.tex')

 plot=G/'plot_data';plot.mkdir(exist_ok=True)
 for name in ['fitted_component_audit.json','results.json']:shutil.copy2(G/name,plot/name)
 shutil.copy2(G/'main_inference/all_results.csv',plot/'all_results.csv')
 for name in ['figure2BC_budget_points.csv','figureS1A_analytic_curves.csv','stable_dilution_60_cells.csv','figureS1B_counterexamples.csv']:shutil.copy2(G/'theory_replay'/name,plot/name)
 (G/'plot_input_manifest.json').write_text(json.dumps({p.name:{'sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest()} for p in plot.iterdir() if p.is_file()},indent=2))
 print('Main Tables 2-4, SI Tables S1-S13 and all figure inputs generated.')

if __name__=='__main__':main()
