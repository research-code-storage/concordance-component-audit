"""Run the analyses mapped to the manuscript and Supplementary Information."""
from pathlib import Path
import argparse,hashlib,json,os,platform,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
EV='data/evidence'
STAGES={
 'inputs':[('code/check_features.py','--release',EV),('code/audit_data_integrity.py',),('code/verify_prototype_selection.py',)],
 'toxicogenomic':[('code/compute_all.py','--release',EV),('code/replay_checks.py','main')],
 'permutation':[('code/permutation_contrast.py','--release',EV)],
 'clinical':[('code/replay_checks.py','survival'),('code/survival_example.py','--study',EV),('code/lung_bootstrap.py','--study',EV),('code/lung_ustat.py','--study',EV),('code/survival_refit.py','--study',EV),('code/clinical_summary.py',)],
 'simulation':[('code/replay_checks.py','simulation'),('code/simulation_usefulness.py','--study',EV)],
 'frozen_reference':[('code/frozen_reference.py','--study',EV)],
 'fitted_simulation':[('code/replay_checks.py','fitted_sim')],
 'theory':[(EV+'/05_reconstruction/code/reproduce_theory_evidence.py','--output','generated/theory_replay')],
 'inference':[('code/verify_inference.py',)],
 'nonlinear':[('b_robustness/run.py','--out','generated/nonlinear'),('b_robustness/verify.py','--results','generated/nonlinear')],
 'report':[('code/render_results.py',),('code/verify_reported.py',),('code/render_figures.py',)]}

def run(job):
 env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 logs=ROOT/'generated/logs';logs.mkdir(parents=True,exist_ok=True)
 tag=Path(job[0]).stem+('_'+job[1] if len(job)>1 and not job[1].startswith('--') else '')
 start=time.monotonic()
 with (logs/(tag+'.log')).open('w',encoding='utf-8') as f:
  process=subprocess.run([sys.executable,*job],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 text=(logs/(tag+'.log')).read_text(encoding='utf-8',errors='replace')
 for line in text.splitlines()[-5:]:print('  '+line,flush=True)
 if process.returncode:raise SystemExit(f'Failed: {job[0]}. See generated/logs/{tag}.log')
 print(f'  completed in {time.monotonic()-start:.1f} s',flush=True)

def manifest():
 sys.path.insert(0,str(ROOT));from build_manifest import files
 recorded={}
 for line in (ROOT/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
  digest,rel=line.split('  ',1);p=(ROOT/rel).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file():raise ValueError('Invalid manifest path: '+rel)
  if rel in recorded:raise ValueError('Duplicate manifest entry: '+rel)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('File checksum mismatch: '+rel)
  recorded[rel]=digest
 assert set(recorded)=={p.relative_to(ROOT).as_posix() for p in files()},'Manifest file inventory differs'
 print(f'Integrity: {len(recorded)} files verified.',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=list(STAGES),help='Run one stage; report requires earlier stages.')
 p.add_argument('--manifest-only',action='store_true')
 p.add_argument('--skip-manifest',action='store_true',help='For preparing the release before its manifest is generated.')
 p.add_argument('--deep',action='store_true',help='Also replay model states, coordinate assets and small-problem gradient checks.')
 p.add_argument('--full-inference',action='store_true',help='Regenerate all 160000 coverage datasets and 8000 bootstrap intervals.')
 a=p.parse_args();os.chdir(ROOT)
 if a.full_inference and a.stage not in (None,'inference'):
  p.error('--full-inference requires the full workflow or --stage inference')
 if not a.skip_manifest:manifest()
 if a.manifest_only:return
 selected=[a.stage] if a.stage else list(STAGES)
 report={'status':'RUNNING','requested_stages':selected,'completed_stages':[],
         'deep_requested':a.deep,'deep_completed':False,'full_inference':a.full_inference,
         'python':platform.python_version(),'platform':platform.platform()}
 status=ROOT/'generated/reproduction.json';status.parent.mkdir(exist_ok=True)
 def save_status():status.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 save_status();start=time.monotonic()
 try:
  for i,name in enumerate(selected,1):
   print(f'[{i}/{len(selected)}] {name}',flush=True)
   jobs=STAGES[name]
   if name=='inference' and a.full_inference:
    jobs=[('code/inference_coverage.py','--out','generated/inference_full','--replicates','2000',
           '--bootstrap-replicates','1000','--bootstrap-resamples','999','--truth-power','20','--truth-blocks','16','--workers','4'),
          ('code/verify_inference.py','--full-replay','generated/inference_full')]
   for job in jobs:run(job)
   report['completed_stages'].append(name);save_status()
  if a.deep:
   os.environ['CONCORDANCE_VERIFY_OUTPUT']=str(ROOT/'generated/small_fit_checks')
   for job in [('code/verify_states.py',),
               (EV+'/05_reconstruction/code/check_small_fits.py',),
               (EV+'/07_reference_study/code/build_reference_assets.py','--output','generated/reference_assets')]:run(job)
   report['deep_completed']=True
  report['status']='PASS'
 except BaseException:
  report['status']='FAILED'
  raise
 finally:
  report['elapsed_seconds']=round(time.monotonic()-start,2);save_status()
 print('Selected analyses completed. Results and numerical checks are in generated/.',flush=True)

if __name__=='__main__':main()
