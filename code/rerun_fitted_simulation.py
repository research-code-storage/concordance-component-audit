"""Optional full retraining of the existing 200-draw simulation, in a separate directory."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data/evidence/01_code/10_extension'))
import run_simulation
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'generated/retrained_simulation');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if out==ROOT/'data/evidence':raise SystemExit('Use a separate output directory')
    run_simulation.ROOT=out;(out/'04_results/extension_double').mkdir(parents=True,exist_ok=True)
    run_simulation.run(R=200)
if __name__=='__main__':main()
