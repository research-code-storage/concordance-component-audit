"""Compare reproduced tables and key results with the published numerical targets."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'generated'
def lookup(value,path):
 for key in path:value=value[int(key)] if isinstance(value,list) else value[key]
 return value
def main():
 expected=json.loads((ROOT/'reported/table_numeric_rows.json').read_text())
 actual=json.loads((G/'table_numeric_rows.json').read_text())
 assert set(expected)==set(actual)
 for name in expected:assert expected[name]==actual[name],f'Table mismatch: {name}'
 targets=json.loads((ROOT/'reported/key_results.json').read_text());records=[]
 for r in targets:
  value=lookup(json.loads((ROOT/r['output']).read_text()),r['field'])
  np.testing.assert_allclose(value,r['expected'],rtol=1e-10,atol=1e-12,err_msg=r['id'])
  records.append({'id':r['id'],'location':r['location'],'value':value,'reported':r['reported'],'output':r['output'],'field':'/'.join(map(str,r['field']))})
 with (G/'key_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 report={'status':'PASS','numerical_tables':len(expected),'table_rows':sum(map(len,expected.values())),
         'key_results':len(records),'targets':'reported/','outputs':'generated/tables/ and generated/key_results.csv'}
 (G/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
