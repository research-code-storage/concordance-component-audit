"""Write SHA256 checksums for the distributed inputs, code and documentation."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
def files():
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or rel.parts[0] in {'generated','.git'}:continue
  if any(x in {'__pycache__','.mplconfig','.pytest_cache'} for x in rel.parts):continue
  if p.name=='MANIFEST.sha256' or p.suffix=='.pyc':continue
  yield p
def main():
 lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in files()]
 (ROOT/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(f'Manifest covers {len(lines)} files.')
if __name__=='__main__':main()
