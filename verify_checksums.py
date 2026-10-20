#!/usr/bin/env python3
"""Verify the exact distributed files listed in SHA256SUMS.txt (offline)."""
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parent
bad=[];n=0
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split('  ',1);p=root/name;n+=1
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(name)
if bad:
    print('Missing or changed files:',*bad,sep='\n');sys.exit(1)
print(f'PASS: {n} distributed file checksums match. Generated files may change after rerunning.')
