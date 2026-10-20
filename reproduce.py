#!/usr/bin/env python3
"""Offline reanalysis only. Original acquisition scripts are never executed."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    try:
        subprocess.run([sys.executable,str(ROOT/'analysis/reproduce.py'),'--root',str(ROOT)],check=True)
        from analysis.documentation import build,reconstruct_additional
        build(ROOT)
        reconstruct_additional(ROOT)
        from analysis.supplementary import build
        build(ROOT)
        if '--no-figures' not in sys.argv:
            from analysis.figures import make_figures,make_cost_figures
            make_figures(ROOT)
            make_cost_figures(ROOT)
    except Exception:
        (ROOT/'results').mkdir(exist_ok=True)
        (ROOT/'results/validation.json').write_text(json.dumps({'status':'failed','instruction':'See command error; existing output files may be stale.'})+'\n')
        raise
    print('Offline reproduction completed. Inspect results/validation.json and docs/OPEN_ITEMS.md.')
