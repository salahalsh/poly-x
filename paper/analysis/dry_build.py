"""Dry build: substitute placeholder values for every token and compile.

Catches LaTeX structure errors (column counts, unbalanced braces, missing
packages) before the real analysis outputs are available, so the final build is
a substitution rather than a debugging session. Writes to a scratch directory
and never touches the real Manuscript_R1.tex / response_to_reviewers.tex.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from polyx_rev import REVISION

SCRATCH = REVISION / '_dryrun'


def placeholder(name: str) -> str:
    if name.startswith('TABLE_'):
        return (r'\begin{table}[htbp]\caption{placeholder}'
                r'\label{tbl:%s}\centering\begin{tabular}{ll}\toprule '
                r'a & b \\ \bottomrule\end{tabular}\end{table}'
                % name.lower().replace('table_', ''))
    if name.startswith(('TBLNO_', 'FIGNO_')):
        return '9'
    if 'VERDICT' in name:
        return 'Placeholder verdict sentence.'
    if name.endswith('_CI'):
        return '[-0.001, 0.001]'
    if name == 'PRI_IQR':
        return '0.00--0.00'
    if name == 'CASE_WINDOW':
        return '450, 500'
    if 'CALIBRATION_SET' in name:
        return 'PE, PP, PS'
    if name.startswith('CONFORMAL_COVERAGE'):
        return r'0.0\%'
    if name.startswith('CASE_') and name.endswith('P25'):
        return r'0.0\%'
    if name == 'CASE_PRI_TEST':
        return 'Mann--Whitney $p = 0.000$'
    return '0.00'


def main():
    SCRATCH.mkdir(exist_ok=True)
    for asset in ('references.bib', 'elsarticle-num.bst',
                  'fig1_architecture.png'):
        src = REVISION / asset
        if src.exists():
            shutil.copy(src, SCRATCH / asset)
    for fig in (REVISION / 'figures').glob('*.png'):
        shutil.copy(fig, SCRATCH / fig.name)
    # Any figure not yet generated: substitute the architecture image so the
    # compile exercises the float layout rather than failing on a missing file.
    placeholder_img = SCRATCH / 'fig1_architecture.png'
    for need in ('fig2_gc_validation.png', 'fig3_ml_parity.png',
                 'fig4_tier_comparison.png', 'fig5_enhanced_features.png',
                 'fig6_representation_benchmark.png', 'fig7_invariance.png'):
        if not (SCRATCH / need).exists() and placeholder_img.exists():
            shutil.copy(placeholder_img, SCRATCH / need)

    rc = 0
    for stem in ('Manuscript_R1', 'response_to_reviewers'):
        src = REVISION / f'{stem}.tex.in'
        if not src.exists():
            continue
        text = src.read_text(encoding='utf-8')
        out = re.sub(r'@@([A-Z0-9_]+)@@',
                     lambda m: placeholder(m.group(1)), text)
        (SCRATCH / f'{stem}.tex').write_text(out, encoding='utf-8')

        for i in range(2):
            r = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', f'{stem}.tex'],
                cwd=SCRATCH, capture_output=True, text=True)
        log = (SCRATCH / f'{stem}.log').read_text(encoding='utf-8',
                                                  errors='replace')
        errors = [l for l in log.splitlines()
                  if l.startswith('!') or 'Fatal error' in l]
        pdf = SCRATCH / f'{stem}.pdf'
        if errors or not pdf.exists():
            rc = 1
            print(f'\n=== {stem}: FAILED ===')
            for e in errors[:15]:
                print('  ', e)
            for l in log.splitlines():
                if l.startswith('l.'):
                    print('   at', l[:110])
        else:
            pages = len(re.findall(rb'/Type\s*/Page[^s]', pdf.read_bytes()))
            print(f'=== {stem}: OK ({pages} pages) ===')
        undefined = sorted(set(re.findall(
            r"Reference `([^']+)' on page", log)))
        if undefined:
            print(f'   undefined references: {undefined[:10]}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
