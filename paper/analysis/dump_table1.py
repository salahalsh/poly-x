"""Dump every factual assertion Table 1 makes, one per line, for verification."""
import re
from pathlib import Path

REV = Path(__file__).resolve().parent.parent
tex = (REV / 'Manuscript_R1.tex').read_text(encoding='utf-8')
m = re.search(r'\\label\{tbl:platforms\}(.*?)\\end\{table\}', tex, re.S)
block = m.group(1)

COLS = ['Polymer Genome', 'PolyID', 'polyBERT', 'PolyMetriX', 'POLY-X']


def clean(c):
    c = re.sub(r'\\(?:textbf|emph|citet|citep)\{([^{}]*)\}', r'\1', c)
    c = re.sub(r'\\[a-zA-Z]+\*?', '', c)
    return c.replace('{', '').replace('}', '').replace('$', '').strip()


rows = []
for line in block.splitlines():
    if '&' not in line or line.strip().startswith('%'):
        continue
    cells = [clean(c) for c in line.replace(r'\\', '').split('&')]
    if len(cells) < 5:
        continue
    rows.append(cells)

n = 0
for r in rows:
    label = r[0]
    if not label or label.lower() in ('', 'reference'):
        pass
    for j, col in enumerate(COLS, start=1):
        if j >= len(r):
            continue
        val = r[j]
        if not val or val in ('n/a', '--'):
            continue
        if col == 'POLY-X':
            continue          # our own claims, checked against our own outputs
        n += 1
        print(f'{n:2d}. [{col}] {label}: {val}')
print(f'\n{n} third-party assertions in Table 1')
