"""Cross-check citations in the revised manuscript against references.bib."""
import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parent.parent

src = (REV / 'Manuscript_R1.tex.in').read_text(encoding='utf-8')
bibtext = (REV / 'references.bib').read_text(encoding='utf-8')

cited = set()
for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]*)\}', src):
    cited |= {c.strip() for c in m.group(1).split(',') if c.strip()}
bib = set(re.findall(r'@\w+\{([^,]+),', bibtext))

missing = sorted(cited - bib)
orphan = sorted(bib - cited)
print(f'{len(cited)} keys cited, {len(bib)} entries in references.bib')
print(f'\nCITED BUT MISSING FROM BIB ({len(missing)}):')
for k in missing:
    print('  ', k)
print(f'\nIN BIB BUT NO LONGER CITED ({len(orphan)}):')
for k in orphan:
    title = re.search(rf'@\w+\{{{re.escape(k)},.*?title\s*=\s*\{{(.*?)\}},',
                      bibtext, re.S)
    t = ' '.join(title.group(1).split())[:70] if title else ''
    print(f'   {k:22s} {t}')
sys.exit(1 if missing else 0)
