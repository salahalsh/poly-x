"""List every sentence that attributes a checkable fact to a cited third party.

The Table 1 audit found three wrong cells, all of them derived from search
summaries rather than fetched sources. This lists the remaining places in the
manuscript where the same mistake could be hiding, so each can be checked or
softened deliberately rather than by accident.
"""
import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parent.parent

# Our own results carry their own provenance (tokens from archived outputs), so
# only claims attributed to somebody else are of interest here.
OURS = re.compile(r'\b(we|our|POLY-X|this work|Section~)\b', re.I)
FACTY = re.compile(
    r'\b('
    r'\d[\d,.]*\s*(?:million|properties|components|polymers|bit|entries|sources|fragments)'
    r'|\d+-D\b|R\$\^2\$|R\^2|\bMAE\b|\bdozens\b|\bfirst\b|\bonly\b|\bno\b\s+\w+\s+(?:reports|provides)'
    r')', re.I)
CITED = re.compile(r'\\cite[a-zA-Z]*\{([^}]*)\}')


def sentences(text):
    text = re.sub(r'\s+', ' ', text)
    return re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', text)


def main():
    tex = (REV / 'Manuscript_R1.tex').read_text(encoding='utf-8')
    body = tex.split(r'\section{Introduction}')[1]
    body = body.split(r'\section*{Declarations}')[0]
    # strip float bodies: Table 1 was audited separately
    body = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', body, flags=re.S)
    body = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', ' ', body, flags=re.S)

    n = 0
    for s in sentences(body):
        keys = CITED.findall(s)
        if not keys:
            continue
        if not FACTY.search(s):
            continue
        flat = ', '.join(sorted({k.strip() for grp in keys
                                 for k in grp.split(',') if k.strip()}))
        mine = bool(OURS.search(s))
        n += 1
        print(f'{n:2d}. [{flat}]{"  (frames it as ours)" if mine else ""}')
        print(f'    {re.sub(r"\\\\citep?\\{[^}]*\\}", "", s).strip()[:260]}')
        print()
    print(f'{n} sentence(s) attribute a checkable fact to a cited source')


if __name__ == '__main__':
    sys.exit(main())
