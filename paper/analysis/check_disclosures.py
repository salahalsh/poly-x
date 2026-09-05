"""
Assert that the Discussion still carries every disclosure that must not be lost.

Restructuring prose is exactly the operation in which an inconvenient sentence
quietly disappears. Each check below corresponds to a result that reflects badly
on the platform and that a referee was told, in the response letter, would be
stated in the Discussion. If any check fails, the rewrite has dropped something
it was not entitled to drop.

Usage:  python check_disclosures.py [path-to-tex]   (default: the built manuscript)
"""
import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parent.parent

# (label, list of regexes - ALL must match somewhere in the Discussion)
REQUIRED = [
    ('ECFP4-vs-polyBERT ranking withdrawn / protocol-dependent',
     [r'(revers|protocol.depend|withdraw)']),
    ('the cluster split contradicts the scaffold split',
     [r'(cluster|leave.clusters.out)']),
    ('conformal intervals under-cover',
     [r'(under.?cover|short(fall|s of)|below (its )?nominal|73\.7|exchangeab)']),
    ('group contribution fails off its calibration domain',
     [r'(coverage|unmatched)']),
    ('the composite reliability index is no better than its best single component',
     [r'(similarit\w+ alone|single component|no better)']),
    ('inter-model disagreement carries no error signal',
     [r'(disagreement|0\.061)']),
    ('machine-learning validation is restricted to Tg',
     [r'(only|restricted|confined).{0,40}\$?T_?g|T_g.{0,30}(only|alone)']),
    ('no experimental confirmation was performed',
     [r'(no polymer was|not synthesis|no synthesis|without experimental|no experimental)']),
]

FORBIDDEN = [
    ('em dash or en dash', r'[\u2013\u2014]'),
    ('"further studies are warranted" boilerplate', r'further (studies|work|research) (are|is) warranted'),
    ('state-of-the-art accuracy claim', r'state[- ]of[- ]the[- ]art (accuracy|performance)(?!.{0,60}not)'),
]


def get_discussion(path: Path) -> str:
    s = path.read_text(encoding='utf-8')
    m = re.search(r'\\section\{Discussion\}(.*?)(?=\\section\{Conclusions\}|%% CONCLUSIONS)',
                  s, re.S)
    if not m:
        raise SystemExit(f'no Discussion section found in {path}')
    return m.group(1)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REV / 'Manuscript_R1.tex'
    if not path.exists():
        path = REV / 'Manuscript_R1.tex.in'
    disc = get_discussion(path)
    low = disc.lower()

    failures = []
    for label, pats in REQUIRED:
        if not all(re.search(p, low, re.I | re.S) for p in pats):
            failures.append(f'MISSING disclosure: {label}')
    for label, pat in FORBIDDEN:
        hits = re.findall(pat, disc, re.I)
        if hits:
            failures.append(f'FORBIDDEN ({label}): {len(hits)} occurrence(s)')

    print(f'Discussion disclosure check on {path.name}')
    if failures:
        print(f'\n{len(failures)} problem(s):')
        for f in failures:
            print('  -', f)
        return 1
    print(f'all {len(REQUIRED)} required disclosures present; '
          f'no forbidden constructions')
    return 0


if __name__ == '__main__':
    sys.exit(main())
