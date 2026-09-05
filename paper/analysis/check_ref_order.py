"""Verify the numbered reference list matches first-citation order.

elsarticle-num numbers references in order of first citation. If the .bbl order
diverges from the order the keys first appear in the text, the numbering a
reader sees is wrong, which is what happens when a .bbl is stale.
"""
import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parent.parent


def main():
    tex = (REV / 'Manuscript_R1.tex').read_text(encoding='utf-8')
    bbl = (REV / 'Manuscript_R1.bbl').read_text(encoding='utf-8')

    body = tex.split(r'\begin{document}')[1].split(r'\bibliography')[0]
    order = []
    for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]*)\}', body):
        for k in (x.strip() for x in m.group(1).split(',')):
            if k and k not in order:
                order.append(k)
    listed = re.findall(r'\\bibitem\{([^}]*)\}', bbl)

    print(f'{len(order)} keys cited in text, {len(listed)} entries in the '
          f'reference list')
    ok = True
    if set(order) != set(listed):
        only_text = sorted(set(order) - set(listed))
        only_bbl = sorted(set(listed) - set(order))
        if only_text:
            print(f'  cited but absent from the list: {only_text}')
            ok = False
        if only_bbl:
            print(f'  listed but never cited: {only_bbl}')
            ok = False
    if order != listed:
        for i, (a, b) in enumerate(zip(order, listed), 1):
            if a != b:
                print(f'  ORDER MISMATCH at position {i}: '
                      f'text expects {a}, list has {b}')
                ok = False
                break
    if ok:
        print('reference list is in first-citation order, and every cited key '
              'appears exactly once')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
