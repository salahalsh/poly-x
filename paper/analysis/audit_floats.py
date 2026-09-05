"""Audit the manuscript's tables and figures against the presentation rules.

Checks, for the resolved manuscript (tables substituted in):
  1. every float is cited in the text at least once;
  2. the first citation appears BEFORE the float itself;
  3. no citation uses a weak pointer phrasing ("are presented in Table 1",
     "As shown in Table 2", "the table below", "see below");
  4. table captions precede their tabular, figure captions follow the graphic;
  5. every caption defines the abbreviations it uses.

Exit code 1 if any check fails.
"""
import json
import re
import sys
from pathlib import Path

from polyx_rev import OUT, REVISION

WEAK = [
    r'(?:are|is)\s+(?:presented|given|shown|reported|listed|summari[sz]ed)\s+in\s+(?:Table|Figure)~?\\ref',
    r'As\s+(?:shown|can be seen)\s+in\s+(?:Table|Figure)',
    r'[Tt]he\s+(?:table|figure)\s+(?:below|above)',
    r'\bsee\s+below\b',
    r'^(?:Table|Figure)~\\ref\{[^}]+\}\s+(?:compares|gives|reports|evaluates|shows|summari[sz]es|lists|presents)\b',
]

ABBREV = ['MAE', 'RMSE', 'RF', 'GB', 'AD', 'PRI', 'ECFP4', 'ECFP6', 'CLS',
          'SD', 'IQR', 'GC', 'ML', 'PSMILES', 'AE']


def main():
    tokens_path = OUT / 's10_tokens.json'
    src = (REVISION / 'Manuscript_R1.tex.in').read_text(encoding='utf-8')

    # Resolve TABLE_ tokens so captions and labels are visible in position.
    built = REVISION / 'Manuscript_R1.tex'
    text = built.read_text(encoding='utf-8') if built.exists() else src

    problems = []

    # ---- 1 & 2: citation exists and precedes the float --------------------
    for kind in ('tbl', 'fig'):
        for m in re.finditer(r'\\label\{' + kind + r':([A-Za-z0-9_]+)\}', text):
            name = m.group(1)
            label_pos = m.start()
            refs = [r.start() for r in
                    re.finditer(r'\\ref\{' + kind + ':' + re.escape(name) + r'\}',
                                text)]
            if not refs:
                problems.append(f'{kind}:{name} is never cited in the text')
                continue
            if min(refs) > label_pos:
                problems.append(
                    f'{kind}:{name} is first cited AFTER it appears '
                    f'(float at {label_pos}, first citation at {min(refs)})')

    # ---- 3: weak citation phrasing ----------------------------------------
    body = text.split('\\section{Introduction}', 1)[-1]
    body = body.split('\\section*{Declarations}', 1)[0]
    for pat in WEAK:
        for m in re.finditer(pat, body, re.M):
            line = body[max(0, m.start() - 90):m.start() + 110].replace('\n', ' ')
            problems.append(f'weak float citation: ...{line}...')

    # ---- 4: caption placement --------------------------------------------
    for m in re.finditer(r'\\begin\{figure\}(.*?)\\end\{figure\}', text, re.S):
        block = m.group(1)
        gi, ci = block.find('includegraphics'), block.find('\\caption')
        if gi >= 0 and ci >= 0 and ci < gi:
            problems.append('a figure caption precedes its graphic '
                            '(captions must sit below figures)')
    for m in re.finditer(r'\\begin\{table\}(.*?)\\end\{table\}', text, re.S):
        block = m.group(1)
        ci, ti = block.find('\\caption'), block.find('\\begin{tabular}')
        if ci >= 0 and ti >= 0 and ci > ti:
            problems.append('a table caption follows its tabular '
                            '(titles must sit above tables)')

    # ---- 5: abbreviations defined in captions -----------------------------
    # Captions contain nested braces (\texttt{...}, $...$), so the caption is
    # extracted by brace matching within each float block rather than by a
    # non-greedy regex, which would run past the caption's first closing brace.
    def caption_of(block: str) -> str:
        i = block.find('\\caption{')
        if i < 0:
            return ''
        depth, j = 0, i + len('\\caption')
        for k in range(j, len(block)):
            if block[k] == '{':
                depth += 1
            elif block[k] == '}':
                depth -= 1
                if depth == 0:
                    return block[j + 1:k]
        return ''

    for env in ('figure', 'table'):
        for m in re.finditer(r'\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}',
                             text, re.S):
            block = m.group(1)
            cap = caption_of(block)
            lm = re.search(r'\\label\{((?:tbl|fig):[^}]+)\}', block)
            if not cap or not lm:
                continue
            label = lm.group(1)
            used = {a for a in ABBREV if re.search(rf'\b{re.escape(a)}\b', cap)}
            # A definition looks like "ABBR, expansion" or "ABBR/OTHER, expansion".
            undefined = {
                a for a in used
                if not re.search(rf'\b{re.escape(a)}(?:/[A-Za-z0-9]+)*,\s+'
                                 rf'(?:[a-z]|[A-Z][a-z])', cap)}
            if undefined:
                problems.append(f'{label}: abbreviations used but not defined '
                                f'in the caption: {sorted(undefined)}')

    print(f'floats audited in {"built" if built.exists() else "template"} '
          f'manuscript')
    if problems:
        print(f'\n{len(problems)} problem(s):')
        for p in problems:
            print('  -', p)
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
