"""
Swap a new Discussion body into the manuscript template, and add any new
bibliography entries it needs.

Reads a JSON payload (the workflow's return value) with:
    discussion_latex     the new Discussion body, starting at the first \\subsection
    verified_citations   [{citekey, authors, year, title, venue, url, ...}]

Refuses to proceed if the new Discussion cites a key that is neither already in
references.bib nor in verified_citations, because that is a fabricated citation.

Usage:  python integrate_discussion.py payload.json [--dry-run]
"""
import json
import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parent.parent
TEMPLATE = REV / 'Manuscript_R1.tex.in'
BIB = REV / 'references.bib'


def strip_latex(tex: str) -> str:
    """Reduce LaTeX to the prose a reader actually reads, for word counting."""
    s = tex
    s = re.sub(r'%.*', '', s)                        # comments
    s = re.sub(r'\\citep?\{[^}]*\}', '', s)          # citations
    s = re.sub(r'\\ref\{[^}]*\}', 'X', s)            # cross-refs -> one token
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = re.sub(r'\$[^$]*\$', 'X', s)                 # inline maths -> one token
    s = re.sub(r'\\(sub)*section\*?\{([^}]*)\}', r' \2 ', s)   # keep headings
    s = re.sub(r'\\emph\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?', ' ', s)           # remaining macros
    s = re.sub(r'[{}~\\]', ' ', s)
    return s


def count_words(tex: str) -> int:
    return len([w for w in strip_latex(tex).split() if any(c.isalnum() for c in w)])


def bib_keys(text: str):
    return set(re.findall(r'@\w+\{([^,]+),', text))


def cited_keys(tex: str):
    keys = set()
    for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]*)\}', tex):
        keys |= {k.strip() for k in m.group(1).split(',') if k.strip()}
    return keys


def _tex(s):
    """Escape the LaTeX specials that turn up in bibliographic metadata.

    Venue names routinely contain ampersands ("Environmental Science &
    Technology"). Unescaped, an ampersand is an alignment character and
    aborts the build.
    """
    s = (s or '').strip()
    for ch in ('&', '%', '#'):
        s = re.sub(r'(?<!\\\\)' + re.escape(ch), '\\\\' + ch, s)
    return s


def make_entry(c: dict) -> str:
    """Build a BibTeX entry from a verified citation record.

    Entry type is inferred from the venue: preprints become @misc with an
    eprint field, conference proceedings become @inproceedings, everything else
    @article. Getting this wrong is cosmetic under a numeric style but reads as
    carelessness to a copy editor.
    """
    url = (c.get('url') or '').strip()
    venue = (c.get('venue') or '').strip()
    doi = ''
    m = re.search(r'(10\.\d{4,9}/[^\s"<>)]+)', url + ' ' + venue)
    if m:
        doi = m.group(1).rstrip('.),;')
    arxiv = re.search(r'arXiv[:\s]*(\d{4}\.\d{4,5})', url + ' ' + venue, re.I)
    is_proc = bool(re.search(
        r'\b(proc\.|proceedings|conference|NeurIPS|ICML|ICLR|Advances in Neural)\b',
        venue, re.I))

    author = c.get('authors', '').strip()
    title = c.get('title', '').strip()
    year = str(c.get('year', '')).strip()
    # Strip any trailing DOI/URL noise the verifier folded into the venue.
    venue_clean = re.sub(r'[;,]?\s*(doi|DOI|https?://)\S*.*$', '', venue).strip(' ;,')

    if arxiv and not doi:
        fields = [f"  author       = {{{_tex(author)}}}",
                  f"  title        = {{{_tex(title)}}}",
                  f"  year         = {{{year}}}",
                  f"  eprint       = {{{arxiv.group(1)}}}",
                  "  archivePrefix = {arXiv}"]
        return "@misc{%s,\n%s,\n}\n" % (c['citekey'], ',\n'.join(fields))

    kind, venue_field = ('inproceedings', 'booktitle') if is_proc else ('article', 'journal')
    fields = [f"  author  = {{{_tex(author)}}}",
              f"  title   = {{{_tex(title)}}}",
              f"  {venue_field:7s} = {{{_tex(venue_clean)}}}",
              f"  year    = {{{year}}}"]
    if doi:
        fields.append(f"  doi     = {{{doi}}}")
    elif url:
        fields.append(f"  note    = {{Available at \\url{{{url}}}}}")
    return "@%s{%s,\n%s,\n}\n" % (kind, c['citekey'], ',\n'.join(fields))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    dry = '--dry-run' in sys.argv

    body = payload['discussion_latex'].strip()
    # Tolerate the agent wrapping the answer in a fence or repeating the section head.
    body = re.sub(r'^```[a-zA-Z]*\n', '', body)
    body = re.sub(r'\n```$', '', body)
    body = re.sub(r'^\\section\*?\{Discussion\}\s*', '', body)

    verified = {c['citekey']: c for c in payload.get('verified_citations', [])}
    bibtext = BIB.read_text(encoding='utf-8')
    existing = bib_keys(bibtext)

    used = cited_keys(body)
    unknown = sorted(used - existing - set(verified))
    if unknown:
        raise SystemExit(
            f'REFUSING TO INTEGRATE: the new Discussion cites {len(unknown)} key(s) that are '
            f'neither in references.bib nor in the verified set. These are fabrications:\n'
            + '\n'.join(f'   {k}' for k in unknown))

    new_keys = sorted((used & set(verified)) - existing)
    print(f'Discussion cites {len(used)} keys: {len(used & existing)} already in the bib, '
          f'{len(new_keys)} new, 0 unknown.')

    # ---- word counts -----------------------------------------------------
    total = count_words(body)
    # Sum EVERY subsection whose title mentions limitations or future work. The
    # brief requires them combined into one; if the draft split them anyway, the
    # 350-word budget still applies to the pair, so counting only the first
    # would let a split draft pass a limit it actually breaches.
    lim_parts = re.findall(
        r'\\subsection\{[^}]*(?:Limitation|Future)[^}]*\}(.*?)(?=\\subsection|\Z)',
        body, re.S | re.I)
    lim = sum(count_words(p) for p in lim_parts)
    if len(lim_parts) > 1:
        print(f'NOTE: limitations/future appear in {len(lim_parts)} separate '
              f'subsections; the brief asks for one combined section.')
    print(f'Word count: {total} total (limit 1500), {lim} limitations+future (limit 350)')
    over = []
    if total > 1500:
        over.append(f'total {total} > 1500')
    if lim > 350:
        over.append(f'limitations+future {lim} > 350')
    if over:
        print('WARNING: over budget: ' + '; '.join(over))

    if dry:
        print('\n--- dry run, nothing written ---')
        return 0 if not over else 1

    # ---- splice into the template ----------------------------------------
    tpl = TEMPLATE.read_text(encoding='utf-8')
    pat = re.compile(r'(\\section\{Discussion\}\s*\n)(.*?)(?=\n%+\n%% CONCLUSIONS)', re.S)
    if not pat.search(tpl):
        raise SystemExit('could not locate the Discussion section in the template')
    tpl = pat.sub(lambda mm: mm.group(1) + '\n' + body + '\n', tpl)
    TEMPLATE.write_text(tpl, encoding='utf-8')
    print(f'[write] {TEMPLATE.name}: Discussion replaced')

    # ---- append new bib entries ------------------------------------------
    if new_keys:
        add = '\n' + '\n'.join(make_entry(verified[k]) for k in new_keys)
        BIB.write_text(bibtext.rstrip() + '\n' + add, encoding='utf-8')
        print(f'[write] references.bib: {len(new_keys)} entries added '
              f'({", ".join(new_keys)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
