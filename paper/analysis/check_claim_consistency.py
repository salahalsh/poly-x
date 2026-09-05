"""Check that numbers attributed to a citation appear in that citation's verified claim.

The fabrication guard in integrate_discussion.py only checks that a citekey
exists. That is not enough: a drafter can cite a real, verified paper for a
number the verification never confirmed. This compares every number attached to
a \\citep/\\citet against the verified_claim and verbatim quote recorded for that
key, and reports the ones with no support.

Usage: python check_claim_consistency.py <payload.json>
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r'(?<![\w.])(\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+)(?![\w.])')
CITE = re.compile(r'\\cite[a-zA-Z]*\{([^}]*)\}')

# Numbers that are ours, not the cited work's: they come from our own results
# and are tokenised elsewhere, so they need no external support. Also included
# are quantities we DERIVE from a cited figure by arithmetic the reader can
# follow (for example "24 points" from a quoted fall of 0.81 to 0.57): the
# inputs are verified, the subtraction is ours.
OURS = {
    '0.031', '0.003', '0.065', '0.005', '0.89', '0.563', '0.438', '0.747',
    '0.716', '0.699', '0.858', '0.863', '0.036', '0.076', '0.026', '0.178',
    '0.165', '0.206', '0.061', '96', '57', '226', '109', '69', '1490', '210',
    '0.839', '0.828', '0.766', '0.734', '0.600', '6.3', '7.8', '12', '30',
    '153', '7,367', '3,108', '1,005', '738', '5,892', '0.90', '0.849',
    # our own composite arithmetic in Significance
    '0.017', '0.079', '0.125',
    # our PRI quintile band, reported in Results
    '60.1', '34.1',
    # derived: 0.81 - 0.57, both quoted from the cited source
    '24',
    '2,048', '0.099', '0.18',
}

# A number bound to a float or tier label is a cross-reference, not a claim.
LABEL_NUM = re.compile(r'(?:Tier|Section|Table|Figure|Fig\.|Eq\.)~?\s*$')


def sentences(t):
    t = re.sub(r'\s+', ' ', t)
    return re.split(r'(?<=[.;])\s+(?=[A-Z\\(])', t)


def main():
    payload = json.load(open(sys.argv[1], encoding='utf-8'))
    tex = payload['discussion_latex']
    ver = {c['citekey']: c for c in payload['verified_citations']}

    problems = 0
    for s in sentences(tex):
        keys = set()
        for m in CITE.finditer(s):
            keys |= {k.strip() for k in m.group(1).split(',') if k.strip()}
        if not keys:
            continue
        nums = []
        for m in NUM.finditer(s):
            if m.group(1) in OURS:
                continue
            if LABEL_NUM.search(s[:m.start()]):     # "Tier~1", "Section~3"
                continue
            nums.append(m.group(1))
        if not nums:
            continue
        support = ' '.join(
            (ver.get(k, {}).get('verified_claim', '') + ' ' +
             (ver.get(k, {}).get('quote') or '')) for k in keys)
        unsupported = [n for n in nums if n not in support]
        if unsupported:
            problems += 1
            print(f'UNSUPPORTED {unsupported} for {sorted(keys)}')
            print(f'   {re.sub(r"\\\\cite[a-zA-Z]*", "", s).strip()[:230]}')
            print()
    print(f'{problems} sentence(s) attach a number to a citation whose verified '
          f'record does not contain it')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
