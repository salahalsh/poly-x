"""
Script 11 - Supporting Information, generated from source rather than typed.

Table S1  Complete group contribution parameter table with provenance
          (Reviewer 2, Minor 4: "the manuscript or Supporting Information
          should specify the parameter value, unit, source, whether it was
          adopted or refitted, and the calibration structures used").
Table S2  The twenty largest Tier 1 failures and the unmatched elements
          driving them (Reviewer 1; Reviewer 2, Major 5).
Table S3  Prospective PI1M candidates from the design case study
          (Reviewer 2, Major 14).
Table S4  Per-seed results for every run in the benchmark
          (Reviewer 2, Major 3 and Minor 3).

Table S1 is parsed directly out of ``group_contribution.py``: the value, the
inline source comment and the module's own calibration header, so it cannot
drift away from the code it documents.

Writes: supplementary_R1.tex
        outputs/s11_gc_parameters.csv
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from polyx_rev import OUT, REVISION, POLYX, dump

GC_SRC = POLYX / 'services' / 'group_contribution.py'

ROW = re.compile(
    r"^\s*\('(?P<smarts>[^']+)',\s*'(?P<name>[^']+)',\s*"
    r"(?P<yg>-?[\d.]+),\s*(?P<ecoh>-?[\d.]+),\s*(?P<vw>-?[\d.]+),\s*"
    r"(?P<pri>\d)\),\s*(?:#\s*(?P<src>.*))?$")


def parse_parameters():
    text = GC_SRC.read_text(encoding='utf-8')
    body = text.split('GROUP_CONTRIBUTIONS = [', 1)[1].split('\n]', 1)[0]
    rows = []
    for line in body.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        d = m.groupdict()
        src = (d['src'] or '').strip()
        # "adopted" if the comment cites a table in a primary source;
        # "calibrated here" if the comment says the value was set to reproduce
        # a measured Tg.
        status = ('calibrated in this work' if 'cal.' in src or 'corrected' in src
                  else 'adopted' if src else 'unattributed')
        rows.append({
            'smarts': d['smarts'], 'name': d['name'],
            'Yg_K_g_per_mol': float(d['yg']),
            'Ecoh_J_per_mol': float(d['ecoh']),
            'Vw_cm3_per_mol': float(d['vw']),
            'priority': int(d['pri']),
            'source': src or '(none given)',
            'status': status,
        })
    # The module header states the structures the Yg column was calibrated on.
    cal = re.search(r'Yg calibrated against experimental Tg[^\n]*\n#\s*(.*?)\n#\s*(.*?)\n',
                    text, re.S)
    calibration_set = []
    if cal:
        blob = ' '.join(cal.groups())
        calibration_set = re.findall(r'([A-Za-z0-9\-]+)=\d+K', blob)
    return pd.DataFrame(rows), calibration_set


def esc(s):
    return (s.replace('\\', r'\textbackslash{}').replace('_', r'\_')
             .replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
             .replace('$', r'\$'))


def table_s1(df, cal):
    # 'name' collides with the namedtuple attribute of itertuples; rename it.
    df = df.rename(columns={'name': 'group_name'})
    body = '\n'.join(
        rf"    \texttt{{{esc(r.smarts)}}} & {esc(r.group_name)} & {r.priority} & "
        rf"{r.Yg_K_g_per_mol:.1f} & {r.Ecoh_J_per_mol:,.0f} & "
        rf"{r.Vw_cm3_per_mol:.1f} & {esc(r.source)} & {r.status} \\"
        for r in df.itertuples())
    calstr = ', '.join(cal) if cal else 'not stated in the source'
    return rf"""\section{{Group contribution parameter table}}

Table~\ref{{tbl:s1}} lists every parameter of the Tier 1 group library, taken
directly from the source file \texttt{{group\_contribution.py}}
so that the table and the code cannot diverge. Units: $Y_g$ in
K$\cdot$g$\,$mol$^{{-1}}$ (divided by 1000 in Eq.~1), $E_{{\mathrm{{coh}}}}$ in
J$\,$mol$^{{-1}}$, $V_w$ in cm$^3\,$mol$^{{-1}}$. Sources: VK = Van Krevelen and
Te Nijenhuis (2009); Bic = Bicerano (2002); PH = Brandrup, Immergut and Grulke
(1999); T$n$ denotes a table number in the cited work.

\textbf{{Calibration structures.}} The module header records that the $Y_g$
column was calibrated against the experimental $T_g$ of the following polymers:
\emph{{{calstr}}}. This is important for interpreting Table~1 of the main text:
\textbf{{all eight of the canonical homopolymers used there appear in this
calibration set}}. The 3.9~K mean absolute error over those eight structures is
therefore a measure of calibration fit and implementation correctness, not of
predictive accuracy, which is why the main text reports the full-collection
evaluation alongside it.

\begin{{table}}[htbp]
  \caption{{Complete Tier 1 group contribution parameter table with provenance.}}
  \label{{tbl:s1}}
  \centering
  \scriptsize
  \begin{{tabular}}{{llccrcll}}
    \toprule
    SMARTS & Group & Pri. & $Y_g$ & $E_{{\mathrm{{coh}}}}$ & $V_w$ & Source & Status \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def table_s2(gc):
    rows = '\n'.join(
        rf"    \texttt{{{esc(r['smiles'][:52])}\ldots}} & {esc(r['family'].replace('_',' '))} & "
        rf"{r['exp_tg']:.0f} & {r['pred_tg']:.0f} & {r['abs_error']:.0f} & "
        rf"{100*r['atom_coverage']:.0f}\% & {esc(r['unmatched_elements'] or 'n/a')} \\"
        for r in gc['worst_20'])
    unmatched = '\n'.join(
        rf"    {esc(k)} & {v:,} \\"
        for k, v in list(gc['most_common_unmatched_elements'].items())[:10])
    return rf"""\section{{Tier 1 failures}}

\begin{{table}}[htbp]
  \caption{{The twenty largest Tier 1 errors over the full collection, with the
  fraction of heavy atoms matched by a functional group and the elements left
  unmatched.}}
  \label{{tbl:s2}}
  \centering
  \scriptsize
  \begin{{tabular}}{{llcccll}}
    \toprule
    PSMILES & Family & Exp.\ $T_g$ & Pred. & $|$error$|$ & Coverage & Unmatched \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}

  \smallskip
  \begin{{tabular}}{{lc}}
    \multicolumn{{2}}{{l}}{{\textbf{{Most frequent unmatched element sets}}}} \\
    \toprule
    Elements & Polymers \\
    \midrule
{unmatched}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def table_s3(path):
    if not path.exists():
        return ''
    d = pd.read_csv(path).head(25)
    rows = '\n'.join(
        rf"    \texttt{{{esc(str(r.psmiles)[:46])}\ldots}} & {r.tier2_ml_tg_K:.0f} & "
        rf"{('%.0f' % r.tier1_gc_tg_K) if pd.notna(r.tier1_gc_tg_K) else 'n/a'} & "
        rf"{r.disagreement_K:.0f} & {r.max_tanimoto_to_training:.2f} & {r.pri:.2f} & "
        rf"{esc(str(r.family).replace('_',' '))} \\"
        for r in d.itertuples())
    return rf"""\section{{Prospective candidates}}

\begin{{table}}[htbp]
  \caption{{Highest-reliability PI1M candidates falling in the target $T_g$
  window of the design case study, ranked by Prediction Reliability Index.
  \textbf{{These are computational proposals only: no polymer listed here was
  synthesised or measured in this work, and no claim is made about them beyond
  what the reliability indicators state.}} None is present in the training
  partition. Similarity is the maximum Tanimoto to that partition computed on
  hydrogen-capped ECFP4, so a value at or near 1.00 marks a repeat unit that the
  shipped Tier~2 representation cannot distinguish from a training polymer even
  though the two are chemically distinct: such a candidate is novel as a
  structure but not as a feature vector, and should be read accordingly.
  $T_g$, glass transition temperature; PRI, Prediction Reliability Index;
  Disagr., inter-model disagreement; ECFP4, extended-connectivity fingerprint of
  diameter 4.}}
  \label{{tbl:s3}}
  \centering
  \scriptsize
  \begin{{tabular}}{{lcccccl}}
    \toprule
    PSMILES & Tier 2 $T_g$ & Tier 1 $T_g$ & Disagr. & Max Tanimoto & PRI & Family \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def table_s4():
    import glob
    rows = []
    for p in sorted(glob.glob(str(OUT / 'runs' / '*.json'))):
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        c = d['components']
        rows.append(
            rf"    {d['protocol']} & {d['seed']} & {esc(d['representation'])} & "
            rf"{c.get('rf', {}).get('test', {}).get('r2', float('nan')):.3f} & "
            rf"{c.get('rf', {}).get('test', {}).get('mae', float('nan')):.1f} & "
            rf"{c.get('gb', {}).get('test', {}).get('r2', float('nan')):.3f} & "
            rf"{c.get('gb', {}).get('test', {}).get('mae', float('nan')):.1f} & "
            rf"{d['ensemble']['w_rf'] if d['ensemble']['w_rf'] is not None else 'n/a'} & "
            rf"{d['ensemble']['test']['r2']:.3f} & "
            rf"{d['ensemble']['test']['mae']:.1f} & "
            rf"{d['ensemble']['test']['rmse']:.1f} \\")
    body = '\n'.join(rows)
    return rf"""\section{{Complete per-run results}}

Every run in the benchmark, with both component models reported separately
(Reviewer 2, Minor 3). NaN entries are runs whose representation uses a single
model (the mean predictor and ridge regression).

\begin{{longtable}}{{llccccccccc}}
  \caption{{All benchmark runs. $w_{{\mathrm{{RF}}}}$ is the blend weight selected
  on the validation partition.}}\label{{tbl:s4}}\\
  \toprule
  Protocol & Seed & Representation & RF $R^2$ & RF MAE & GB $R^2$ & GB MAE & $w_{{\mathrm{{RF}}}}$ & Ens.\ $R^2$ & Ens.\ MAE & RMSE \\
  \midrule
  \endfirsthead
  \toprule
  Protocol & Seed & Representation & RF $R^2$ & RF MAE & GB $R^2$ & GB MAE & $w_{{\mathrm{{RF}}}}$ & Ens.\ $R^2$ & Ens.\ MAE & RMSE \\
  \midrule
  \endhead
{body}
  \bottomrule
\end{{longtable}}"""


def main():
    df, cal = parse_parameters()
    df.to_csv(OUT / 's11_gc_parameters.csv', index=False)
    dump('s11_gc_calibration_set.json', {
        'calibration_structures': cal,
        'n_parameters': int(len(df)),
        'note': ('Parsed from the header of GROUP_CONTRIBUTIONS in '
                 'group_contribution.py. All eight canonical validation '
                 'polymers of Table 1 appear in this list.'),
    })
    print(f'[params] {len(df)} groups; calibration set: {cal}')

    gc = json.loads((OUT / 's04_gc_summary.json').read_text(encoding='utf-8'))

    doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{amsmath,amssymb}}
\usepackage{{url}}
\usepackage{{hyperref}}
\sloppy
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\renewcommand{{\thesection}}{{S\arabic{{section}}}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\setcounter{{table}}{{0}}

\title{{Supporting Information\\
POLY-X: A Multi-Tier Platform for Polymer Glass Transition Temperature
Prediction with Calibrated Reliability Assessment}}
\author{{Iqbal H.\ Jebril \and Salah A.\ Alshehade}}
\date{{}}

\begin{{document}}
\maketitle

{table_s1(df, cal)}

{table_s2(gc)}

{table_s3(OUT / 's08_pi1m_candidates.csv')}

{table_s4()}

\end{{document}}
"""
    (REVISION / 'supplementary_R1.tex').write_text(doc, encoding='utf-8')
    print(f'[write] {REVISION / "supplementary_R1.tex"}')


if __name__ == '__main__':
    main()
