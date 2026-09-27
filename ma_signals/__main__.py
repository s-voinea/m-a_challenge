"""CLI: python -m ma_signals data/trans50m_2k18.xlsx --out outputs/

Use `--synthetic` for a dry run on generated data when the xlsx is absent.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import run
from .sectors import UNCLASSIFIED, build_llm_prompt, select_for_llm
from .signals import flow_correlation, latest_snapshot

FEATURE_COLS = ['consideration', 'premium_pct', 'stake_pct', 'is_sponsor', 'is_tender_offer', 'is_hostile',
                'is_competing_bid', 'is_take_private', 'is_spac', 'is_nonbinding', 'debt_financed',
                'antitrust_review', 'national_security_review', 'termination_fee_ratio', 'cross_border']


def _md_table(df, floatfmt='.2f'):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else format(v, floatfmt))
    cols = [str(df.index.name or '')] + [str(c) for c in df.columns]
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    rows += ['| ' + ' | '.join([str(i)] + [str(v) for v in r]) + ' |' for i, r in zip(df.index, df.to_numpy())]
    return '\n'.join(rows)


def feature_coverage(deals):
    rows = {}
    for c in FEATURE_COLS:
        s = deals[c]
        if s.dtype == bool:
            rows[c] = {'coverage_or_rate': s.mean(), 'kind': 'flag rate'}
        elif c == 'consideration':
            rows[c] = {'coverage_or_rate': s.ne('unknown').mean(), 'kind': 'parsed'}
        else:
            rows[c] = {'coverage_or_rate': s.notna().mean(), 'kind': 'parsed'}
    cov = pd.DataFrame(rows).T.rename_axis('feature')
    cov['coverage_or_rate'] = cov['coverage_or_rate'].astype(float)
    return cov


def write_report(out, path, t_elapsed):
    deals, sig, diag = out['deals'], out['signals'], out['classifier_diag']
    snap = latest_snapshot(sig)
    cols = ['n_deals', 'expected_n', 'flow_z', 'count_mom', 'premium_z', 'sponsor_z', 'stock_z', 'heat', 'sparse']
    recent = sig[~sig['provisional']]
    last12 = recent[recent.index.get_level_values('month') > recent.index.get_level_values('month').max() - 12]
    top_surprises = last12['flow_z'].dropna().sort_values(ascending=False).head(10).to_frame()
    top_surprises.index = [f'{m} · {s}' for m, s in top_surprises.index]
    top_surprises.index.name = 'month · sector'
    src = deals['sector_source'].value_counts(normalize=True).rename('share').to_frame()
    mix = deals.groupby('sector').agg(n=('transaction_id', 'size'), sponsor=('is_sponsor', 'mean'),
                                      median_premium=('premium_pct', 'median'),
                                      median_value=('value_usd_mm', 'median')).sort_values('n', ascending=False)
    lines = [
        '# M&A deal-flow signals: run report', '',
        f'- Deals after cleaning (partial last month dropped): **{len(deals):,}**',
        f'- Date range: {deals["announced_date"].min():%Y-%m-%d} to {deals["announced_date"].max():%Y-%m-%d}',
        f'- Runtime: {t_elapsed:.1f}s', '',
        '## Sector classification', '',
        f'- Rule coverage: {diag["rule_coverage"]:.1%}. Final coverage after the text model: {diag["final_coverage"]:.1%} '
        f'({diag["model_added"]:,} deals added by the model).',
        f'- Out-of-fold model/rule agreement: {diag.get("oof_agreement_with_rules", float("nan")):.1%}. '
        'This is an upper bound on accuracy, not an accuracy. Check it against `llm_labeling_queue.csv` once labelled.', '',
        _md_table(src), '',
        '## Sector composition', '', _md_table(mix), '',
        '## Text-feature coverage', '', _md_table(feature_coverage(deals)), '',
        f'## Latest reliable month ({snap.attrs["month"]}) ranked by heat', '',
        '`heat` = mean of cross-sectional z-scores of flow surprise, count momentum, premium and contested-deal share. '
        'It says how crowded a sector is. It has not been validated against returns; see `information_coefficient`.', '',
        _md_table(snap[cols]), '',
        '## Largest market-neutral flow surprises, last 12 reliable months', '', _md_table(top_surprises), '',
        '## Caveats', '',
        '- The last 3 months are marked `provisional`: CIQ backfills deals after announcement, so recent counts are biased low.',
        '- Rows marked `sparse` have a trailing median below 3 deals/month; aggregate them quarterly before using.',
        '- The dataset has only Closed/Announced/Effective statuses, so withdrawn deals appear to be excluded. '
        'Completion-rate analysis therefore suffers from survivorship bias.',
        '- Buyer country is inferred from the listing exchange (listed buyers only). Cross-border share covers that subset.',
    ]
    Path(path).write_text('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('xlsx', nargs='?', help='path to trans50m_2k18.xlsx')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--synthetic', action='store_true', help='run on generated data (no xlsx needed)')
    ap.add_argument('--gold', help='CSV with transaction_id,sector gold labels (e.g. from the LLM queue)')
    ap.add_argument('--llm-queue', type=int, default=400, help='rows to export for LLM labelling')
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.synthetic:
        from .synthetic import make_raw
        raw = make_raw().drop(columns='_true_sector')
    elif a.xlsx:
        raw = pd.read_excel(a.xlsx, header=1)
    else:
        ap.error('give an xlsx path or --synthetic')

    gold = None
    if a.gold:
        from .load import clean
        g = pd.read_csv(a.gold)
        ids = clean(raw)['transaction_id']
        gold = g.set_index('transaction_id')['sector'].reindex(ids.to_numpy())
        gold.index = ids.index
    out = run(raw, gold=gold)

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    deals = out['deals']
    deals.drop(columns=['month']).to_csv(od / 'deals_enriched.csv.gz', index=False)
    out['panel'].to_csv(od / 'sector_panel_monthly.csv')
    out['signals'].to_csv(od / 'sector_signals_monthly.csv')
    flow_correlation(out['panel']).to_csv(od / 'flow_change_correlation.csv')

    labels = deals[['sector', 'sector_source', 'sector_model_conf']]
    q = select_for_llm(deals, labels, out['model'], n=a.llm_queue)
    queue = deals.loc[q, ['transaction_id', 'target', 'buyer', 'sector', 'sector_model_conf']].copy()
    queue['prompt'] = [build_llm_prompt(deals.loc[i]) for i in q]
    queue.to_csv(od / 'llm_labeling_queue.csv', index=False)

    write_report(out, od / 'report.md', time.time() - t0)
    print(f'Wrote {od}/ in {time.time() - t0:.1f}s  |  coverage {out["classifier_diag"]["final_coverage"]:.1%}')


if __name__ == '__main__':
    main()
