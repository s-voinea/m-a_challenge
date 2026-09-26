"""Synthetic CIQ-shaped data for tests and demos. Not real deals.

The generator plants known structure so the pipeline can be checked end to
end: sector-specific context words that are not seed keywords (so the text
model has something to learn), sponsor buyers, premiums, and an injected
deal-flow burst (`BURST`) that `flow_surprise` should pick up.
"""
import numpy as np
import pandas as pd

from .load import COL_RENAME

# sector -> (seed keyword phrases, non-seed context words)
_SECTOR_TEXT = {
    'Semiconductors & Chips': (['semiconductor', 'wafer', 'fabless'], ['nanometer node', 'tapeout', 'die yield', 'photomask']),
    'Biotech & Pharma': (['biotech', 'oncology', 'clinical-stage'], ['phase ii readout', 'molecule pipeline', 'fda approval', 'preclinical asset']),
    'AI & Machine Learning': (['artificial intelligence', 'machine learning'], ['inference workloads', 'model training', 'gpu cluster', 'foundation model']),
    'Clean Energy & EV': (['solar', 'renewable', 'battery'], ['megawatt portfolio', 'gigawatt capacity', 'grid interconnection', 'ppa offtake']),
    'Real Estate': (['real estate', 'office property', 'apartments'], ['tenant roster', 'occupancy rate', 'lease-up', 'cap rate']),
    'Financial Services': (['bank', 'insurance', 'asset management'], ['deposit base', 'loan book', 'assets under administration', 'premiums written']),
    'Software & Cloud': (['saas', 'enterprise software'], ['annual recurring revenue', 'subscription platform', 'net retention', 'workflow suite']),
    'Oil & Gas': (['oil', 'natural gas', 'upstream'], ['acreage position', 'permian basin', 'well inventory', 'rig count']),
}
BURST = ('AI & Machine Learning', pd.Period('2023-06', 'M'), pd.Period('2023-09', 'M'), 3.0)

_SPONSORS = ['KKR & Co. Inc. (NYSE:KKR)', 'Blackstone Inc. (NYSE:BX)', 'Summit Ridge Capital Partners',
             'Northwind Equity Fund III', 'Thoma Bravo, L.P.']
_STRATEGICS = ['Acme Industries Inc. (NYSE:ACME)', 'Globex Corporation (NasdaqGS:GBX)',
               'Initech SA (ENXTPA:INI)', 'Umbrella plc (LSE:UMB)', 'Hooli K.K. (TSE:1234)',
               'Stark Holdings Ltd. (SEHK:888)', 'Wayne Enterprises (TSX:WAY)']
_COUNTRIES = ['United States', 'United Kingdom', 'France', 'Japan', 'China', 'Canada', 'Germany']


def make_raw(n=6000, start='2018-01-01', end='2025-10-22', seed=0):
    """Return a DataFrame with the raw CIQ column names (as read with header=1)."""
    rng = np.random.default_rng(seed)
    sectors = list(_SECTOR_TEXT)
    base_p = rng.dirichlet(np.ones(len(sectors)) * 3)
    days = pd.date_range(start, end, freq='D')
    t = pd.to_datetime(rng.choice(days, size=n))
    # Mild cycle: more deals in 2021.
    keep = rng.random(n) < np.where(t.year == 2021, 1.0, 0.75)
    t = t[keep]
    m = len(t)
    sec = rng.choice(sectors, size=m, p=base_p)

    # Injected burst: add extra AI deals inside the burst window.
    b_sec, b0, b1, mult = BURST
    per = t.to_period('M')
    in_win = (per >= b0) & (per <= b1)
    n_extra = int((mult - 1) * (in_win & (sec == b_sec)).sum()) + 30
    extra_days = pd.date_range(b0.start_time, b1.end_time.normalize(), freq='D')
    t = t.append(pd.to_datetime(rng.choice(extra_days, size=n_extra)))
    sec = np.concatenate([sec, np.repeat(b_sec, n_extra)])
    m = len(t)

    value = np.round(50 * np.exp(rng.exponential(1.1, size=m)), 2)
    sponsor = rng.random(m) < 0.25
    buyer = np.where(sponsor, rng.choice(_SPONSORS, size=m), rng.choice(_STRATEGICS, size=m))
    tgt_country = rng.choice(_COUNTRIES, size=m, p=[.5, .12, .08, .08, .1, .07, .05])

    comments, targets = [], []
    for i in range(m):
        seeds, ctx = _SECTOR_TEXT[sec[i]]
        use_seed = rng.random() < 0.6  # 40% of deals carry only context words
        desc = rng.choice(seeds) if use_seed else rng.choice(ctx)
        desc2 = rng.choice(ctx)
        cons = rng.choice(['in cash', 'in an all-stock transaction', 'in cash and stock', ''])
        prem = f' The offer represents a premium of {rng.integers(10, 70)}%.' if rng.random() < 0.3 else ''
        fee = f' A termination fee of ${value[i] * 0.03:.1f} million is payable.' if rng.random() < 0.2 else ''
        hostile = ' The proposal was unsolicited.' if rng.random() < 0.03 else ''
        name = f'Target{i} {"Technologies" if rng.random() < .5 else "Group"} Inc.'
        targets.append(name)
        comments.append(f'{buyer[i]} entered into a definitive agreement to acquire {name}, a company focused on '
                        f'{desc} with {desc2}, for ${value[i]:.1f} million {cons} on {t[i]:%B %d, %Y}.'
                        f'{prem}{fee}{hostile}')

    raw = pd.DataFrame({
        'announced_date': t,
        'announced_date_incl_bids': t - pd.to_timedelta(rng.integers(0, 30, size=m) * (rng.random(m) < .2), unit='D'),
        'target': targets,
        'ticker': np.where(rng.random(m) < 0.08, 'NYSE:TGT', '-'),
        'transaction_type': 'Merger/Acquisition',
        'status': rng.choice(['Closed', 'Announced', 'Effective'], size=m, p=[.88, .1, .02]),
        'value_usd_mm': value,
        'buyer': buyer,
        'seller': np.where(rng.random(m) < .6, 'Seller Co.', '-'),
        'transaction_id': [f'IQTR{i}' for i in range(m)],
        'comments': comments,
        'target_country': np.where(rng.random(m) < .55, tgt_country, '-'),
        'seller_country': np.where(rng.random(m) < .6, tgt_country, '-'),
        'security_type': rng.choice(['Common Equity', 'Asset'], size=m, p=[.62, .38]),
    })
    raw['_true_sector'] = sec
    inv = {v: k for k, v in COL_RENAME.items()}
    return raw.rename(columns=inv)
