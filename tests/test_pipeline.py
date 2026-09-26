import numpy as np
import pandas as pd
import pytest

from ma_signals import run
from ma_signals.features import extract
from ma_signals.load import clean, parse_listing
from ma_signals.sectors import UNCLASSIFIED, buyer_home_sector, rule_labels, rule_scores
from ma_signals.synthetic import BURST, make_raw


@pytest.fixture(scope='module')
def synth():
    raw = make_raw(n=4000, seed=1)
    truth = raw.pop('_true_sector')
    return run(raw), truth


def _deals(comments, buyer='Acme Corp. (NYSE:ACM)', target='Target Inc.', value=1000.0):
    raw = pd.DataFrame({
        'All Transactions Announced Date': ['2024-01-15'] * len(comments),
        'All Transactions Announced Date (Including Bids and Letters of Intent)': ['2024-01-10'] * len(comments),
        'Target/Issuer': target, 'Exchange:Ticker': '-', 'Transaction Types': 'Merger/Acquisition',
        'Transaction Status': 'Closed', 'Total Transaction Value ($USDmm, Historical rate)': value,
        'Buyers/Investors': buyer, 'Sellers': '-', 'CIQ Transaction ID': [f'IQ{i}' for i in range(len(comments))],
        'Transaction Comments': comments, 'Country/Region of Incorporation [Target/Issuer]': 'France',
        'Country/Region of Incorporation [Sellers]': '-', 'Target Security Types': 'Common Equity',
    })
    return clean(raw)


def test_parse_listing():
    assert parse_listing('Bristol-Myers Squibb Company (NYSE:BMY)') == ('NYSE', 'BMY')
    assert all(pd.isna(x) for x in parse_listing('Private Co'))


def test_clean_buyer_country_proxy():
    d = _deals(['x'])
    assert d.loc[0, 'buyer_country_proxy'] == 'United States'
    assert d.loc[0, 'bid_lead_days'] == 5


@pytest.mark.parametrize('text,col,expected', [
    ('The offer represents a premium of approximately 32.5% to the closing price.', 'premium_pct', 32.5),
    ('The price reflects a 45% premium over the 30-day VWAP.', 'premium_pct', 45.0),
    ('Acme agreed to acquire the remaining 40% stake in Target.', 'stake_pct', 40.0),
    ('A termination fee of $50 million is payable.', 'termination_fee_ratio', 0.05),
    ('A termination fee of $1.2 billion is payable.', 'termination_fee_ratio', 0.5),  # capped -> NaN below
])
def test_numeric_extraction(text, col, expected):
    f = extract(_deals([text]))
    val = f.loc[0, col]
    if col == 'termination_fee_ratio' and expected >= 0.5:
        assert pd.isna(val) or val <= 0.5
    else:
        assert val == pytest.approx(expected)


@pytest.mark.parametrize('text,flag', [
    ('Acme launched a tender offer for all outstanding shares.', 'is_tender_offer'),
    ('Acme made an unsolicited proposal to acquire Target.', 'is_hostile'),
    ('Upon completion, Target will be delisted from NYSE.', 'is_take_private'),
    ('The deal is subject to CFIUS clearance.', 'national_security_review'),
    ('The deal is subject to expiry of the Hart-Scott-Rodino waiting period.', 'antitrust_review'),
    ('The transaction will be funded through a bridge loan.', 'debt_financed'),
    ('Acme signed a non-binding letter of intent.', 'is_nonbinding'),
])
def test_flags(text, flag):
    assert bool(extract(_deals([text])).loc[0, flag])


def test_consideration():
    f = extract(_deals(['for $10 per share in cash.', 'in an all-stock transaction.',
                        'for $5 in cash and 0.5 shares of Acme common stock.', 'for $1 billion.']))
    assert list(f['consideration']) == ['cash', 'stock', 'mixed', 'unknown']


def test_sponsor_detection():
    buyers = ['KKR & Co. Inc. (NYSE:KKR)', 'Riverbend Capital Partners', 'Riverbend Partners, L.P. (NYSE:RBP)']
    d = pd.concat([_deals(['x'], buyer=b) for b in buyers], ignore_index=True)
    assert list(extract(d)['is_sponsor']) == [True, True, False]


def test_rules_avoid_known_false_positives():
    d = _deals(['Acme acquired a biotech firm with a deep clinical-stage molecule pipeline and strong intellectual property.',
                'Acme acquired a property and casualty insurance carrier.'])
    labels = rule_labels(rule_scores(d))
    assert labels[0] == 'Biotech & Pharma'
    assert labels[1] == 'Financial Services'


def test_buyer_home_sector_is_leave_one_out():
    d = pd.DataFrame({'lead_buyer': ['A', 'A', 'A', 'B'],
                      'sector': ['Oil & Gas', 'Oil & Gas', 'Real Estate', UNCLASSIFIED]})
    home = buyer_home_sector(d, min_other_deals=2)
    assert home[2] == 'Oil & Gas'          # the RE deal's buyer is an O&G acquirer
    assert home[0] in ('Oil & Gas', 'Real Estate')  # tie after leaving itself out
    assert pd.isna(home[3])


def test_classifier_extends_coverage_accurately(synth):
    out, truth = synth
    d = out['deals']
    assert out['classifier_diag']['final_coverage'] > out['classifier_diag']['rule_coverage']
    m = d['sector_source'].eq('model')
    assert (d.loc[m, 'sector'] == truth.loc[d.index[m]]).mean() > 0.85


def test_flow_surprise_detects_injected_burst(synth):
    out, _ = synth
    sector, b0, b1, _ = BURST
    z = out['signals'].xs(sector, level='sector')['flow_z']
    inside = z.loc[b0:b1].mean()
    outside = z.drop(z.loc[b0:b1].index).mean()
    assert inside > 1.5 and abs(outside) < 0.5


def test_signals_are_point_in_time(synth):
    """Appending future data must not change past signal values."""
    out, _ = synth
    panel = out['panel']
    from ma_signals.signals import build_signals
    cut = pd.Period('2022-12', 'M')
    early = build_signals(panel[panel.index.get_level_values('month') <= cut])
    full = out['signals']
    cols = ['flow_z', 'count_mom', 'sponsor_z', 'premium_z']
    a = early.loc[early.index.get_level_values('month') <= cut, cols]
    b = full.loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b)


def test_partial_month_dropped_and_provisional_flagged(synth):
    out, _ = synth
    sig = out['signals']
    last = sig.index.get_level_values('month').max()
    assert last == pd.Period('2025-09', 'M')  # synthetic data ends 2025-10-22
    assert sig.xs(last, level='month')['provisional'].all()
