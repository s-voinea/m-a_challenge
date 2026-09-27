"""Monthly sector panel -> robust, point-in-time deal-flow signals.

Design choices (and why)
- **Counts over summed value.** Summed value is dominated by a handful of
  mega-deals (the old notebook shows top-5 deals = 50-90% of value in several
  sectors). Counts carry more information per observation. Value enters only
  winsorised.
- **Market-neutral surprise.** Sector counts are compared with
  `market count x trailing sector share`. This removes the common M&A cycle
  and market-wide seasonality (e.g. December spikes) in one step, and it needs
  no seasonal model.
- **Overdispersion-aware z-scores.** Deal arrivals cluster, so a plain
  binomial variance understates noise. The dispersion factor is estimated on
  the trailing window only.
- **Point-in-time.** Every expectation uses data strictly before month t.
  The trailing partial month is dropped, and the last months are flagged as
  provisional because database backfill adds deals late.
"""
import numpy as np
import pandas as pd

from .sectors import UNCLASSIFIED

SHARE_COLS = {
    'sponsor_share': 'is_sponsor',
    'cash_share': 'cash_flag',
    'stock_share': 'stock_flag',
    'take_private_share': 'is_take_private',
    'contested_share': 'contested_flag',
    'debt_financed_share': 'debt_financed',
    'minority_share': 'minority_flag',
}

# Liquid ETFs that roughly track each sector. Only for optional validation
# (see `information_coefficient`); check the tickers before relying on them.
SECTOR_ETF_CANDIDATES = {
    'Semiconductors & Chips': 'SOXX', 'Biotech & Pharma': 'XBI', 'Clean Energy & EV': 'ICLN',
    'Cybersecurity': 'CIBR', 'Aerospace & Defense': 'ITA', 'Robotics & Automation': 'BOTZ',
    'Software & Cloud': 'IGV', 'AI & Machine Learning': 'AIQ', 'Quantum Computing': 'QTUM',
    'Healthcare & Medtech': 'IHI', 'Fintech': 'FINX', 'Telecom & 5G': 'IYZ',
    'Oil & Gas': 'XLE', 'Mining & Metals': 'XME', 'Real Estate': 'VNQ', 'Financial Services': 'XLF',
}


def drop_partial_tail(df, date_col='announced_date', min_days=25):
    """Drop the final calendar month if the data covers fewer than `min_days` of it."""
    last = df[date_col].max()
    if last.day < min_days:
        return df[df[date_col] < last.to_period('M').start_time]
    return df


def _prep_flags(df):
    d = df.copy()
    d['cash_flag'] = d['consideration'].eq('cash')
    d['stock_flag'] = d['consideration'].isin(['stock', 'mixed'])
    d['contested_flag'] = d['is_hostile'] | d['is_competing_bid']
    d['minority_flag'] = d['is_minority'].fillna(False).astype(bool)
    return d


def sector_panel(df, sector_col='sector', winsor_q=0.99, include_unclassified=False):
    """Month x sector panel of counts, robust value and deal-composition shares.

    Returns (panel indexed by [month, sector], market Series of all-deal counts).
    """
    d = _prep_flags(df)
    cap = d['value_usd_mm'].quantile(winsor_q)
    d['value_w'] = d['value_usd_mm'].clip(upper=cap)
    months = pd.period_range(d['month'].min(), d['month'].max(), freq='M')
    market = d.groupby('month').size().reindex(months, fill_value=0).rename('market_n')

    if not include_unclassified:
        d = d[d[sector_col].ne(UNCLASSIFIED)]
    g = d.groupby(['month', sector_col], observed=True)
    agg = {'n_deals': ('transaction_id', 'size'), 'value_w': ('value_w', 'sum'),
           'median_value': ('value_usd_mm', 'median'), 'premium_median': ('premium_pct', 'median'),
           'n_premium': ('premium_pct', 'count'), 'cross_border_n': ('cross_border', 'sum'),
           'geo_known_n': ('has_buyer_country', 'sum')}
    agg.update({k: (v, 'mean') for k, v in SHARE_COLS.items()})
    panel = g.agg(**agg)

    sectors = sorted(d[sector_col].unique())
    full = pd.MultiIndex.from_product([months, sectors], names=['month', 'sector'])
    panel.index = panel.index.set_names(['month', 'sector'])
    panel = panel.reindex(full)
    panel[['n_deals', 'value_w', 'n_premium', 'cross_border_n', 'geo_known_n']] = \
        panel[['n_deals', 'value_w', 'n_premium', 'cross_border_n', 'geo_known_n']].fillna(0)
    panel['cross_border_share'] = panel['cross_border_n'] / panel['geo_known_n'].replace(0, np.nan)
    panel = panel.join(market, on='month')
    return panel, market


def flow_surprise(panel, window=24, min_periods=12, prior_n=2.0):
    """Market-neutral abnormal deal flow, as a z-score per (month, sector).

    expected_t = market_n_t * p_hat, where p_hat is the sector's share of all
    deals over the previous `window` months (shrunk toward 1/n_sectors by
    `prior_n` pseudo-deals). The variance uses a binomial term times a
    trailing dispersion factor phi >= 1.
    """
    n = panel['n_deals'].unstack('sector')
    N = panel['market_n'].unstack('sector').iloc[:, 0]
    k = n.shape[1]
    roll_n = n.shift(1).rolling(window, min_periods=min_periods).sum()
    roll_N = N.shift(1).rolling(window, min_periods=min_periods).sum()
    p = (roll_n + prior_n / k).div(roll_N + prior_n, axis=0)
    expected = p.mul(N, axis=0)
    var = expected * (1 - p)
    pearson = (n - expected) / np.sqrt(var)
    phi = (pearson.pow(2).shift(1).rolling(window, min_periods=min_periods).mean()).clip(lower=1.0)
    z = (n - expected) / np.sqrt(var * phi)
    return z.stack(future_stack=True).rename('flow_z'), expected.stack(future_stack=True).rename('expected_n')


def ewm_momentum(series_wide, fast=3, slow=12):
    """log(EWMA_fast + 1) - log(EWMA_slow + 1), per column. Positive = accelerating."""
    f = series_wide.ewm(halflife=fast, min_periods=fast).mean()
    s = series_wide.ewm(halflife=slow, min_periods=slow).mean()
    return np.log1p(f) - np.log1p(s)


def trailing_z(series_wide, window=24, min_periods=12):
    """Z-score of each value against that column's own previous `window` months."""
    mu = series_wide.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = series_wide.shift(1).rolling(window, min_periods=min_periods).std()
    return (series_wide - mu) / sd.replace(0, np.nan)


def build_signals(panel, provisional_months=3, smooth_halflife=2):
    """Assemble the signal table (index [month, sector]).

    Columns
    - flow_z / flow_z_smooth: market-neutral abnormal deal count (EWMA-smoothed).
    - count_mom: fast/slow EWMA momentum on counts.
    - value_mom: same on winsorised value.
    - sponsor_z, stock_z, premium_z, contested_z: composition shifts vs the sector's own history.
    - heat: cross-sectional mean of z-scored flow_z_smooth, count_mom, premium_z,
      contested_z. It measures how hot/crowded a sector is, *not* a validated
      return forecast.
    - provisional: True for the last `provisional_months` (backfill risk).
    - sparse: True when the sector's trailing 12m median count is below 3.
    """
    wide = lambda c: panel[c].unstack('sector')
    z, expected = flow_surprise(panel)
    zw = z.unstack('sector')
    sig = pd.DataFrame(index=panel.index)
    sig['n_deals'] = panel['n_deals']
    sig['expected_n'] = expected
    sig['flow_z'] = z
    sig['flow_z_smooth'] = zw.ewm(halflife=smooth_halflife, min_periods=1).mean().stack(future_stack=True)
    sig['count_mom'] = ewm_momentum(wide('n_deals')).stack(future_stack=True)
    sig['value_mom'] = ewm_momentum(wide('value_w')).stack(future_stack=True)

    # Composition shares: smooth over 3 months first, since monthly shares on
    # a handful of deals are mostly noise.
    for name, col in [('sponsor_z', 'sponsor_share'), ('stock_z', 'stock_share'),
                      ('premium_z', 'premium_median'), ('contested_z', 'contested_share')]:
        w = wide(col).rolling(3, min_periods=1).mean()
        sig[name] = trailing_z(w).stack(future_stack=True)

    comps = ['flow_z_smooth', 'count_mom', 'premium_z', 'contested_z']
    xs = sig[comps].groupby(level='month').transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    sig['heat'] = xs.mean(axis=1, skipna=True)

    months = sig.index.get_level_values('month')
    last = months.max()
    sig['provisional'] = months > (last - provisional_months)
    med12 = wide('n_deals').rolling(12, min_periods=6).median().stack(future_stack=True)
    sig['sparse'] = med12.reindex(sig.index) < 3
    return sig


def latest_snapshot(sig, include_provisional=False):
    """Most recent reliable month, ranked by heat."""
    s = sig if include_provisional else sig[~sig['provisional']]
    m = s.index.get_level_values('month').max()
    snap = s.xs(m, level='month').sort_values('heat', ascending=False)
    snap.attrs['month'] = str(m)
    return snap


def flow_correlation(panel, min_months=36):
    """Correlation of month-on-month changes in log counts (not levels).

    Level correlations between trending series are mostly spurious. Differences
    show whether two sectors' surprises actually move together.
    """
    n = panel['n_deals'].unstack('sector')
    d = np.log1p(n).diff()
    d = d.loc[:, d.notna().sum() >= min_months]
    return d.corr()


def information_coefficient(sig, returns, signal_col='heat', horizon=3):
    """Monthly cross-sectional Spearman IC of a signal vs forward sector returns.

    returns: DataFrame (PeriodIndex month x sector) of simple monthly returns,
    e.g. from the ETFs in SECTOR_ETF_CANDIDATES. The signal at month t is
    compared with the compounded return over months t+1..t+horizon, so there is
    no overlap with the signal month. Overlapping horizons autocorrelate the IC
    series; the t-stat here is naive, so treat |t| < 3 as noise.
    """
    s = sig[signal_col].unstack('sector')
    r = returns.reindex(columns=s.columns)
    fwd = np.log1p(r).rolling(horizon).sum().shift(-horizon)
    ics = []
    for m in s.index.intersection(fwd.index):
        a, b = s.loc[m], fwd.loc[m]
        ok = a.notna() & b.notna()
        if ok.sum() >= 5:
            ics.append((m, a[ok].rank().corr(b[ok].rank())))
    ic = pd.Series(dict(ics), name='ic')
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 2 else np.nan
    return ic, {'mean_ic': ic.mean(), 'ic_ir': ic.mean() / ic.std(ddof=1) if len(ic) > 2 else np.nan,
                'naive_t': t, 'n_months': len(ic), 'horizon': horizon}
