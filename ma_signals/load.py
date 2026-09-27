"""Load and clean the Capital IQ M&A export (trans50m_2k18.xlsx schema)."""
import re

import numpy as np
import pandas as pd

COL_RENAME = {
    'All Transactions Announced Date': 'announced_date',
    'All Transactions Announced Date (Including Bids and Letters of Intent)': 'announced_date_incl_bids',
    'Target/Issuer': 'target',
    'Exchange:Ticker': 'ticker',
    'Transaction Types': 'transaction_type',
    'Transaction Status': 'status',
    'Total Transaction Value ($USDmm, Historical rate)': 'value_usd_mm',
    'Buyers/Investors': 'buyer',
    'Sellers': 'seller',
    'CIQ Transaction ID': 'transaction_id',
    'Transaction Comments': 'comments',
    'Country/Region of Incorporation [Target/Issuer]': 'target_country',
    'Country/Region of Incorporation [Sellers]': 'seller_country',
    'Target Security Types': 'security_type',
}

# Listing exchange -> country. Used to infer the *buyer's* country, which the
# export does not contain. Only covers exchanges common in CIQ tickers; unknown
# exchanges map to NaN rather than a guess.
EXCHANGE_COUNTRY = {
    'NYSE': 'United States', 'NasdaqGS': 'United States', 'NasdaqGM': 'United States',
    'NasdaqCM': 'United States', 'NYSEAM': 'United States', 'OTCPK': 'United States',
    'BATS': 'United States', 'ARCA': 'United States',
    'TSX': 'Canada', 'TSXV': 'Canada', 'CNSX': 'Canada',
    'LSE': 'United Kingdom', 'AIM': 'United Kingdom',
    'ENXTPA': 'France', 'ENXTAM': 'Netherlands', 'ENXTBR': 'Belgium', 'ENXTLS': 'Portugal',
    'XTRA': 'Germany', 'DB': 'Germany', 'SWX': 'Switzerland', 'BIT': 'Italy', 'BME': 'Spain',
    'OM': 'Sweden', 'CPSE': 'Denmark', 'OB': 'Norway', 'HLSE': 'Finland', 'WBAG': 'Austria',
    'ISE': 'Ireland', 'WSE': 'Poland',
    'TSE': 'Japan', 'SEHK': 'Hong Kong', 'SHSE': 'China', 'SZSE': 'China',
    'KOSE': 'South Korea', 'KOSDAQ': 'South Korea', 'TWSE': 'Taiwan', 'TPEX': 'Taiwan',
    'NSEI': 'India', 'BSE': 'India', 'SGX': 'Singapore', 'ASX': 'Australia', 'NZSE': 'New Zealand',
    'SET': 'Thailand', 'KLSE': 'Malaysia', 'IDX': 'Indonesia', 'PSE': 'Philippines',
    'SASE': 'Saudi Arabia', 'ADX': 'United Arab Emirates', 'DFM': 'United Arab Emirates',
    'TASE': 'Israel', 'JSE': 'South Africa', 'BOVESPA': 'Brazil', 'BMV': 'Mexico',
}

_TICKER_RE = re.compile(r'\(([A-Za-z]+):([^)]+)\)')


def parse_listing(name):
    """Return (exchange, ticker) from 'Company (EXCH:TICK)', else (NaN, NaN)."""
    if not isinstance(name, str):
        return np.nan, np.nan
    m = _TICKER_RE.search(name)
    return (m.group(1), m.group(2)) if m else (np.nan, np.nan)


def clean_name(name):
    """Strip the '(EXCH:TICK)' suffix so the same entity matches across rows."""
    if not isinstance(name, str):
        return np.nan
    return _TICKER_RE.sub('', name).strip(' ;,')


def clean(df_raw):
    """Rename, type-cast and enrich a raw CIQ export. Returns a new DataFrame."""
    df = df_raw.rename(columns=COL_RENAME).copy()
    if 'transaction_type' in df.columns:
        df = df.drop(columns='transaction_type')

    str_cols = [c for c in ('target', 'ticker', 'status', 'buyer', 'seller', 'transaction_id',
                            'comments', 'target_country', 'seller_country', 'security_type')
                if c in df.columns]
    for c in str_cols:
        df[c] = df[c].astype('object').where(df[c].notna(), np.nan)
        df[c] = df[c].replace('-', np.nan)

    for c in ('announced_date', 'announced_date_incl_bids'):
        df[c] = pd.to_datetime(df[c], errors='coerce')
    df['value_usd_mm'] = pd.to_numeric(df['value_usd_mm'], errors='coerce')
    df = df.dropna(subset=['announced_date', 'value_usd_mm']).reset_index(drop=True)

    # Multiple buyers are ';'-separated; the lead buyer drives entity features.
    df['n_buyers'] = df['buyer'].fillna('').str.count(';') + df['buyer'].notna().astype(int)
    df['lead_buyer_raw'] = df['buyer'].str.split(';').str[0].str.strip()
    df['lead_buyer'] = df['lead_buyer_raw'].map(clean_name)
    buyer_listing = df['lead_buyer_raw'].map(parse_listing)
    df['buyer_exchange'] = buyer_listing.str[0]
    df['buyer_is_listed'] = df['buyer_exchange'].notna()
    df['buyer_country_proxy'] = df['buyer_exchange'].map(EXCHANGE_COUNTRY)

    df['target_is_listed'] = df['ticker'].notna()
    df['seller_country_first'] = df['seller_country'].str.split(';').str[0].str.strip()
    df['is_asset_deal'] = df['security_type'].fillna('').str.strip().eq('Asset')

    # Days from first bid / LOI to formal announcement: a contested or long
    # pre-announcement process is itself informative.
    df['bid_lead_days'] = (df['announced_date'] - df['announced_date_incl_bids']).dt.days.clip(lower=0)

    df['log_value'] = np.log(df['value_usd_mm'])
    df['month'] = df['announced_date'].dt.to_period('M')
    return df


def load_xlsx(path):
    """Read the CIQ Excel export (row 0 is a title row) and clean it."""
    return clean(pd.read_excel(path, header=1))
