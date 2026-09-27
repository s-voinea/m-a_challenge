"""Structured deal features parsed from the free-text `comments` field.

Extractors run on lower-cased text through `_rx` (RE2 when installed), so the
full ~39k-row dataset takes seconds. Each feature says something about how the deal
is financed, how aggressive it is, or what the buyer is trying to achieve.
Absence of a match means "not mentioned", not "false".
"""
import re

import numpy as np
import pandas as pd

from . import _rx

# Named sponsors: matched against the lead buyer name regardless of listing.
KNOWN_SPONSORS = [
    'kkr', 'kohlberg kravis', 'blackstone', 'carlyle', 'apollo global', 'apollo management', 'tpg',
    'bain capital', 'warburg pincus', 'advent international', 'cvc capital', 'eqt ', 'eqt ab',
    'permira', 'hellman & friedman', 'thoma bravo', 'vista equity', 'silver lake', 'brookfield',
    'cinven', 'bc partners', 'ardian', 'partners group', 'l catterton', 'clayton, dubilier',
    'leonard green', 'stone point', 'gtcr', 'ares management', 'blue owl', 'insight partners',
    'general atlantic', 'francisco partners', 'platinum equity', 'hg capital', 'hgcapital',
    'cerberus', 'searchlight', 'triton', 'nordic capital', 'pai partners', 'bridgepoint',
    'macquarie', 'global infrastructure partners', 'stonepeak', 'antin', 'i squared', 'ki partners',
]
# Generic sponsor/fund name tokens: only trusted when the buyer is unlisted, so
# that e.g. listed "... Partners, L.P." operating companies are not tagged.
_SPONSOR_TOKEN_RE = (r'\b(?:capital|partners|equity|investors?|investments?|fund|funds|ventures|'
                     r'holdings?\s+l\.?p|management\s+l\.?l\.?c|advisors)\b')

_MULT = {'thousand': 1e-3, 'million': 1.0, 'mn': 1.0, 'm': 1.0, 'billion': 1e3, 'bn': 1e3, 'b': 1e3}
_MONEY = r'(?:\$|usd\s?|us\$)\s?([\d,]+(?:\.\d+)?)\s*(thousand|million|mn|billion|bn|m|b)\b'


def _money_mm(amount, unit):
    """Convert a regex (amount, unit) pair to USD millions."""
    amt = pd.to_numeric(amount.map(lambda a: a.replace(',', '') if isinstance(a, str) else np.nan), errors='coerce')
    return amt * unit.map(lambda u: _MULT.get(u, np.nan) if isinstance(u, str) else np.nan)


def _flag(text, pattern):
    return _rx.contains(text.tolist(), pattern, text.index)


def _pct(text, pattern):
    return pd.to_numeric(_rx.extract(text.tolist(), pattern, text.index)[0], errors='coerce')


def extract(df):
    """Return a DataFrame of parsed features aligned to `df.index`."""
    c = df['comments'].fillna('').astype(str).str.lower()
    f = pd.DataFrame(index=df.index)

    # --- Consideration mix: equity-funded deals cluster at valuation peaks.
    cash = _flag(c, r'\bin cash\b|\bcash consideration\b|\ball[- ]cash\b|\bper share in cash\b|\bpaid in cash\b')
    stock = _flag(c, r'\ball[- ]stock\b|\bshare exchange\b|\bshares? of (?:the )?(?:buyer|acquir|common stock)|'
                     r'\bin (?:common )?stock\b|\bstock consideration\b|\bnewly issued shares\b|\bexchange ratio\b|'
                     r'\b\d+(?:\.\d+)?\s+(?:new |ordinary |common )?shares? of\b|\bcash and stock\b|\bstock and cash\b')
    f['consideration'] = np.select([cash & stock, cash, stock], ['mixed', 'cash', 'stock'], default='unknown')

    # --- Price aggressiveness and stake.
    f['premium_pct'] = _pct(c, r'premium of (?:approximately |about |around )?(\d{1,3}(?:\.\d+)?)\s?%')
    f.loc[f['premium_pct'].isna(), 'premium_pct'] = _pct(
        c[f['premium_pct'].isna()], r'(\d{1,3}(?:\.\d+)?)\s?% premium')
    f.loc[f['premium_pct'] > 300, 'premium_pct'] = np.nan  # parse noise guard

    f['stake_pct'] = _pct(c, r'acquire (?:a |an |an additional |the remaining |remaining )?'
                             r'(\d{1,3}(?:\.\d+)?)\s?% (?:stake|interest|equity)')
    f['is_minority'] = f['stake_pct'] < 50
    f['is_remaining_stake'] = _flag(c, r'\bremaining (?:\d{1,3}(?:\.\d+)?\s?% )?(?:stake|interest|shares)\b')

    # --- Deal mechanics.
    f['is_tender_offer'] = _flag(c, r'\btender offer\b')
    f['is_hostile'] = _flag(c, r'\bunsolicited\b|\bhostile\b|\bproxy (?:fight|contest)\b')
    f['is_competing_bid'] = _flag(c, r'\bcompeting (?:bid|offer|proposal)\b|\bsuperior proposal\b|\bbidding war\b')
    f['is_take_private'] = _flag(c, r'\b(?:go|going|take|taking|taken) private\b|\bwill be delisted\b|\bdelisting\b')
    f['is_spac'] = _flag(c, r'special purpose acquisition|\bspac\b|\bblank check\b')
    f['is_merger_of_equals'] = _flag(c, r'merger of equals')
    f['is_nonbinding'] = _flag(c, r'non[- ]binding|letter of intent|memorandum of understanding|\bmou\b|'
                                  r'expression of interest|indicative offer')

    # --- Financing and regulatory friction.
    f['debt_financed'] = _flag(c, r'\bdebt financing\b|\bbridge (?:loan|facility)\b|\bterm loan\b|'
                                  r'\bcommitment letter\b|\bfinanced (?:through|by|with) (?:new )?(?:debt|borrowings|loans)\b')
    f['antitrust_review'] = _flag(c, r'\bantitrust\b|\bhart[- ]scott|\bhsr\b|\bcompetition (?:authority|commission|bureau)\b|'
                                     r'\bsamr\b|\bcma\b|european commission')
    f['national_security_review'] = _flag(c, r'\bcfius\b|national security|foreign investment review|\bfirb\b')

    fee = _rx.extract(c.tolist(), r'(?:termination|break(?:up)?|reverse termination) fee of ' + _MONEY, c.index, 2)
    f['termination_fee_mm'] = _money_mm(fee[0], fee[1])
    f['termination_fee_ratio'] = f['termination_fee_mm'] / df['value_usd_mm']
    f.loc[f['termination_fee_ratio'] > 0.5, 'termination_fee_ratio'] = np.nan

    # --- Buyer type: financial sponsor vs strategic.
    buyer = df['lead_buyer'].fillna('').str.lower()
    known = buyer.str.contains('|'.join(re.escape(s) for s in KNOWN_SPONSORS), regex=True)
    generic = buyer.str.contains(_SPONSOR_TOKEN_RE, regex=True) & ~df['buyer_is_listed']
    pe_text = _flag(c, r'\bprivate equity\b|\bportfolio company of\b|\bbacked by\b|\bfund(?:s)? managed by\b|'
                       r'\bfund(?:s)? advised by\b')
    f['is_sponsor'] = known | generic | pe_text
    f['is_consortium'] = df['n_buyers'] > 1

    # --- Geography. Buyer country is only a proxy (listing exchange).
    f['cross_border'] = (df['buyer_country_proxy'].notna() & df['target_country'].notna()
                         & (df['buyer_country_proxy'] != df['target_country']))
    f['has_buyer_country'] = df['buyer_country_proxy'].notna() & df['target_country'].notna()

    f['text_len'] = c.str.len()
    return f


def add_features(df):
    """Return `df` with the extracted features joined on."""
    return df.join(extract(df))
