"""Deal-flow signal extraction for the CIQ M&A dataset. See docs/METHODOLOGY.md."""
from .features import add_features
from .load import clean, load_xlsx
from .sectors import DEEPTECH, buyer_home_sector, classify
from .signals import build_signals, drop_partial_tail, flow_correlation, latest_snapshot, sector_panel


def run(df_raw, gold=None):
    """Full pipeline on a raw CIQ DataFrame. Returns a dict of outputs."""
    df = add_features(clean(df_raw))
    labels, diag, model = classify(df, gold=gold)
    df = df.join(labels)
    df['buyer_home_sector'] = buyer_home_sector(df)
    df = drop_partial_tail(df)
    panel, market = sector_panel(df)
    sig = build_signals(panel)
    return {'deals': df, 'panel': panel, 'market': market, 'signals': sig,
            'classifier_diag': diag, 'model': model}
