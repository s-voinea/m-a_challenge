"""Sector classification: scored keyword rules -> weak-supervised text model.

Pipeline
1. Scored rules. Every sector's patterns are counted over the target name
   (weight 2) and the deal comments (weight 1). The buyer name is excluded:
   "Blackstone Real Estate" buying a chip maker is not a real-estate deal.
   A label is kept only if it beats the runner-up by a margin. This replaces
   first-match-wins, whose result depended on dict order.
2. Weak supervision. A TF-IDF + logistic model is trained on rule-labelled
   rows *with the rule keywords masked out*, so it has to learn from context
   words ("clinical-stage", "wafer fab", "tenant"...) instead of re-learning the
   rules. The rule-unlabelled rows contain no keywords either, so masking also
   makes training rows look like the rows being predicted.
3. Optional LLM pass (`select_for_llm` / `build_llm_prompt`): label only the
   few hundred rows the model is least sure about, spread across clusters,
   then retrain with those gold labels. That costs about 1-3% of the tokens
   of LLM-labelling all ~39k deals.
"""
import numpy as np
import pandas as pd

from . import _rx

TAXONOMY = {
    'Quantum Computing': [r'quantum comput', r'\bqubits?\b', r'quantum (?:technolog|sensing|software|processor)'],
    'AI & Machine Learning': [r'artificial intelligence', r'machine learning', r'deep learning', r'neural network',
                              r'generative ai', r'\bllms?\b', r'computer vision', r'natural language processing',
                              r'\bai[- ](?:powered|driven|based|platform|software|company|startup)\b'],
    'Semiconductors & Chips': [r'semiconductor', r'\bchips?\b', r'\bwafers?\b', r'\bfoundr(?:y|ies)\b',
                               r'integrated circuit', r'\bfabless\b', r'\bfpgas?\b', r'lithograph', r'\bmems\b',
                               r'\banalog (?:and|&) mixed[- ]signal\b'],
    'Cybersecurity': [r'cyber ?security', r'information security', r'threat (?:detection|intelligence)',
                      r'endpoint (?:security|protection)', r'identity (?:and access )?management', r'\bfirewalls?\b',
                      r'zero[- ]trust', r'security operations'],
    'Aerospace & Defense': [r'aerospace', r'\bdefen[cs]e (?:contractor|electronics|systems|industry|company)',
                            r'\bmilitary\b', r'\bmissiles?\b', r'\bradars?\b', r'\bsatellites?\b', r'space launch',
                            r'unmanned aerial', r'\bavionics\b', r'\baircraft (?:component|part|engine)'],
    'Robotics & Automation': [r'\brobot', r'industrial automation', r'factory automation', r'\bautonomous (?:vehicle|driving|system)',
                              r'\bdrones?\b', r'\bcobots?\b', r'\blidar\b', r'motion control', r'machine vision'],
    'Biotech & Pharma': [r'biotech', r'pharmaceutical', r'\bpharma\b', r'drug (?:discovery|development|candidate)',
                         r'gene (?:therapy|editing)', r'cell therapy', r'\bgenomic', r'\bbiologics?\b', r'\boncology\b',
                         r'\btherapeutics?\b', r'clinical[- ]stage', r'clinical trials?', r'biopharm', r'\bvaccines?\b',
                         r'\bantibod(?:y|ies)\b', r'\bbiosimilar', r'\bcdmo\b', r'\bcro\b'],
    'Advanced Materials': [r'advanced materials?', r'nanomaterial', r'\bgraphene\b', r'metamaterial',
                           r'specialty chemical', r'\bcomposites?\b', r'carbon fib(?:er|re)', r'rare earth'],
    'Clean Energy & EV': [r'\bsolar\b', r'wind (?:energy|farm|power|turbine)', r'\brenewable', r'clean energy',
                          r'\bbattery\b|\bbatteries\b', r'energy storage', r'green hydrogen|\bhydrogen\b', r'fuel cells?',
                          r'electric vehicles?', r'\bevs?\b', r'charging (?:station|infrastructure|network)',
                          r'\bphotovoltaic'],
    'Fintech': [r'\bfintech\b', r'payment (?:processing|platform|solutions?)', r'\bblockchain\b', r'\bcrypto',
                r'digital (?:banking|payments?|wallet)', r'\binsurtech\b', r'\bneobank', r'buy now,? pay later'],
    'Healthcare & Medtech': [r'medical (?:device|technolog|equipment|imaging)', r'\bmedtech\b', r'\bdiagnostics?\b',
                             r'\btelemedicine\b|\btelehealth\b', r'digital health', r'\bsurgical\b', r'\bimplants?\b',
                             r'\bhospitals?\b', r'healthcare services', r'\bclinics?\b'],
    'Software & Cloud': [r'\bsaas\b', r'software[- ]as[- ]a[- ]service', r'cloud[- ](?:based|computing|platform|software|services)',
                         r'enterprise software', r'\bsoftware (?:company|provider|developer|platform|solutions)',
                         r'\berp\b', r'\bcrm\b', r'\bdevops\b', r'data analytics', r'\bdata ?cent(?:er|re)s?\b'],
    'Telecom & 5G': [r'\btelecom', r'\b5g\b', r'wireless (?:network|infrastructure|spectrum)', r'fib(?:er|re)[- ]optic',
                     r'\bbroadband\b', r'mobile (?:operator|network)', r'\btowers?\b portfolio|\bcell towers?\b', r'\bspectrum\b'],
    'Real Estate': [r'real estate', r'\breits?\b', r'(?:office|industrial|retail|residential|commercial|logistics) propert',
                    r'\bapartments?\b', r'\bhotels?\b', r'\bresorts?\b', r'shopping (?:center|centre|mall)',
                    r'\bwarehouses?\b', r'square (?:feet|foot|metres|meters)', r'\bsq\.? ?ft\b', r'\bland parcel'],
    'Oil & Gas': [r'\boil\b', r'natural gas', r'\bpetroleum\b', r'\brefiner(?:y|ies)\b',
                  r'\b(?:oil|gas|crude|natural gas|midstream|liquids|lng) pipelines?\b', r'\bpipeline (?:operator|company|system|network|assets)\b',
                  r'\bupstream\b', r'\bmidstream\b', r'\blng\b', r'\bdrilling\b', r'\bexploration and production\b',
                  r'\bboe\b|barrels? of oil'],
    'Mining & Metals': [r'\bmining\b', r'\bgold\b', r'\bcopper\b', r'\blithium\b', r'iron ore', r'\bmineral', r'\bsteel\b',
                        r'\bsmelt', r'\bnickel\b', r'\bzinc\b', r'\bcoal\b', r'\bmines?\b'],
    'Financial Services': [r'\bbanks?\b', r'\bbanking\b', r'\binsurance\b|\binsurer\b|\breinsur', r'asset management',
                           r'wealth management', r'\bbrokerage\b', r'investment management', r'\blending\b',
                           r'property and casualty|p&c'],
    'Consumer & Retail': [r'consumer (?:products|goods|brands?)', r'\bretail(?:er|ers)?\b', r'e-?commerce',
                          r'\bfood\b', r'\bbeverages?\b', r'\bapparel\b', r'\bluxury\b', r'\brestaurants?\b',
                          r'\bcosmetics?\b', r'\bpersonal care\b'],
}

DEEPTECH = ['Quantum Computing', 'AI & Machine Learning', 'Semiconductors & Chips', 'Cybersecurity',
            'Aerospace & Defense', 'Robotics & Automation', 'Biotech & Pharma', 'Advanced Materials',
            'Clean Energy & EV', 'Healthcare & Medtech', 'Software & Cloud']

# Phrases that fire a rule but mean something else. Removed before scoring.
NEGATIVE_PHRASES = [r'intellectual property', r'property and casualty', r'property & casualty',
                    r'property, plant and equipment', r'chief (?:ai|technology) officer',
                    r'bank(?:s)? (?:loan|facility|financing|debt)', r'financed (?:by|through) (?:a )?banks?',
                    r'(?:advisor|adviser)s?\b[^.]{0,80}\.', r'legal (?:advisor|counsel)[^.]{0,120}\.']

UNCLASSIFIED = 'Unclassified'
_SECTORS = list(TAXONOMY)
# One combined pattern, one named group per sector: a single pass per text.
_COMBINED = '|'.join(f'(?P<g{i}>' + '|'.join(TAXONOMY[s]) + ')' for i, s in enumerate(_SECTORS))
_NEG = '|'.join(NEGATIVE_PHRASES)
_ALL_SEEDS = '|'.join(p for ps in TAXONOMY.values() for p in ps)
_TICKER = r'\([a-z]+:[^)]+\)'


def rule_scores(df):
    """Return a (n_rows x n_sectors) DataFrame of weighted keyword hit counts."""
    name = _rx.sub(_rx.lower(df['target']), _NEG)
    text = _rx.sub(_rx.lower(df['comments']), _NEG)
    k = len(_SECTORS)
    scores = 2 * _rx.count_groups(name, _COMBINED, k) + _rx.count_groups(text, _COMBINED, k)
    return pd.DataFrame(scores, index=df.index, columns=_SECTORS)


def rule_labels(scores, min_score=1, min_margin=1):
    """Top-scoring sector if it clears `min_score` and leads by `min_margin`."""
    arr = scores.to_numpy()
    order = np.argsort(-arr, axis=1, kind='stable')
    top = arr[np.arange(len(arr)), order[:, 0]]
    second = arr[np.arange(len(arr)), order[:, 1]]
    ok = (top >= min_score) & (top - second >= min_margin)
    return pd.Series(np.where(ok, scores.columns.to_numpy()[order[:, 0]], UNCLASSIFIED), index=scores.index)


def model_text(df, mask_seeds=True):
    """Text fed to the model: target name + comments, seed keywords optionally masked."""
    text = _rx.lower(df['target'].fillna('').astype(str) + ' . ' + df['comments'].fillna('').astype(str))
    text = _rx.sub(text, _TICKER)  # tickers are IDs, not semantics
    if mask_seeds:
        text = _rx.sub(text, _ALL_SEEDS)
    return pd.Series(text, index=df.index)


def make_text_model(random_state=0):
    """Unfitted TF-IDF (word 1-2 grams) + multinomial logistic regression (SGD).

    SGD keeps fitting to a few seconds on ~20k docs x 200k features.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.4, sublinear_tf=True,
                          strip_accents='unicode', max_features=200_000,
                          token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b')
    clf = SGDClassifier(loss='log_loss', alpha=2e-5, class_weight='balanced', max_iter=50, tol=1e-4,
                        random_state=random_state)
    return make_pipeline(vec, clf)


def classify(df, prob_threshold=0.55, gold=None, cv_folds=5, random_state=0):
    """Assign `sector`, `sector_source` (rule/model/gold/none) and `sector_conf`.

    gold: optional Series indexed like df with human/LLM labels. They override
    rule labels and get 5x weight in training.

    Returns (labels DataFrame, diagnostics dict). The diagnostics include
    out-of-fold agreement between model and rules on rule-labelled rows. That
    number is an upper bound on accuracy, not an accuracy: there is no ground
    truth here without a gold set.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    scores = rule_scores(df)
    rules = rule_labels(scores)
    train_labels = rules.copy()
    weights = pd.Series(1.0, index=df.index)
    if gold is not None:
        gold = gold.dropna()
        train_labels.loc[gold.index] = gold
        weights.loc[gold.index] = 5.0

    text = model_text(df)
    is_train = train_labels.ne(UNCLASSIFIED)
    counts = train_labels[is_train].value_counts()
    usable = counts[counts >= max(cv_folds, 10)].index  # too-rare classes can't be learned
    is_train &= train_labels.isin(usable)

    diag = {'rule_coverage': float(rules.ne(UNCLASSIFIED).mean()), 'n_train': int(is_train.sum()),
            'classes': list(usable)}

    fit_params = {'sgdclassifier__sample_weight': weights[is_train].to_numpy()}
    if cv_folds and is_train.sum() > 200:
        skf = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
        oof = cross_val_predict(make_text_model(random_state), text[is_train], train_labels[is_train],
                                cv=skf, params=fit_params)
        diag['oof_agreement_with_rules'] = float((oof == train_labels[is_train].to_numpy()).mean())

    model = make_text_model(random_state).fit(text[is_train], train_labels[is_train], **fit_params)
    proba = model.predict_proba(text)
    classes = model.classes_
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)

    out = pd.DataFrame(index=df.index)
    out['sector_rule'] = rules
    out['sector_model'] = pred
    out['sector_model_conf'] = conf
    use_rule = rules.ne(UNCLASSIFIED)
    use_model = ~use_rule & (conf >= prob_threshold)
    out['sector'] = np.where(use_rule, rules, np.where(use_model, pred, UNCLASSIFIED))
    out['sector_source'] = np.where(use_rule, 'rule', np.where(use_model, 'model', 'none'))
    if gold is not None:
        out.loc[gold.index, 'sector'] = gold
        out.loc[gold.index, 'sector_source'] = 'gold'
    rule_top = scores.max(axis=1)
    out['sector_conf'] = np.where(use_rule, np.minimum(1.0, 0.5 + 0.1 * rule_top), np.where(use_model, conf, 0.0))
    out['is_deeptech'] = out['sector'].isin(DEEPTECH)

    diag['final_coverage'] = float(out['sector'].ne(UNCLASSIFIED).mean())
    diag['model_added'] = int(use_model.sum())
    return out, diag, model


def select_for_llm(df, labels, model, n=400, n_clusters=40, random_state=0):
    """Pick rows for LLM/human labelling: low-confidence rows spread over clusters.

    Uncertainty sampling alone picks near-duplicates. Clustering the TF-IDF space
    (TruncatedSVD -> MiniBatchKMeans) and taking the least-confident rows from
    each cluster covers the unlabelled space at the same cost.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    pool = labels.index[labels['sector_source'].ne('rule')]
    if len(pool) == 0:
        return pool
    X = model[0].transform(model_text(df.loc[pool]))
    k = min(n_clusters, max(1, len(pool) // 5))
    Z = TruncatedSVD(min(100, X.shape[1] - 1), random_state=random_state).fit_transform(X)
    cl = MiniBatchKMeans(k, random_state=random_state, n_init=3).fit_predict(Z)
    cand = pd.DataFrame({'conf': labels.loc[pool, 'sector_model_conf'].to_numpy(), 'cl': cl}, index=pool)
    per = int(np.ceil(n / k))
    picked = cand.sort_values('conf').groupby('cl', group_keys=False).head(per)
    return picked.sort_values('conf').index[:n]


def build_llm_prompt(row, sectors=None):
    """One-deal classification prompt. Send it with any LLM client you like."""
    sectors = sectors or list(TAXONOMY) + [UNCLASSIFIED]
    return (
        'Classify the TARGET company of this M&A deal into exactly one sector from the list. '
        'Judge the target\'s business, not the buyer\'s. Reply with the sector name only.\n'
        f'Sectors: {"; ".join(sectors)}\n\n'
        f'Target: {row.get("target", "")}\nBuyer: {row.get("buyer", "")}\n'
        f'Deal description: {str(row.get("comments", ""))[:2000]}'
    )


def buyer_home_sector(df, sector_col='sector', min_other_deals=2):
    """Buyer's dominant target sector computed *excluding the current deal*.

    Assigning a buyer's sector from all of its own deals makes a
    buyer-sector x target-sector matrix diagonal by construction (the old
    notebook reports ~95-97% "intra-sector"). Leaving the current deal out
    removes that circularity. Buyers with fewer than `min_other_deals` other
    classified deals get NaN.
    """
    d = df[['lead_buyer', sector_col]]
    known = d[d['lead_buyer'].notna() & d[sector_col].ne(UNCLASSIFIED)]
    counts = pd.crosstab(known['lead_buyer'], known[sector_col])
    res = pd.Series(np.nan, index=df.index, dtype='object')
    rows = d.index[d['lead_buyer'].isin(counts.index)]
    c = counts.loc[d.loc[rows, 'lead_buyer']].to_numpy().copy()
    own = counts.columns.get_indexer(d.loc[rows, sector_col])  # -1 when this deal is unclassified
    has_own = own >= 0
    c[np.flatnonzero(has_own), own[has_own]] -= 1  # leave this deal out
    ok = c.sum(1) >= min_other_deals
    res.loc[rows[ok]] = counts.columns.to_numpy()[c.argmax(1)][ok]
    return res
