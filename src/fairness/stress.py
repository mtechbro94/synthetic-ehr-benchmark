"""Subgroup fairness stress tests for the synthetic-EHR benchmark.

Quantifies how synthetic data generation reshapes privacy risk and utility
across demographic subgroups (sex, age bands, race), and whether the
privacy-utility trade-off is equitable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utility.evaluation import tstr_auc, trts_auc
from src.privacy.membership_inference import membership_inference_auc


def _subgroup_mask(df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Boolean mask for one subgroup spec, e.g. {'age_min': 65} or {'sex': 'F'}."""
    mask = np.ones(len(df), dtype=bool)
    for key, value in spec.items():
        if key.endswith('_min') and key[:-4] in df.columns:
            mask &= df[key[:-4]].to_numpy() >= value
        elif key.endswith('_max') and key[:-4] in df.columns:
            mask &= df[key[:-4]].to_numpy() <= value
        elif key in df.columns:
            mask &= df[key].to_numpy() == value
        else:
            raise KeyError(f"unknown subgroup key {key!r}")
    return mask


def subgroup_utility_gap(
    real_X, real_y, syn_X, syn_y,
    subgroup_specs: dict,
    subgroup_labels: pd.DataFrame | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """TSTR/TRTS AUC per subgroup vs. overall, returning the gap per subgroup.

    subgroup_specs maps a subgroup name (e.g. 'female', 'age>=65') to its spec;
    subgroup_labels is a DataFrame aligned with real_X/syn_X rows holding the
    columns the specs reference (sex, age, ...).
    """
    overall_tstr = tstr_auc(real_X, real_y, syn_X, syn_y, seed=seed)
    overall_trts = trts_auc(real_X, real_y, syn_X, syn_y, seed=seed)
    rows = []
    for name, spec in subgroup_specs.items():
        mask_r = _subgroup_mask(subgroup_labels, spec)
        mask_s = _subgroup_mask(subgroup_labels, spec)
        if mask_r.sum() < 20 or mask_s.sum() < 20:
            rows.append({'subgroup': name, 'n_real': int(mask_r.sum()), 'n_syn': int(mask_s.sum()),
                         'tstr_auc': np.nan, 'trts_auc': np.nan, 'tstr_gap': np.nan, 'trts_gap': np.nan,
                         'note': 'subgroup too small (<20 rows)'})
            continue
        t = tstr_auc(real_X[mask_r], real_y[mask_r], syn_X[mask_s], syn_y[mask_s], seed=seed)
        r = trts_auc(real_X[mask_r], real_y[mask_r], syn_X[mask_s], syn_y[mask_s], seed=seed)
        rows.append({'subgroup': name, 'n_real': int(mask_r.sum()), 'n_syn': int(mask_s.sum()),
                     'tstr_auc': t, 'trts_auc': r,
                     'tstr_gap': t - overall_tstr, 'trts_gap': r - overall_trts, 'note': ''})
    return pd.DataFrame(rows)


def subgroup_privacy_risk(
    real_X, syn_X, subgroup_specs: dict, subgroup_labels: pd.DataFrame, seed: int = 0,
) -> pd.DataFrame:
    """Membership-inference AUC per subgroup on the synthetic data.

    A value near 0.5 means no leakage for that subgroup; higher values mean
    the generator memorised that subgroup disproportionately.
    """
    rows = []
    for name, spec in subgroup_specs.items():
        mask_r = _subgroup_mask(subgroup_labels, spec)
        mask_s = _subgroup_mask(subgroup_labels, spec)
        if mask_r.sum() < 40 or mask_s.sum() < 40:
            rows.append({'subgroup': name, 'mia_auc': np.nan, 'note': 'subgroup too small (<40 rows)'})
            continue
        auc = membership_inference_auc(real_X[mask_r], syn_X[mask_s], seed=seed)
        rows.append({'subgroup': name, 'mia_auc': auc, 'note': ''})
    return pd.DataFrame(rows)


def privacy_utility_tradeoff(
    util_report: pd.DataFrame, priv_report: pd.DataFrame,
) -> pd.DataFrame:
    """Join per-subgroup utility gap with per-subgroup privacy risk.

    tradeoff = |tstr_gap| + |mia_auc - 0.5|: the combined inequity a subgroup
    carries across both axes; lower is a fairer generation.
    """
    df = util_report.merge(priv_report, on='subgroup', how='outer')
    df['tradeoff'] = df['tstr_gap'].abs() + (df['mia_auc'] - 0.5).abs()
    return df.sort_values('tradeoff', ascending=False).reset_index(drop=True)


def fairness_report(real_X, real_y, syn_X, syn_y, subgroup_specs, subgroup_labels, seed=0):
    """End-to-end subgroup stress test: utility gaps, privacy risks, trade-off."""
    util = subgroup_utility_gap(real_X, real_y, syn_X, syn_y, subgroup_specs, subgroup_labels, seed)
    priv = subgroup_privacy_risk(real_X, syn_X, subgroup_specs, subgroup_labels, seed)
    tradeoff = privacy_utility_tradeoff(util, priv)
    scored = tradeoff.dropna(subset=['tradeoff'])
    worst = scored.iloc[0]['subgroup'] if len(scored) else None
    return {
        'utility_gaps': util,
        'privacy_risks': priv,
        'tradeoff': tradeoff,
        'worst_subgroup': worst,
        'max_utility_gap': float(util['tstr_gap'].abs().max()) if util['tstr_gap'].notna().any() else np.nan,
        'max_privacy_excess': float((priv['mia_auc'] - 0.5).abs().max()) if priv['mia_auc'].notna().any() else np.nan,
    }
