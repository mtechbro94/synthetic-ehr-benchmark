"""Aggregate fairness metrics for the benchmark report."""
from __future__ import annotations

import numpy as np


def fairness_summary(report: dict) -> dict:
    """Compact headline numbers from fairness_report for the master report."""
    util = report['utility_gaps']
    priv = report['privacy_risks']
    gaps = util['tstr_gap'].dropna().abs()
    excess = (priv['mia_auc'].dropna() - 0.5).abs()
    return {
        'worst_subgroup': report.get('worst_subgroup'),
        'max_utility_gap': float(gaps.max()) if len(gaps) else np.nan,
        'mean_utility_gap': float(gaps.mean()) if len(gaps) else np.nan,
        'max_privacy_excess': float(excess.max()) if len(excess) else np.nan,
        'mean_privacy_excess': float(excess.mean()) if len(excess) else np.nan,
        'n_subgroups_scored': int(len(report['tradeoff'].dropna(subset=['tradeoff']))),
    }
