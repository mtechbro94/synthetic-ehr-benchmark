"""Clinical downstream task evaluation for synthetic EHR/waveform data.

Wraps the cohort construction from src.data.cohort (48h deterioration label,
sepsis flag) into reusable evaluation tasks so TSTR/TRTS report numbers that
matter clinically, not generic classifier scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from .evaluation import tstr, trts


def deterioration_task(features: pd.DataFrame, label_col: str = 'deterioration_48h'):
    """48h deterioration prediction task from a feature frame."""
    cols = [c for c in features.columns if c != label_col]
    X = features[cols].select_dtypes(include=[np.number]).fillna(0).to_numpy()
    y = features[label_col].to_numpy()
    return X, y


def evaluate_task(real_train, real_test, synth_train, model='logistic'):
    """TSTR + TRTS for one downstream task, with baseline (real-only) AUC."""
    baseline = _real_only_auc(real_train, real_test)
    return {
        'baseline_real_only': baseline,
        'tstr': tstr(real_train, real_test, synth_train, model=model),
        'trts': trts(real_train, real_test, synth_train, model=model),
    }


def _real_only_auc(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def subgroup_utility(features, synth, group_col, label_col='deterioration_48h'):
    """Per-subgroup TSTR AUC: surfaces utility gaps that aggregate scores hide,
    feeding the fairness suite in src.fairness.
    """
    out = {}
    for g, idx in features.groupby(group_col).groups.items():
        sub = features.loc[idx]
        mask = np.isin(synth.index, idx) if hasattr(synth, 'index') else slice(None)
        try:
            X, y = deterioration_task(sub, label_col)
            Xs, ys = deterioration_task(synth[mask] if hasattr(synth, 'loc') else synth, label_col)
            res = tstr((X, y), (X, y), (Xs, ys), model='logistic')
            out[str(g)] = res['logistic']
        except Exception as exc:
            out[str(g)] = {'error': str(exc)}
    return out
