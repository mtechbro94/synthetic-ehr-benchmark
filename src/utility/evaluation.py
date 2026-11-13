"""Downstream utility evaluation: TSTR/TRTS and discriminative scores."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

MODELS = {
    'logistic': LogisticRegression(max_iter=1000),
    'rf': RandomForestClassifier(n_estimators=200, random_state=0),
}


def _as_array(X):
    return np.asarray(X, dtype=float)


def _pick(model_name):
    if isinstance(model_name, dict):
        return model_name
    return {'kind': model_name}


def _fit_model(kind, X, y):
    if kind == 'logistic':
        return LogisticRegression(max_iter=1000).fit(X, y)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
    raise ValueError(f'unknown model kind: {kind}')


def tstr(real_train, real_test, synth_train, target='label', model='logistic'):
    """Train on Synthetic, Test on Real (TSTR).

    Returns metrics for each requested model kind, evaluated on real test data.
    """
    kinds = [model] if isinstance(model, str) else list(model)
    Xs = _as_array(synth_train[0] if isinstance(synth_train, tuple) else synth_train)
    ys = _as_array(synth_train[1] if isinstance(synth_train, tuple) else synth_train)
    Xr_tr, yr_tr = real_train
    Xr_te, yr_te = real_test
    out = {}
    for kind in kinds:
        clf = _fit_model(kind, Xs, ys)
        pred = clf.predict(Xr_te)
        scores = clf.predict_proba(Xr_te)[:, 1] if hasattr(clf, 'predict_proba') else None
        entry = {'accuracy': float(accuracy_score(yr_te, pred)), 'f1': float(f1_score(yr_te, pred))}
        if scores is not None and len(np.unique(yr_te)) == 2:
            entry['auc'] = float(roc_auc_score(yr_te, scores))
        out[kind] = entry
    return out


def trts(real_train, real_test, synth_test, model='logistic'):
    """Train on Real, Test on Synthetic (TRTS): agreement between the real-trained
    model's predictions and the synthetic data's own labels, a proxy for how well
    synthetic data preserves the conditional structure of the real data.
    """
    kinds = [model] if isinstance(model, str) else list(model)
    Xs = _as_array(synth_test[0] if isinstance(synth_test, tuple) else synth_test)
    ys = _as_array(synth_test[1] if isinstance(synth_test, tuple) else synth_test)
    Xr, yr = real_train
    out = {}
    for kind in kinds:
        clf = _fit_model(kind, Xr, yr)
        pred = clf.predict(Xs)
        entry = {'agreement': float(accuracy_score(ys, pred)), 'f1': float(f1_score(ys, pred))}
        if hasattr(clf, 'predict_proba') and len(np.unique(ys)) == 2:
            entry['auc'] = float(roc_auc_score(ys, clf.predict_proba(Xs)[:, 1]))
        out[kind] = entry
    return out


def discriminative_score(real, synth):
    """Post-hoc discriminator: can a classifier tell synthetic rows from real rows?
    0.5 AUC = indistinguishable. Reports AUC and the linear discriminative distance.
    """
    X = np.vstack([_as_array(real), _as_array(synth)])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    clf = LogisticRegression(max_iter=1000).fit(X[: len(y) // 2], y[: len(y) // 2])
    scores = clf.predict_proba(X[len(y) // 2:])[:, 1]
    yh = y[len(y) // 2:]
    auc = float(roc_auc_score(yh, scores))
    return {'auc': auc, 'discriminative_distance': abs(auc - 0.5)}

# Back-compat aliases: earlier revisions exported tstr_auc/trts_auc.
tstr_auc = tstr
trts_auc = trts
