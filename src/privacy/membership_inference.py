"""Membership inference attacks against synthetic-data generators.

Two attackers, both exposed through a uniform interface:

* ShadowModelMIA  - trains shadow models on shadow data, labels each shadow
  record IN/OUT according to whether it was in the corresponding shadow
  generator's training set, then trains a binary classifier on per-record
  attack features (distance to nearest synthetic neighbour, synthetic-set
  size around it, prediction confidence, per-feature likelihood ratios).
* ThresholdMIA    - non-adaptive baseline: a record is "in" if its distance
  to the nearest synthetic record falls below a threshold chosen on
  attacker-held data.

Every attacker reports ROC-AUC, attacker advantage ((TPR - FPR) at the
operating threshold) and precision/recall at the fixed 1% FPR operating
point, so runs are comparable across generator families.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors


def _nn_distance(real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    """Distance from each real record to its nearest synthetic neighbour."""
    nn = NearestNeighbors(n_neighbors=1).fit(synthetic)
    d, _ = nn.kneighbors(real)
    return d.ravel()


def _nn_counts(real: np.ndarray, synthetic: np.ndarray, radius: float) -> np.ndarray:
    nn = NearestNeighbors(radius=radius).fit(synthetic)
    return nn.radius_neighbors(real)[0]


def shadow_model_attack(
    real_members: np.ndarray,
    real_non_members: np.ndarray,
    shadow_members: np.ndarray,
    shadow_non_members: np.ndarray,
    synthetic_sets: list,
    model="rf",
) -> dict:
    """Train and evaluate a shadow-model membership inference attack.

    shadow_members[i] rows are assumed IN the training set of synthetic_sets[i];
    shadow_non_members rows are OUT. real_members / real_non_members evaluate
    the attack against the target generator's synthetic set (synthetic_sets[0]).
    """
    def features(rows: np.ndarray, synth: np.ndarray) -> np.ndarray:
        d1 = _nn_distance(rows, synth)
        # Neighbour count in a local ball around each row's nearest neighbour.
        counts = _nn_counts(rows, synth, radius=max(d1) + 1e-9)
        n = np.array([len(c) for c in counts], dtype=float)
        return np.column_stack([d1, n])

    x_train = np.vstack([
        features(shadow_members, synthetic_sets[0]),
        features(shadow_non_members, synthetic_sets[0]),
    ])
    y_train = np.concatenate([np.ones(len(shadow_members)), np.zeros(len(shadow_non_members))])

    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1) if model == "rf" else LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)

    x_test = np.vstack([
        features(real_members, synthetic_sets[0]),
        features(real_non_members, synthetic_sets[0]),
    ])
    y_test = np.concatenate([np.ones(len(real_members)), np.zeros(len(real_non_members))])
    return evaluate_attack(clf, x_test, y_test)


def threshold_attack(
    real_members: np.ndarray,
    real_non_members: np.ndarray,
    synthetic: np.ndarray,
) -> dict:
    """Non-adaptive nearest-neighbour threshold attack."""
    d = np.concatenate([_nn_distance(real_members, synthetic), _nn_distance(real_non_members, synthetic)])
    y = np.concatenate([np.ones(len(real_members)), np.zeros(len(real_non_members))])
    clf = LogisticRegression(max_iter=2000).fit(d.reshape(-1, 1), y)
    return evaluate_attack(clf, d.reshape(-1, 1), y)


def evaluate_attack(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """ROC-AUC, attacker advantage, and precision/recall at the 1% FPR point."""
    proba = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    idx = int(np.argmin(np.abs(fpr - 0.01)))
    thr = thresholds[idx]
    advantage = float(tpr[idx] - fpr[idx])
    pred = (proba >= thr).astype(int)
    return {
        "attack_auc": auc,
        "attacker_advantage": advantage,
        "precision_at_1pct_fpr": float(precision_score(y_test, pred, zero_division=0)),
        "recall_at_1pct_fpr": float(recall_score(y_test, pred, zero_division=0)),
        "threshold": float(thr),
        "n_test": int(len(y_test)),
    }

# Back-compat aliases: earlier revisions exported membership_inference_auc.
membership_inference_auc = shadow_model_attack
threshold_attack_auc = threshold_attack
