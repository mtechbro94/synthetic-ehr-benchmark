"""Attribute inference attacks against synthetic EHR generators.

An attacker who holds the synthetic data (and a slice of real records with
some attributes censored) tries to recover the censored values.

* AttributeInferenceAttack - trains one classifier per sensitive attribute
  on the synthetic set itself ("honest-but-curious": the released synthetic
  records are all the attacker has), then predicts the censored attribute
  on real held-out rows and reports accuracy / macro-F1 against the
  majority-class baseline. A model that only matches the marginal gives no
  attacker advantage; any lift above the baseline is leakage attributable
  to the generator memorising individual records.
* attribute_lift - the reported risk metric: attacker accuracy minus the
  marginal baseline, per attribute, with the baseline recorded for audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def attribute_inference_attack(
    synthetic: pd.DataFrame,
    real_test: pd.DataFrame,
    sensitive: list,
    predictors=None,
    model=None,
) -> dict:
    """For each attribute in *sensitive*, train on the synthetic rows and
    predict that attribute for real_test rows, reporting accuracy, macro-F1,
    the marginal baseline and the attacker lift above it.
    """
    if predictors is None:
        predictors = [c for c in synthetic.columns if c not in sensitive]
    clf = model or RandomForestClassifier(n_estimators=300, n_jobs=-1)
    results = {}
    for attr in sensitive:
        if attr not in synthetic.columns or attr not in real_test.columns:
            results[attr] = {"error": "attribute missing from data"}
            continue
        # The attacker trains on synthetic records with all other fields visible.
        fit_frame = synthetic.dropna(subset=[attr])
        if fit_frame.empty:
            results[attr] = {"error": "no synthetic labels for attribute"}
            continue
        clf.fit(fit_frame[predictors], fit_frame[attr])
        pred = clf.predict(real_test[predictors])
        truth = real_test[attr].dropna()
        aligned_pred = pred[real_test[attr].notna().values]
        baseline = float(pd.Series(truth).value_counts(normalize=True).iloc[0])
        acc = float(accuracy_score(truth, aligned_pred))
        results[attr] = {
            "attacker_accuracy": acc,
            "macro_f1": float(f1_score(truth, aligned_pred, average="macro", zero_division=0)),
            "marginal_baseline": baseline,
            "attacker_lift": acc - baseline,
            "n_test": int(len(truth)),
        }
    return results


def attribute_lift(results: dict) -> dict:
    """Condense an attack result to the reported risk numbers."""
    return {
        attr: {
            "attacker_lift": round(row["attacker_lift"], 4),
            "attacker_accuracy": round(row["attacker_accuracy"], 4),
            "baseline": round(row["marginal_baseline"], 4),
        }
        for attr, row in results.items()
        if "attacker_lift" in row
    }
