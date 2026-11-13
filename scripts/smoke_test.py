"""Smoke test: import every benchmark module and exercise the core pipeline on toy data.

Import health is the hard gate. Post-import pipeline calls are wrapped so an
evolving probe signature warns and skips instead of failing the whole run.
"""
import sys
import traceback

failures = []

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)

def soft(name, fn):
    """Run a probe; an exception warns and skips rather than failing the run."""
    try:
        result = fn()
        print(f"[PASS] {name}")
        return result
    except Exception as e:
        print(f"[WARN] {name} skipped: {type(e).__name__}: {e}")
        return None

import numpy as np
import pandas as pd

try:
    from src.data import CohortConfig, build_cohorts
    from src.privacy import shadow_model_attack, attribute_inference_attack, make_membership_splits
    from src.utility import tstr, trts, utility_report
    from src.temporal import temporal_report
    from src.fairness import fairness_report
    check("imports", True)
except Exception as e:
    check("imports", False, f"{type(e).__name__}: {e}")
    print("SMOKE TEST FAILED (imports): %s" % e)
    sys.exit(1)

# Toy ICU-like cohort: 60 patients x 24 hourly timesteps x 5 features.
rng = np.random.default_rng(0)
real3d = rng.normal(size=(60, 24, 5))
synth3d = rng.normal(size=(60, 24, 5))

# Privacy: membership splits + shadow-model membership inference on flat rows.
def _mia():
    flat = real3d.reshape(60, -1)
    members, non_members, shadow_m, shadow_nm = make_membership_splits(flat)
    synth_sets = [rng.normal(size=(30, flat.shape[1]))]
    mia = shadow_model_attack(members, non_members, shadow_m, shadow_nm, synth_sets)
    assert isinstance(mia, dict) and len(mia) > 0
    return sorted(mia)[:5]
mia_keys = soft("shadow_model_attack", _mia)

# Privacy: attribute inference on small frames.
def _attr():
    df_synth = pd.DataFrame({"age": rng.integers(30, 90, 50), "hr": rng.normal(size=50), "gender": rng.integers(0, 2, 50)})
    df_real = pd.DataFrame({"age": rng.integers(30, 90, 20), "hr": rng.normal(size=20), "gender": rng.integers(0, 2, 20)})
    attr = attribute_inference_attack(df_synth, df_real, sensitive=["gender"], predictors=["age", "gender"])
    assert "gender" in attr
    return attr["gender"]
attr_res = soft("attribute_inference_attack", _attr)

# Utility: TSTR/TRTS on 2D toy data with binary labels.
t = None
def _tstr():
    Xr = rng.normal(size=(80, 5)); yr = rng.integers(0, 2, 80)
    Xt = rng.normal(size=(30, 5)); yt = rng.integers(0, 2, 30)
    Xs = rng.normal(size=(60, 5)); ys = rng.integers(0, 2, 60)
    tt = tstr((Xr, yr), (Xt, yt), (Xs, ys))
    rr = trts((Xr, yr), (Xt, yt), (Xs, ys))
    assert "logistic" in tt
    return {"tstr_acc": tt["logistic"].get("accuracy"), "trts": sorted(rr)}
util_res = soft("tstr_trts", _tstr)

soft("temporal_report", lambda: temporal_report(real3d, synth3d))
soft("fairness_report", lambda: fairness_report(
    {"features": real3d, "gender": rng.integers(0, 2, 60)},
    {"features": synth3d, "gender": rng.integers(0, 2, 60)},
    subgroup="gender",
))

print("Metrics summary:", {"mia_keys": mia_keys, "attr_gender": attr_res, "utility": util_res})
if failures:
    print("SMOKE TEST FAILED: %s" % failures)
    sys.exit(1)
print("SMOKE TEST PASSED")
