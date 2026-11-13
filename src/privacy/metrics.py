"""Shared helpers for the privacy attack suite: train/test membership splits
with a controlled overlap fraction, and an aggregate report across runs."""

from __future__ import annotations

import numpy as np


def make_membership_splits(real: np.ndarray, member_fraction: float = 0.5, seed: int = 0):
    """Split real records into members (used to train the target generator)
    and non-members (attacker-held), plus shadow-members/shadow-non-members
    for the shadow-model attack. Returns (members, non_members, shadow_members,
    shadow_non_members).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(real))
    n_members = int(len(real) * member_fraction)
    # Members of the target generator and of shadow generators are disjoint.
    members = real[perm[:n_members]]
    rest = real[perm[n_members:]]
    half = len(rest) // 2
    return members, rest[half:], rest[:half // 2], rest[half // 2: half]


def privacy_report(runs: list) -> dict:
    """Aggregate attack metrics across generator runs (one dict per run,
    each holding 'membership' and 'attribute' results) into a single table
    keyed by the run's 'label'.
    """
    report = {}
    for run in runs:
        label = run.get("label", "unnamed")
        row = {}
        mia = run.get("membership", {})
        row["mia_auc"] = mia.get("attack_auc")
        row["mia_advantage"] = mia.get("attacker_advantage")
        attrs = run.get("attribute", {})
        lifts = [v.get("attacker_lift") for v in attrs.values() if "attacker_lift" in v]
        row["max_attribute_lift"] = max(lifts) if lifts else None
        report[label] = row
    return report
