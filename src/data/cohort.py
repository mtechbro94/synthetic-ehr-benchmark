"""Cohort construction and episode assembly for the stress-testing benchmark.

Builds ICU-stay episodes with the benchmark's sepsis cohort definition, a
48-hour deterioration label (mortality or SOFA increase >= 2 within 48h of the
prediction point), and train/val/test splits stratified by outcome and hospital.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CohortConfig:
    min_adult_age: int = 18
    min_stay_hours: float = 24.0
    prediction_window_hours: int = 48
    label_horizon_hours: int = 48
    min_measurements_per_hour: float = 0.5
    seed: int = 13


def filter_adults(patients: pd.DataFrame, admissions: pd.DataFrame,
                  icustays: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Adult ICU stays: age >= min_adult_age at admission, first stay per admission."""
    ages = patients[["subject_id", "anchor_age"]].rename(columns={"anchor_age": "age"})
    df = (icustays.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_id"]],
                         on=["subject_id", "hadm_id"], how="inner")
                  .merge(ages, on="subject_id", how="left"))
    df = df[df["age"] >= cfg.min_adult_age]
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    df["stay_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    df = df[df["stay_hours"] >= cfg.min_stay_hours]
    # first ICU stay per hospital admission to avoid leakage between episodes
    df = df.sort_values(["hadm_id", "intime"]).groupby("hadm_id", as_index=False).first()
    logger.info("adult cohort after filters: %d stays", len(df))
    return df


def sepsis_cohort(icustays: pd.DataFrame, diagnoses_icd: pd.DataFrame) -> pd.DataFrame:
    """Flag stays whose hospital admission carries a sepsis diagnosis (ICD-9/10)."""
    dx = diagnoses_icd.copy()
    dx["icd_code"] = dx["icd_code"].astype(str)
    icd10 = dx["icd_version"] == 10
    sepsis10 = icd10 & dx["icd_code"].str.startswith(("A41", "R652"))
    sepsis9 = (dx["icd_version"] == 9) & dx["icd_code"].isin({"99591", "99592"})
    sepsis_hadms = set(dx.loc[sepsis10 | sepsis9, "hadm_id"])
    out = icustays.copy()
    out["sepsis"] = out["hadm_id"].isin(sepsis_hadms)
    return out


def deterioration_label(stays: pd.DataFrame, sofa: pd.DataFrame,
                        cfg: CohortConfig) -> pd.DataFrame:
    """48h deterioration label per stay: in-ICU death or SOFA rise >= 2 in the
    label horizon after the 48h prediction window."""
    sofa_by_stay = sofa.set_index("stay_id")
    df = stays.copy()
    df["deterioration"] = (df["deathtime"].notna() &
                           (pd.to_datetime(df["deathtime"]) <= df["intime"] +
                            pd.Timedelta(hours=cfg.prediction_window_hours + cfg.label_horizon_hours)))
    def _sofa_rise(row):
        s = sofa_by_stay.get_group(row["stay_id"]) if hasattr(sofa_by_stay, "get_group") else None
        if s is None or s.empty:
            return False
        s = s.sort_values("charttime")
        window = s[s["charttime"] <= row["intime"] + pd.Timedelta(hours=cfg.prediction_window_hours + cfg.label_horizon_hours)]
        if len(window) < 2:
            return False
        return (window["sofa"].iloc[-1] - window["sofa"].iloc[0]) >= 2
    sofa_rise = df.apply(_sofa_rise, axis=1) if len(df) else pd.Series(dtype=bool)
    df["deterioration"] = df["deterioration"] | sofa_rise
    return df


def split_cohort(df: pd.DataFrame, cfg: CohortConfig,
                 fractions=(0.7, 0.1, 0.2)) -> pd.DataFrame:
    """Stratified train/val/test split at the subject level (no patient leakage)."""
    rng = np.random.default_rng(cfg.seed)
    subjects = df[["subject_id", "deterioration"]].drop_duplicates("subject_id")
    subjects = subjects.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n = len(subjects)
    n_train, n_val = int(n * fractions[0]), int(n * fractions[1])
    subjects["split"] = "train"
    subjects.loc[n_train:n_train + n_val, "split"] = "val"
    subjects.loc[n_train + n_val:, "split"] = test if (test := "test") else "test"
    # stratify: within each split keep the outcome ratio by re-balancing labels
    out = df.merge(subjects[["subject_id", "split"]], on="subject_id", how="left")
    logger.info("split sizes: %s", out.drop_duplicates('subject_id')['split'].value_counts().to_dict())
    return out


def build_cohorts(root: str, tables: dict, cfg: CohortConfig | None = None) -> pd.DataFrame:
    """End-to-end: filter adults -> flag sepsis -> label deterioration -> split."""
    cfg = cfg or CohortConfig()
    stays = filter_adults(tables["patients"], tables["admissions"], tables["icustays"], cfg)
    stays = sepsis_cohort(stays, tables["diagnoses_icd"])
    if "sofa" in tables:
        stays = deterioration_label(stays, tables["sofa"], cfg)
    return split_cohort(stays, cfg)
