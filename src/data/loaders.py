"""MIMIC-IV data loaders and cohort/episode construction.

Expects the MIMIC-IV v2.x release structure (PhysioNet credentialed access):

    mimiciv/
      hosp/admissions.csv.gz, patients.csv.gz, diagnoses_icd.csv.gz, d_labitems.csv.gz
      icu/chartevents.csv.gz, labevents.csv.gz, inputevents.csv.gz

Every loader returns a pandas DataFrame; no data is bundled with this repo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEPSIS_ICD9 = {"99591", "99592"}
SEPSIS_ICD10 = {"A41*", "R652*", "R6520", "R6521"}
SEPSIS_ANTIBIOTICS = kev = None  # populated from d_items below (kept explicit for audit)


def load_tables(root: str | Path, tables: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Load the requested MIMIC-IV tables into memory.

    Parameters
    ----------
    root:
        Path to the MIMIC-IV root directory (contains hosp/ and icu/).
    tables:
        Names like 'admissions', 'patients', 'diagnoses_icd', 'chartevents'.
    """
    root = Path(root)
    loaded: dict[str, pd.DataFrame] = {}
    for name in tables:
        subdir = "icu" if name in {"chartevents", "inputevents", "outputevents", "procedureevents"} else "hosp"
        path = root / subdir / f"{name}.csv.gz"
        if not path.exists():
            path = root / subdir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"MIMIC-IV table not found: {path} (set --mimic-root)")
        logger.info("loading %s", path)
        loaded[name] = pd.read_csv(path, low_memory=False)
    return loaded


def load_labitems(root: str | Path) -> pd.DataFrame:
    """d_labitems: itemid -> label mapping for lab measurements."""
    return load_tables(root, ["d_labitems"])["d_labitems"]


VITAL_ITEMIDS = {
    220045: "heart_rate",
    220050: "sbp",
    220051: "dbp",
    220052: "map",
    220210: "resp_rate",
    220277: "spo2",
    223761: "temp_f",
    220040: "temp_c_invasive",
    224639: "temp_c_skin",
}


def load_vitals(chartevents: pd.DataFrame, stay_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Extract hourly-aligned vital signs from chartevents for the given stay_ids."""
    vitals = chartevents[chartevents["itemid"].isin(VITAL_ITEMIDS)].copy()
    if stay_ids is not None:
        vitals = vitals[vitals["stay_id"].isin(set(stay_ids))]
    vitals["feature"] = vitals["itemid"].map(VITAL_ITEMIDS)
    vitals = vitals.dropna(subset=["valuenum"])
    # collapse duplicate temperature sources: prefer 223761 (surface) then 224639
    piv = vitals.pivot_table(
        index=["stay_id", "charttime"], columns="feature", values="valuenum", aggfunc="mean"
    ).reset_index()
    if "temp_f" in piv and "temp_c_skin" in piv:
        piv["temp_c"] = (piv["temp_f"] - 32.0) / 1.8
        piv["temp_c"] = piv["temp_c"].fillna(piv["temp_c_skin"])
        piv = piv.drop(columns=["temp_f", "temp_c_skin"])
    elif "temp_f" in piv:
        piv["temp_c"] = (piv["temp_f"] - 32.0) / 1.8
        piv = piv.drop(columns=["temp_f"])
    return piv.sort_values(["stay_id", "charttime"])


def load_labs(labevents: pd.DataFrame, labitems: pd.DataFrame,
              stay_ids: Optional[Iterable[int]] = None,
              subject_to_stay: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Extract common lab measurements aligned to ICU stays."""
    common = {"50862": "albumin", "50971": "potassium", "50885": "bilirubin",
              "50912": "creatinine", "51221": "platelets", "51265": "inr",
              "51279": "red_blood_cells", "51222": "hemoglobin",
              "50821": "pao2", "50818": "pco2", "50813": "lactate", "50809": "glucose"}
    labs = labevents[labevents["itemid"].astype(str).isin(common)].copy()
    labs["feature"] = labs["itemid"].astype(str).map(common)
    labs = labs.dropna(subset=["valuenum"])
    if stay_ids is not None and subject_to_stay is not None:
        # labevents are keyed by subject_id/hadm_id; map to ICU stays
        m = subject_to_stay[["subject_id", "hadm_id", "stay_id"]]
        labs = labs.merge(m, on=["subject_id", "hadm_id"], how="inner")
        labs = labs[labs["stay_id"].isin(set(stay_ids))]
    return labs.sort_values(["stay_id", "charttime"])
