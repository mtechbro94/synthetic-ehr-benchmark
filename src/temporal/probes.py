"""Temporal realism probes for the synthetic-EHR benchmark.

Each probe compares a synthetic cohort against the real cohort it was
fitted on and returns a distance score (0 = identical temporal structure,
larger = worse), plus per-probe diagnostics.

Probes
------
1. sequence_length      -- distribution of per-observation counts per stay.
2. sampling_rate        -- gaps between consecutive vitals samples.
3. state_transition     -- first-order transition matrices over discretized
                           vitals states (hypo-, normal, hyper- per feature).
4. temporal_autocorr    -- autocorrelation of each monitored feature over
                           time within stays.
5. feature_drift        -- slow drift of feature means across ICU day 1..N.
"""

import numpy as np

DEFAULT_VITAL_ITEMIDS = {
    "heart_rate": 220045,
    "sbp": 220050,
    "dbp": 220051,
    "resp_rate": 220210,
    "spo2": 220277,
    "temperature": 223761,
}


def _stays(series_dict):
    """Yield (key, features x time matrix) for every stay with >= 2 samples."""
    for key, df in series_dict.items():
        if df is None or len(df) < 2:
            continue
        cols = [c for c in df.columns if c in DEFAULT_VITAL_ITEMIDS or c in
                set(DEFAULT_VITAL_ITEMIDS.keys())]
        if not cols:
            cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(cols) < 1:
            continue
        yield key, df[cols].to_numpy(dtype=float)


def sequence_length_probe(real_series, syn_series, bins=16):
    """Distributional distance of stay lengths (total observations)."""
    real = np.array([len(v) for _, v in _stays(real_series)], dtype=float)
    syn = np.array([len(v) for _, v in _stays(syn_series)], dtype=float)
    if real.size == 0 or syn.size == 0:
        return {"distance": None, "note": "no usable stays"}
    lo, hi = min(real.min(), syn.min()), max(real.max(), syn.max())
    if hi <= lo:
        hi = lo + 1.0
    hr, _ = np.histogram(real, bins=bins, range=(lo, hi), density=True)
    hs, _ = np.histogram(syn, bins=bins, range=(lo, hi), density=True)
    js = 0.5 * np.abs(hr - hs).sum() / (0.5 * (hr + hs + 1e-12)).sum()
    return {"distance": float(js), "real_mean_len": float(real.mean()),
            "syn_mean_len": float(syn.mean()), "real_n": int(real.size),
            "syn_n": int(syn.size)}


def sampling_rate_probe(real_series, syn_series):
    """Compare inter-sample gap statistics (median gap in normalized hours)."""
    def gaps(series_dict):
        out = []
        for _, df in _stays(series_dict):
            for col in ("charttime", "charttime_hours", "hours", "time"):
                if col in df.columns:
                    t = df[col].to_numpy(dtype=float)
                    if t.size >= 2:
                        dt = np.diff(np.sort(t))
                        dt = dt[dt >= 0]
                        out.extend(dt.tolist())
                    break
        return np.array(out, dtype=float)
    gr, gs = gaps(real_series), gaps(syn_series)
    if gr.size == 0 or gs.size == 0:
        return {"distance": None, "note": "no timestamps available"}
    mr, ms = float(np.median(gr)), float(np.median(gs))
    return {"distance": float(abs(mr - ms) / (mr + 1e-9)),
            "real_median_gap": mr, "syn_median_gap": ms}


def _discretize(x):
    """Map feature values to 3 states: 0 hypo, 1 normal, 2 hyper."""
    lo, hi = np.nanpercentile(x, 10), np.nanpercentile(x, 90)
    z = np.full_like(x, 1.0)
    z[x < lo] = 0.0
    z[x > hi] = 2.0
    return z


def state_transition_probe(real_series, syn_series):
    """L1 distance between first-order state-transition matrices (pooled)."""
    def mat(series_dict):
        M = np.zeros((3, 3))
        n_feat = 0
        for _, arr in _stays(series_dict):
            for f in range(arr.shape[1]):
                col = arr[:, f]
                col = col[~np.isnan(col)]
                if col.size < 3:
                    continue
                z = _discretize(col).astype(int)
                for a, b in zip(z[:-1], z[1:]):
                    M[a, b] += 1
            n_feat += 1
        if M.sum() == 0:
            return None
        return M / M.sum()
    Mr, Ms = mat(real_series), mat(syn_series)
    if Mr is None or Ms is None:
        return {"distance": None, "note": "insufficient series"}
    return {"distance": float(np.abs(Mr - Ms).sum()),
            "real_matrix": Mr.tolist(), "syn_matrix": Ms.tolist()}


def autocorr_probe(real_series, syn_series, max_lag=4):
    """Mean absolute autocorrelation difference per lag, averaged over features."""
    def ac(series_dict):
        accs = {lag: [] for lag in range(1, max_lag + 1)}
        for _, arr in _stays(series_dict):
            for f in range(arr.shape[1]):
                col = arr[:, f]
                col = col[~np.isnan(col)]
                if col.size < max_lag + 5:
                    continue
                c = col - col.mean()
                var = (c ** 2).mean() + 1e-12
                for lag in range(1, max_lag + 1):
                    accs[lag].append(float((c[:-lag] * c[lag:]).mean() / var))
        return {lag: (float(np.mean(v)) if v else None) for lag, v in accs.items()}
    ar, asyn = ac(real_series), ac(syn_series)
    dists = [abs(ar[l] - asyn[l]) for l in range(1, max_lag + 1)
             if ar[l] is not None and asyn[l] is not None]
    return {"distance": float(np.mean(dists)) if dists else None,
            "real_autocorr": ar, "syn_autocorr": asyn}


def feature_drift_probe(real_series, syn_series, max_day=4):
    """L1 drift of per-ICU-day feature means, averaged over features and days."""
    def drift(series_dict):
        out = []
        for _, arr in _stays(series_dict):
            # Assume rows sorted by time; split into equal day buckets by index.
            n = arr.shape[0]
            for d in range(max_day):
                seg = arr[int(n * d / max_day):int(n * (d + 1) / max_day)]
                if seg.shape[0] == 0:
                    continue
                out.append(np.nanmean(seg, axis=0))
        return np.mean(out, axis=0) if out else None
    mr, ms = drift(real_series), drift(syn_series)
    if mr is None or ms is None:
        return {"distance": None, "note": "insufficient series"}
    denom = np.abs(mr) + 1e-9
    return {"distance": float(np.mean(np.abs(mr - ms) / denom)),
            "real_day_means": mr.tolist(), "syn_day_means": ms.tolist()}


def temporal_report(real_series, syn_series):
    """Aggregate report over all temporal probes."""
    probes = {
        "sequence_length": sequence_length_probe(real_series, syn_series),
        "sampling_rate": sampling_rate_probe(real_series, syn_series),
        "state_transition": state_transition_probe(real_series, syn_series),
        "autocorrelation": autocorr_probe(real_series, syn_series),
        "feature_drift": feature_drift_probe(real_series, syn_series),
    }
    dists = [p["distance"] for p in probes.values() if p.get("distance") is not None]
    return {"probes": probes,
            "temporal_score": float(np.mean(dists)) if dists else None,
            "probes_scored": len(dists), "probes_total": len(probes)}
