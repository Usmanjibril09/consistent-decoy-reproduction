"""Established metrics only. FIT% (Ljung): 100 (1 - ||y - yhat|| / ||y - mean(y)||), per response tag."""
from __future__ import annotations
import numpy as np


def fit_percent(y, yhat):
    """y, yhat: (..., R). Returns (per-tag FIT%, mean over tags with variance). Diverged predictions give -inf."""
    R = y.shape[-1]
    y = np.asarray(y, np.float64).reshape(-1, R)
    yh = np.asarray(yhat, np.float64).reshape(-1, R)
    if y.shape != yh.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {yh.shape}")
    den = np.linalg.norm(y - y.mean(0), axis=0)
    fin = np.isfinite(yh)
    with np.errstate(all="ignore"):
        num = np.linalg.norm(np.where(fin, y - yh, 0.0), axis=0)
        f = 100.0 * (1.0 - num / den)
    f[~fin.all(0)] = -np.inf
    f[den < 1e-12] = np.nan
    ok = ~np.isnan(f)
    return f, (float(f[ok].mean()) if ok.any() else float("nan"))
