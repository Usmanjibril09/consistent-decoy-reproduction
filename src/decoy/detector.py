"""Attacker-side physics-rule detector (PROTOCOL 7, E6; D35). Rebuilt from the D35 record (original fitting code not kept).
Rules fitted by least squares on real training windows in physical units, using every real window step the rule's lag allows
(warm-up included); kept if validation R2 >= 0.25 (same sample definition). Threshold per rule = p99.9 over training windows
of the per-window maximum |residual| across the free-run steps; a window is detected if any kept rule exceeds its threshold.
Not enforced by C2. fit() takes a Plant from data.load_plant (training/validation only)."""
import numpy as np

VALVES = ["P1_FCV01Z", "P1_FCV02Z", "P1_FCV03Z", "P1_LCV01Z", "P1_PCV01Z", "P1_PCV02Z"]
FLOW_PAIRS = [("P1_FT01", "P1_FT01Z"), ("P1_FT02", "P1_FT02Z"), ("P1_FT03", "P1_FT03Z")]
LAGS = [0, 1, 2, 5, 10, 20, 30, 45, 60]          # upper bound = HAI warm-up
RATES = [("PitchDot", "Pitch"), ("YawDot", "Yaw")]
KS = [1, 2, 5, 10, 20]                           # centred difference (x[t+k]-x[t-k]) * fs / 2k
KEEP_R2, Q = 0.25, 0.999


def physical(P, X):
    return X.astype(np.float64) * np.asarray(P.info["sd"]) + np.asarray(P.info["mu"])


def _r2(y, yh):
    y, yh = np.ravel(y), np.ravel(yh)
    return float(1.0 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def _first(rule):
    """First window step at which the rule is computable from real data."""
    return rule["lag"] if rule["kind"] == "flow" else rule["k"]


def _io(rule, cols, Xp, start):
    """Design (N, S, p+1) and target (N, S) for window steps start..end of physical windows Xp (N, L, C)."""
    L = Xp.shape[1]
    ti = cols.index(rule["target"])
    if rule["kind"] == "flow":
        lag, vi = rule["lag"], [cols.index(v) for v in rule["inputs"]]
        A, y = Xp[:, start - lag:L - lag][..., vi], Xp[:, start:, ti]
    else:
        k, si = rule["k"], cols.index(rule["inputs"][0])
        t = np.arange(start, L - k)
        A = ((Xp[:, t + k, si] - Xp[:, t - k, si]) * rule["fs"] / (2 * k))[..., None]
        y = Xp[:, t, ti]
    return np.concatenate([A, np.ones(A.shape[:2] + (1,))], -1), y


def _fit_one(rule, cols, tr, va, W):
    s0 = _first(rule)
    A, y = _io(rule, cols, tr, s0)
    coef = np.linalg.lstsq(A.reshape(-1, A.shape[-1]), y.ravel(), rcond=None)[0]
    r = dict(rule, coef=coef.tolist(), tr_r2=_r2(y, A @ coef))
    Af, yf = _io(r, cols, tr, W)
    r["thr"] = float(np.quantile(np.abs(yf - Af @ coef).max(1), Q))
    Av, yv = _io(r, cols, va, s0)
    r["va_r2"] = _r2(yv, Av @ coef)
    Avf, yvf = _io(r, cols, va, W)
    r["va_fa"] = float((np.abs(yvf - Avf @ coef).max(1) > r["thr"]).mean())
    return r


def fit(P, fs):
    cols, W = list(P.info["columns"]), P.W
    tr, va = physical(P, P.tr), physical(P, P.va)
    cand = []
    if all(v in cols for v in VALVES):
        for pair in FLOW_PAIRS:
            best = [max((_fit_one({"kind": "flow", "target": tg, "inputs": VALVES, "lag": g}, cols, tr, va, W)
                         for g in LAGS), key=lambda r: r["tr_r2"]) for tg in pair]
            win = max(best, key=lambda r: r["va_r2"])
            for r in best:
                r["kept"] = bool(r is win and r["va_r2"] >= KEEP_R2)
            cand += best
    if all(c in cols for pair in RATES for c in pair):
        for tg, src in RATES:
            r = max((_fit_one({"kind": "rate", "target": tg, "inputs": [src], "k": k, "fs": int(fs)}, cols, tr, va, W)
                     for k in KS), key=lambda r: r["tr_r2"])
            r["kept"] = bool(r["va_r2"] >= KEEP_R2)
            cand.append(r)
    if not cand:
        raise ValueError(f"{P.name}: no detector rule applies to columns {cols}")
    return {"plant": P.name, "W": W, "q": Q, "keep_r2": KEEP_R2, "fit_samples": "all real window steps from the rule's lag",
            "threshold_samples": "free-run steps, per-window maximum", "candidates": cand,
            "rules": [r for r in cand if r["kept"]]}


def window_max(det, cols, Xp):
    """Per-window, per-kept-rule maximum |residual| over free-run steps; Xp physical (N, L, C)."""
    out = []
    for r in det["rules"]:
        A, y = _io(r, cols, Xp, det["W"])
        out.append(np.abs(y - A @ np.asarray(r["coef"])).max(1))
    return np.stack(out, 1)


def detect(det, cols, Xp):
    return (window_max(det, cols, Xp) > np.array([r["thr"] for r in det["rules"]])).any(1)
