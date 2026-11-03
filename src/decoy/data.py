"""Phase 2 data access (D28): processed npz -> z-scored windows. Only train/val splits can be opened."""
from __future__ import annotations
import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "experiment.toml"
SPLITS = {"train": {"train"}, "val": {"val", "validation"}}


def stop(msg):
    raise SystemExit("STOP: " + msg)


def load_cfg():
    with open(CFG_PATH, "rb") as f:
        return tomllib.load(f)


def cfg_hash():
    return hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()[:12]


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@dataclass
class Plant:
    name: str
    tr: np.ndarray          # (N, W+H, nc+ne+nr) float32, column order [command, exogenous, response]
    va: np.ndarray
    nc: int
    ne: int
    nr: int
    W: int
    H: int
    resp_names: list
    info: dict = field(default_factory=dict)


def load_plant(cfg, name, smoke=False):
    T = cfg["train"][name]
    proc = cfg.get("paths", {}).get("processed")
    if not proc:
        stop("configs/experiment.toml has no [paths] processed")
    base = ROOT / proc / name
    parts = {"train": list(T["files_train"]), "val": list(T["files_val"])}
    if set(parts["train"]) & set(parts["val"]):
        stop(f"{name}: train/val file overlap")
    recs, info = {}, {"files": {}}
    for part, fl in parts.items():
        for f in fl:
            p = base / f"{f}.npz"
            if not p.exists():
                stop(f"missing {p}")
            z = np.load(p, allow_pickle=False)
            sp = str(z["split"])
            if sp not in SPLITS[part]:
                stop(f"{p.name} has split '{sp}', expected one of {sorted(SPLITS[part])}")
            recs[f] = {k: z[k] for k in ("X", "columns", "roles", "normal")}
            info["files"][f] = {"part": part, "split": sp, "rows": int(len(z["X"])), "sha256": file_sha(p)}
    ref = recs[parts["train"][0]]
    for f, r in recs.items():
        if not (np.array_equal(r["columns"], ref["columns"]) and np.array_equal(r["roles"], ref["roles"])):
            stop(f"{f}: columns/roles differ from {parts['train'][0]}")
    roles, cols = ref["roles"].astype(str), ref["columns"].astype(str)
    ic, ie, ir = (np.flatnonzero(roles == k) for k in ("command", "exogenous", "response"))
    if len(ic) == 0 or len(ir) == 0:
        stop(f"{name}: no command or no response columns")
    order = np.concatenate([ic, ie, ir])
    raw = np.concatenate([recs[f]["X"][:, order] for f in parts["train"]]).astype(np.float64)
    mu, sd = raw.mean(0), raw.std(0)
    del raw
    if (sd < 1e-8).any():
        stop(f"{name}: zero-variance modelled tags in training {cols[order][sd < 1e-8].tolist()}")
    W, H, stride = int(T["warm"]), int(T["horizon"]), int(T["stride"])
    L = W + H
    edge_files, er = set(T["edge_files"]), int(T["edge_rows"])
    cap = int(cfg["train"]["smoke"]["max_windows"])

    def wins(part):
        out = []
        for f in parts[part]:
            r = recs[f]
            Xn = ((r["X"][:, order].astype(np.float64) - mu) / sd).astype(np.float32)
            if not np.isfinite(Xn).all():
                stop(f"{f}: non-finite values")
            v = r["normal"].astype(bool).copy()
            if f in edge_files and er > 0:
                v[:er] = False
                v[-er:] = False
            bad = np.concatenate([[0], np.cumsum(~v)])
            s = np.arange(0, len(Xn) - L + 1, stride)
            s = s[(bad[s + L] - bad[s]) == 0]
            info["files"][f]["windows"] = int(len(s))
            out += [Xn[i:i + L] for i in s]
        if not out:
            stop(f"{name} {part}: no valid windows")
        A = np.stack(out)
        if smoke and len(A) > cap:
            A = A[np.unique(np.linspace(0, len(A) - 1, cap).astype(int))]
        return A

    tr, va = wins("train"), wins("val")
    info.update({"columns": cols[order].tolist(), "nc": int(len(ic)), "ne": int(len(ie)), "nr": int(len(ir)),
                 "mu": mu.tolist(), "sd": sd.tolist(), "warm": W, "horizon": H, "stride": stride,
                 "windows_train": int(len(tr)), "windows_val": int(len(va)), "smoke": bool(smoke)})
    return Plant(name, tr, va, len(ic), len(ie), len(ir), W, H, cols[ir].tolist(), info)
