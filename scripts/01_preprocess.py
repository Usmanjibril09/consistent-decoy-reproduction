#!/usr/bin/env python3
"""Phase 1 - preprocessing. PROTOCOL v1 (tag protocol-v1) sections 3.1-3.3; decisions D1-D23.

Reads   data/raw (read-only), configs/experiment.toml, spec/hai2305_command_attacks.csv
Writes  data/processed/{qappd,hai,hai2204}/*.npz (raw physical units, float32, labels, normal masks)
        results/phase1/{manifest.json, stats.json, tag_roles.csv, probe_inclusion.csv,
                        qappd_anomaly_segments.csv, report.txt}; copy of report.txt -> spec/phase1_report.txt
No model fitting or tuning. Held-out files are used only for labels, masks and the protocol 3.3 command rule.
"""
import hashlib, json, platform, shutil, subprocess, sys, tomllib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = []


def say(s=""):
    print(s)
    REPORT.append(str(s))


def stop(msg):
    sys.exit(f"STOP: {msg}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def segments(y):
    d = np.diff(np.r_[0, np.asarray(y, np.int8), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def normal_mask(y, margin):
    m = np.asarray(y) == 0
    for _, e in zip(*segments(y)):
        m[e:e + margin] = False
    return m


def finite(df, name):
    a = df.to_numpy(dtype=float)
    bad = (~np.isfinite(a)).sum(axis=0)
    if bad.any():
        stop(f"{name}: non-finite values {dict(zip(df.columns[bad > 0], bad[bad > 0].tolist()))}; nothing is imputed")
    return a


def stats(X, cols):
    return {c: {"mean": float(X[:, k].mean()), "std": float(X[:, k].std()),
                "min": float(X[:, k].min()), "max": float(X[:, k].max()),
                "p1": float(np.percentile(X[:, k], 1)), "p99": float(np.percentile(X[:, k], 99))}
            for k, c in enumerate(cols)}


def save(path, log, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    log["outputs"][str(path.relative_to(ROOT))] = sha256(path)


def epoch_s(t):
    return t.to_numpy().astype("datetime64[s]").astype(np.int64)


# ---------------------------------------------------------------- QAPPD (PROTOCOL 3.2)
def qappd(cfg, log):
    q = cfg["qappd"]; fs = int(q["fs_hz"]); flag = q["flag"]; cmd = q["commands"]
    src = ROOT / cfg["paths"]["raw"] / q["dir"]; out = ROOT / cfg["paths"]["processed"] / "qappd"
    split = {f"train_{i}": "train" for i in q["train"]}
    split |= {f"train_{i}": "val" for i in q["val"]}
    split |= {f"train_{i}": "heldout" for i in q["heldout_normal"]}
    split |= {f"test_{i}": "test" for i in q["test"]}
    if len(split) != sum(len(q[k]) for k in ("train", "val", "heldout_normal", "test")):
        stop("QAPPD: a run is assigned to more than one split")
    runs, cols, order = {}, None, None
    for name in split:
        f = src / f"{name}.pkl"
        df = pd.read_pickle(f); log["raw_sha256"][str(f.relative_to(ROOT))] = sha256(f)
        if cols is None:
            cols = list(df.columns)
            miss = [c for c in cmd + [flag] if c not in cols]
            if miss: stop(f"QAPPD: missing columns {miss}")
            order = cmd + [c for c in cols if c not in cmd and c != flag]
        if list(df.columns) != cols: stop(f"QAPPD {name}: columns differ from first file")
        en = df[flag].to_numpy() > 0
        if not en.any(): stop(f"QAPPD {name}: Enable never on")
        a = int(np.argmax(en)); b = len(en) - int(np.argmax(en[::-1]))
        if not en[a:b].all():
            stop(f"QAPPD {name}: {(~en[a:b]).sum()} Enable=0 rows inside the run; trimming would break continuity")
        y = np.zeros(b - a, np.int8)
        if name.startswith("test_"):
            fl = src / f"test_label_{name[5:]}.pkl"
            lab = np.asarray(pd.read_pickle(fl)).astype(int); log["raw_sha256"][str(fl.relative_to(ROOT))] = sha256(fl)
            if len(lab) != len(en): stop(f"QAPPD {name}: label length {len(lab)} != {len(en)}")
            if set(np.unique(lab)) - {0, 1}: stop(f"QAPPD {name}: labels not binary")
            if lab[:a].any() or lab[b:].any(): stop(f"QAPPD {name}: anomaly labelled inside trimmed Enable=0 rows")
            y = lab[a:b].astype(np.int8)
        runs[name] = (finite(df.iloc[a:b][order], f"QAPPD {name}"), y, a, len(en) - b)
    roles = ["command" if c in cmd else "response" for c in order]
    st = stats(np.vstack([runs[n][0] for n, s in split.items() if s == "train"]), order)

    k = order.index(q["period_channel"]); per = []
    for n, s in split.items():
        if s != "train": continue
        x = runs[n][0][::10, k]; x = x - x.mean(); N = len(x)
        ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * N)) ** 2)[:N]; ac /= ac[0]
        lo, hi = 2 * fs // 10, 120 * fs // 10
        per.append((lo + int(np.argmax(ac[lo:hi]))) * 10 / fs)
    period = float(np.median(per)); margin = int(round(period * fs))

    for n, s in split.items():
        X, y, _, _ = runs[n]
        save(out / f"{n}.npz", log, X=X.astype(np.float32), columns=np.array(order), roles=np.array(roles),
             y=y, normal=normal_mask(y, margin), fs=np.int64(fs), split=np.array(s))

    seg = []
    for n, s in split.items():
        if s != "test": continue
        X, y, _, _ = runs[n]
        for j, (a, b) in enumerate(zip(*segments(y)), 1):
            r = {"run": n, "segment": j, "start_s": a / fs, "length_s": (b - a) / fs}
            for c in cmd:
                v = X[a:b, order.index(c)]
                r[f"{c}_maxabs"] = float(np.abs(v).max())
                r[f"{c}_share_outside_train_minmax"] = float(((v < st[c]["min"]) | (v > st[c]["max"])).mean())
            seg.append(r)

    say("=== QAPPD ===")
    say(f"channels: {len(order)} ({len(cmd)} command, {len(order) - len(cmd)} response); derived (kept as response): {q['derived']}")
    for n, s in split.items():
        X, y, a, b = runs[n]
        say(f"{n:9s} {s:8s} rows {len(X):7d}  trimmed head {a} tail {b}  anomalous {y.mean():.4f}")
    say(f"cycle period (training {q['period_channel']} autocorrelation, per run {per}) median {period:.1f} s -> margin {margin} samples")
    for c in cmd:
        say(f"training range {c}: {st[c]['min']:.3f} .. {st[c]['max']:.3f}")
    say(pd.DataFrame(seg).round(3).to_string(index=False))
    tags = [("qappd", c, r, "derived" if c in q["derived"] else "") for c, r in zip(order, roles)]
    return st, tags, seg, {"period_s": period, "margin_samples": margin, "split": split}


# ---------------------------------------------------------------- HAI (PROTOCOL 3.1, 3.3)
def read_hai(f, log):
    df = pd.read_csv(f); log["raw_sha256"][str(f.relative_to(ROOT))] = sha256(f)
    tc = [c for c in df.columns if c.strip().lower() == "timestamp"]
    if len(tc) != 1: stop(f"{f.name}: expected one timestamp column, found {tc}")
    t = pd.to_datetime(df[tc[0]])
    dt = t.diff().dt.total_seconds().to_numpy()[1:]
    if not np.all(dt == 1): stop(f"{f.name}: time step not 1 s at {int((dt != 1).sum())} rows")
    return t, df.drop(columns=tc)


def hai(cfg, log):
    h = cfg["hai"]; fs = int(h["fs_hz"]); cmd = h["commands"]
    raw = ROOT / cfg["paths"]["raw"]; out = ROOT / cfg["paths"]["processed"]
    files = {n: "train" for n in h["train"]} | {n: "val" for n in h["val"]} | {n: "test" for n in h["test"]}
    data = {n: read_hai(raw / h["dir"] / f"{n}.csv", log) for n in files}
    cols = list(data[h["train"][0]][1].columns)
    for n, (_, df) in data.items():
        if list(df.columns) != cols: stop(f"HAI {n}: columns differ from {h['train'][0]}")
    p1 = [c for c in cols if c.startswith(h["prefix"])]
    miss = [c for c in cmd if c not in p1]
    if miss: stop(f"HAI: command tags missing {miss}")
    exo = [c for c in p1 if c not in cmd and (c.endswith("D") or "SP" in c)]
    resp = [c for c in p1 if c not in cmd and c not in exo]
    drop = list(h.get("exclude_tags", []))
    if set(drop) & set(cmd): stop(f"HAI: excluded tags include command tags {sorted(set(drop) & set(cmd))}")
    unknown = [x for x in drop if x not in p1]
    if unknown: stop(f"HAI: excluded tags not in data {unknown}")
    exo = [x for x in exo if x not in drop]; resp = [x for x in resp if x not in drop]
    log["hai_excluded_tags"] = drop
    if drop: say(f"HAI excluded tags (D25): {drop}")
    order = cmd + exo + resp
    X = {n: finite(df[order], f"HAI {n}") for n, (_, df) in data.items()}
    st = stats(np.vstack([X[n] for n, s in files.items() if s == "train"]), order)
    role = {c: "command" for c in cmd} | {c: "exogenous" for c in exo} | {c: "response" for c in resp}
    for c in order:
        if st[c]["std"] == 0:
            if c in cmd: stop(f"HAI: command {c} is constant in training")
            role[c] = "constant"
    roles = np.array([role[c] for c in order])

    margin = int(h["post_attack_margin_s"] * fs); Y = {}
    for n, s in files.items():
        t = data[n][0]
        if s != "test":
            Y[n] = np.zeros(len(t), np.int8); continue
        fl = raw / h["dir"] / f"label-{n.split('-', 1)[1]}.csv"
        lab = pd.read_csv(fl); log["raw_sha256"][str(fl.relative_to(ROOT))] = sha256(fl)
        lt = pd.DatetimeIndex(pd.to_datetime(lab["timestamp"])); dt = pd.DatetimeIndex(t)
        if len(lab) != len(t):
            stop(f"HAI {n}: label rows {len(lab)} != data rows {len(t)}")
        if np.array_equal(lt.to_numpy(), dt.to_numpy()):
            mode = "exact"
        elif (lt.second == 0).all() and np.array_equal(lt.to_numpy(), dt.floor("min").to_numpy()):
            mode = "minute (D24: label timestamps truncated to minutes; rows matched by position)"
        else:
            stop(f"HAI {n}: label timestamps do not match data timestamps")
        log.setdefault("label_alignment", {})[n] = mode
        print(f"HAI {n}: label alignment {mode.split()[0]}")
        Y[n] = lab["label"].to_numpy().astype(np.int8)
    for n, s in files.items():
        save(out / "hai" / f"{n}.npz", log, X=X[n].astype(np.float32), columns=np.array(order), roles=roles,
             y=Y[n], normal=normal_mask(Y[n], margin), t=epoch_s(data[n][0]), fs=np.int64(fs), split=np.array(s))

    inc = cfg["inclusion"]; W = int(inc["pre_window_s"] * fs); thr = float(inc["threshold_frac"]); probes = []
    A = pd.read_csv(ROOT / "spec" / "hai2305_command_attacks.csv")
    for r in A.itertuples():
        n = f"hai-{r.file}"; s, e = segments(Y[n])
        if r.segment > len(s): stop(f"{r.attack}: segment {r.segment} not in {n}")
        i, j = s[r.segment - 1], e[r.segment - 1]
        if i < W: stop(f"{r.attack}: pre-attack window shorter than {W} s")
        k = order.index(r.command); pre, dur = X[n][i - W:i, k], X[n][i:j, k]
        rng = st[r.command]["p99"] - st[r.command]["p1"]
        if rng <= 0: stop(f"{r.command}: zero training range")
        exc = max(dur.max() - pre.max(), pre.min() - dur.min(), 0.0)
        probes.append({"attack": r.attack, "file": r.file, "onset": str(data[n][0].iloc[i]), "duration_s": int(j - i),
                       "command": r.command, "exceed": round(float(exc), 4), "train_range_p1_p99": round(float(rng), 4),
                       "exceed_frac": round(float(exc / rng), 4), "included": bool(exc >= thr * rng)})

    ct = cfg["hai_crosstime"]; ctinfo = {}
    for n in ct["files"]:
        t, df = read_hai(raw / ct["dir"] / f"{n}.csv", log)
        miss = [c for c in order if c not in df.columns]
        if miss: stop(f"HAI 22.04 {n}: missing tags {miss}")
        att = [c for c in df.columns if "attack" in c.lower()]
        if att and df[att].to_numpy().any(): stop(f"HAI 22.04 {n}: attack labels non-zero in a training file")
        Xc = finite(df[order], f"HAI 22.04 {n}")
        save(out / "hai2204" / f"{n}.npz", log, X=Xc.astype(np.float32), columns=np.array(order), roles=roles,
             y=np.zeros(len(t), np.int8), normal=np.ones(len(t), bool), t=epoch_s(t), fs=np.int64(fs),
             split=np.array("crosstime"))
        ctinfo[n] = {"rows": len(t), "start": str(t.iloc[0]), "end": str(t.iloc[-1]), "label_columns": att}

    say("\n=== HAI 23.05 ===")
    for rl in ("command", "exogenous", "response", "constant"):
        say(f"{rl:9s} ({(roles == rl).sum():2d}): {[c for c in order if role[c] == rl]}")
    for n, s in files.items():
        t = data[n][0]
        say(f"{n:11s} {s:5s} rows {len(t):7d}  {t.iloc[0]} -> {t.iloc[-1]}  attack {Y[n].mean():.4f}  "
            f"normal-mask {normal_mask(Y[n], margin).mean():.4f}")
    say(f"post-attack margin {margin} s")
    say("\nprotocol 3.3 inclusion rule (pre-window %d s, threshold %.0f%% of training p1-p99 range):" % (W, thr * 100))
    say(pd.DataFrame(probes).to_string(index=False))
    say(f"included {sum(p['included'] for p in probes)} of {len(probes)}")
    say("\n=== HAI 22.04 cross-time ===")
    for n, v in ctinfo.items():
        say(f"{n}: rows {v['rows']}  {v['start']} -> {v['end']}  label columns {v['label_columns']}")
    tags = [("hai", c, role[c], "") for c in order]
    return st, tags, probes, {"margin_s": margin, "split": files, "crosstime": ctinfo}


def main():
    dirty = git("status", "--porcelain", "--", "scripts", "configs", "src", "spec")
    if dirty: stop("uncommitted changes in scripts/configs/src/spec; commit first:\n" + dirty)
    cfgp = ROOT / "configs" / "experiment.toml"; cfg = tomllib.loads(cfgp.read_text())
    if cfg["inclusion"]["range_pct"] != [1, 99]: stop("range_pct must be [1, 99] (PROTOCOL 3.3)")
    log = {"phase": 1, "git_commit": git("rev-parse", "HEAD"), "git_describe": git("describe", "--tags", "--always"),
           "config_sha256": sha256(cfgp), "script_sha256": sha256(Path(__file__)),
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "raw_sha256": {}, "outputs": {}}
    say(f"Phase 1 preprocessing | commit {log['git_commit'][:12]} ({log['git_describe']}) | config {log['config_sha256'][:12]}")
    qst, qtags, seg, qinfo = qappd(cfg, log)
    hst, htags, probes, hinfo = hai(cfg, log)
    res = ROOT / cfg["paths"]["results"]; res.mkdir(parents=True, exist_ok=True)
    (res / "stats.json").write_text(json.dumps({"qappd": qst, "hai": hst}, indent=1))
    pd.DataFrame(qtags + htags, columns=["dataset", "tag", "role", "note"]).to_csv(res / "tag_roles.csv", index=False)
    pd.DataFrame(probes).to_csv(res / "probe_inclusion.csv", index=False)
    pd.DataFrame(seg).to_csv(res / "qappd_anomaly_segments.csv", index=False)
    log |= {"qappd": qinfo, "hai": hinfo}
    (res / "manifest.json").write_text(json.dumps(log, indent=1, default=str))
    say(f"\noutputs: {len(log['outputs'])} npz files; raw files hashed: {len(log['raw_sha256'])}")
    (res / "report.txt").write_text("\n".join(REPORT) + "\n")
    shutil.copy(res / "report.txt", ROOT / "spec" / "phase1_report.txt")
    say("DONE")


if __name__ == "__main__":
    main()
