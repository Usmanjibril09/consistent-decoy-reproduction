#!/usr/bin/env python3
"""D48(f) paper-settings sensitivity on stored HAI/QAPPD SimProcess features (no feature extraction, no decoy training).
U = upstream ml._train_model unchanged (pinned 7533f59): top-11 relevance features as coded (descending p-value),
SMOTE 0.9, RF grid n_estimators [50, 200] scored on recall, 90/10 real and 30/70 simulated random split, halved test.
U-split = U with the three train_test_split calls replaced by ordered per-source splits (same fractions, 4-row purge,
none on the halving call). Outputs C1-Full, C1, B8, B8-L, B1 x seeds 0-4 x {hai, qappd} = 50.
--smoke: gate (our D45/D46 classify() reproduces stored C1-Full_s0 / B1_s0 results on both plants), then U and U-split on
C1-Full_s0 per plant with checks and timing; prints no sensitivity value. The full run needs this script's smoke record and D66."""
import argparse, ast, hashlib, json, math, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "results/phase3"
SPD, DEC, OUT = P3 / "simprocess", P3 / "decoy", P3 / "sensitivity"
UP = Path.home() / "epic_work/upstream_7533f59"
COMMIT = "7533f59e94e9f54d8e3f151cc53a02a5c6146b6f"
METHODS = ("C1-Full", "C1", "B8", "B8-L", "B1")
SEEDS = (0, 1, 2, 3, 4)
PLANTS = ("hai", "qappd")
ARMS = ("U", "U-split")
TOP_K, BAL, PURGE = 11, 0.9, 4
GRID = {"n_estimators": [50, 200], "max_depth": [None]}      # upstream fast_mode=1 RF grid (ml.py lines 214-218)
SMOKE_REC = ROOT / "spec/epic_sensitivity_smoke.json"
LOGF = [None]

def say(m):
    print(m, flush=True)
    if LOGF[0]:
        with open(LOGF[0], "a") as f: f.write(m + "\n")

def stop(m):
    say(f"STOP: {m}"); sys.exit(1)

def need(ok, m):
    if not ok: stop(m)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def fmt(s):
    s = int(s); return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"

def git(*a, cwd=ROOT):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True).stdout.strip()

def upstream():
    rel = "Discriminator/simprocess/ml.py"
    need((UP / rel).exists(), f"{UP / rel} missing")
    need(git("hash-object", rel, cwd=UP) == git("rev-parse", f"{COMMIT}:{rel}", cwd=UP), "upstream ml.py differs from the pinned commit")
    need(git("status", "--porcelain", "--untracked-files=no", cwd=UP) == "", "upstream clone has modified tracked files")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(UP / "Discriminator"))
    import simprocess.ml as ml
    need(Path(ml.__file__).resolve() == (UP / rel).resolve(), f"imported ml.py from {ml.__file__}")
    return ml

def feat(pn, s, o, manifest):
    d = SPD / "feat" / pn / s; npz = d / f"{o}.npz"; meta = json.loads((d / f"{o}.json").read_text())
    h = sha(npz)
    need(h == meta["npz_sha256"], f"{pn} {s} {o}: npz sha256 differs from its json")
    need(h in manifest, f"{pn} {s} {o}: npz not in the D47 manifest")
    z = np.load(npz, allow_pickle=False); F, rows, ids = z["F"], z["rows"].astype(int), z["ids"].astype(int)
    need(ids.tolist() == meta["ids"] and int(rows.sum()) == len(F), f"{pn} {s} {o}: ids or rows inconsistent")
    need(bool(np.isfinite(F).all()), f"{pn} {s} {o}: non-finite feature values")
    off = np.concatenate([[0], np.cumsum(rows)])
    return [F[off[i]:off[i + 1]] for i in range(len(rows))], ids, h

def plant_ctx(pn, manifest):
    sel = json.loads((SPD / "selection.json").read_text())[pn]
    names = json.loads((SPD / f"feature_names_{pn}.json").read_text())
    idx = json.loads((DEC / pn / "normal" / "index.json").read_text())
    R, ids, h = feat(pn, "normal", "real", manifest)
    need(ids.tolist() == list(sel["normal_train"]) + list(sel["normal_test"]), f"{pn}: real feature windows != selection train + test")
    need(all(len(a) == 0 or a.shape[1] == len(names) for a in R), f"{pn}: feature width != feature_names")
    return dict(pn=pn, names=names, idx=idx, R=R, ids=ids, ntr=len(sel["normal_train"]), real_sha=h)

def our_classify():
    src = (ROOT / "scripts/03d_simprocess.py").read_text()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "classify"]
    need(len(fn) == 1, "classify() not found exactly once in scripts/03d_simprocess.py")
    code = ast.get_source_segment(src, fn[0]); g = {"np": np, "time": time, "stop": stop}
    exec(code, g)
    return g["classify"], hashlib.sha256(code.encode()).hexdigest()

def gate(ctxs, manifest):
    classify, ch = our_classify(); out = {}
    for c in ctxs:
        for o in ("C1-Full_s0", "B1_s0"):
            D, ids, _ = feat(c["pn"], "normal", o, manifest)
            need(ids.tolist() == c["ids"].tolist(), f"{c['pn']} {o}: decoy windows differ from real windows")
            n = c["ntr"]; t = time.time()
            say(f"   gate {c['pn']} {o}: fitting our D45/D46 classifier (seed 0) ...")
            r = classify(c["R"][:n], D[:n], c["R"][n:], D[n:], [], 0)
            ref = json.loads((SPD / "pred" / c["pn"] / f"{o}_c0.json").read_text())
            ok = (r["accuracy"] == ref["accuracy"] and abs(r["m2b"] - ref["m2b"]) <= 1e-12
                  and r["best_n_estimators"] == ref["best_n_estimators"] and r["train_rows"] == ref["train_rows"])
            need(ok, f"gate {c['pn']} {o}: ours gives accuracy {r['accuracy']} P(real) {r['m2b']} trees {r['best_n_estimators']} rows {r['train_rows']}; "
                     f"stored {ref['accuracy']} {ref['m2b']} {ref['best_n_estimators']} {ref['train_rows']}")
            say(f"   gate {c['pn']} {o}: reproduces the stored result exactly (accuracy {r['accuracy']:.3f}, {r['train_rows']} rows, {time.time() - t:.0f} s)")
            out[f"{c['pn']}/{o}"] = {"accuracy": r["accuracy"], "m2b": r["m2b"], "seconds": time.time() - t}
    return ch, out

def frame(c, o, manifest):
    D, ids, h = feat(c["pn"], "normal", o, manifest)
    need(ids.tolist() == c["ids"].tolist(), f"{c['pn']} {o}: decoy windows differ from real windows")
    idx = c["idx"]
    order = sorted(range(len(ids)), key=lambda i: (idx[ids[i]]["file"], int(idx[ids[i]]["start"]), i))   # time order per file
    X, real, src = [], [], []
    for lab, W in ((1, c["R"]), (0, D)):
        for i in order:
            A = W[i]
            if len(A):
                X.append(A); real += [lab] * len(A)
                src += [f"real:{idx[ids[i]]['file']}" if lab else o] * len(A)
    df = pd.DataFrame(np.concatenate(X).astype(np.float64), columns=c["names"])
    df["real"] = real; df["source"] = src
    n0 = len(df); df = df.dropna()                                                     # as train_binary
    df = df[~df["source"].str.contains("Gan", case=False, na=False)]
    df = df[~df["source"].str.contains("dynamic_", case=False, na=False)]
    need(len(df) == n0, f"{c['pn']} {o}: upstream row filters removed {n0 - len(df)} rows")
    return df.reset_index(drop=True), h

def rank(ml, df):
    rel = ml.calculate_relevance_table(df.drop(columns=["real", "source"]), df["real"])
    f = rel[["feature", "p_value"]].copy(); f.columns = ["Feature", "Importance"]
    f = f.sort_values(by="Importance", ascending=False)                                # ml.py line 449, as coded
    feats = f["Feature"].tolist()[:TOP_K]
    need(len(feats) == TOP_K, f"ranking returned {len(feats)} features")
    return feats, int(f["Importance"].isna().sum())

def train(ml, df, feats, arm):
    calls = [0]; orig = ml.train_test_split
    def counted(*a, **k):
        calls[0] += 1; return orig(*a, **k)
    def ordered(X, y, test_size=None, shuffle=True, random_state=None, stratify=None):
        calls[0] += 1
        purge = 0 if test_size == 0.5 else PURGE
        tr, te = [], []
        for s in pd.unique(y["source"]):
            ix = y.index[(y["source"] == s).to_numpy()]
            n = len(ix); nte = math.ceil(test_size * n); ntr = max(n - nte - purge, 0)
            tr.extend(ix[:ntr]); te.extend(ix[n - nte:])
        return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    ml.train_test_split = ordered if arm == "U-split" else counted
    np.random.seed(42)                                   # state of a fresh upstream process (ml.py line 24)
    t = time.time()
    try:
        r = ml._train_model(df[feats], df[["real", "source"]], ml.RandomForestClassifier(), GRID, dataset_balancing_ratio=BAL)
    finally:
        ml.train_test_split = orig
    need(calls[0] == 3, f"{arm}: train_test_split called {calls[0]} times, expected 3")
    r.pop("best_estimator", None)
    return json.loads(json.dumps(r, default=lambda v: v.item() if hasattr(v, "item") else str(v))), time.time() - t

def brief(r, o):
    mp = r["mean_probas"]; rk = [k for k in mp if k.startswith("real:")]
    need(o in mp and rk, f"{o}: test set lacks the decoy or real sources ({list(mp)})")
    return {"accuracy": r["accuracy"], "f1": r["f1"], "recall": r["recall"], "precision": r["precision"],
            "p_real_decoy": mp[o], "p_real_real": float(np.mean([mp[k] for k in rk])), "best_params": r["best_params"],
            "balancing": r["DatasetBalancing"]}

def job(ml, c, o, manifest):
    t0 = time.time(); df, h = frame(c, o, manifest); t1 = time.time()
    say(f"      {c['pn']} {o}: {len(df)} rows; ranking {len(c['names'])} features (upstream relevance table) ...")
    feats, nnan = rank(ml, df); t2 = time.time()
    rec = {"plant": c["pn"], "output": o, "features_sha256": h, "real_features_sha256": c["real_sha"],
           "rows_real": int(df["real"].sum()), "rows_decoy": int((df["real"] == 0).sum()), "top11": feats,
           "nan_p_values": nnan, "seconds_frame": t1 - t0, "seconds_rank": t2 - t1, "arms": {}}
    for arm in ARMS:
        say(f"      {c['pn']} {o}: training arm {arm} ...")
        r, s = train(ml, df, feats, arm)
        rec["arms"][arm] = {"seconds": s, "summary": brief(r, o), "result": r}
    return rec

def summary(full):
    ours = json.loads((SPD / "summary.json").read_text()); S = {}
    f = lambda v: "  n/a" if v is None else f"{v:.3f}"
    say("== D48(f) sensitivity: mean (sd) over seeds 0-4 | ours = D45/D46 stored, mean over classifier seeds")
    for pn in PLANTS:
        for m in METHODS:
            row = {"ours_accuracy": float(np.mean([ours[f"{pn}/{m}_s{s}"]["accuracy"] for s in SEEDS])),
                   "ours_m2b": float(np.mean([ours[f"{pn}/{m}_s{s}"]["m2b"] for s in SEEDS])) if pn == "hai" else None}
            for arm in ARMS:
                for k in ("accuracy", "p_real_decoy"):
                    v = [json.loads((full / pn / f"{m}_s{s}.json").read_text())["arms"][arm]["summary"][k] for s in SEEDS]
                    row[f"{arm}_{k}"] = float(np.mean(v)); row[f"{arm}_{k}_sd"] = float(np.std(v, ddof=1))
            S[f"{pn}/{m}"] = row
            say(f"   {pn:5s} {m:8s} ours acc {f(row['ours_accuracy'])} P(real) {f(row['ours_m2b'])} | "
                + " | ".join(f"{a} acc {f(row[a + '_accuracy'])} ({row[a + '_accuracy_sd']:.3f}) P(real) {f(row[a + '_p_real_decoy'])} ({row[a + '_p_real_decoy_sd']:.3f})" for a in ARMS)
                + ("  [QAPPD P(real): diagnostic, not M2b (D43)]" if pn == "qappd" else ""))
    return S

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOGF[0] = OUT / ("smoke.log" if a.smoke else "run.log")
    head = git("rev-parse", "--short=7", "HEAD"); me = sha(Path(__file__))
    say(f"D48(f) paper-settings sensitivity | repo {head} | upstream {COMMIT[:7]} | {'SMOKE' if a.smoke else 'FULL'} | script {me[:12]}")
    manifest = (ROOT / "spec/phase3_simprocess_manifest.sha256").read_text()
    miss = [f"{pn}/{m}_s{s}" for pn in PLANTS for m in METHODS for s in SEEDS
            if not (SPD / "feat" / pn / "normal" / f"{m}_s{s}.npz").exists()]
    need(not miss, f"stored features missing: {miss}")
    if not a.smoke:
        need(SMOKE_REC.exists(), "no smoke record; run --smoke first")
        need(json.loads(SMOKE_REC.read_text())["script_sha256"] == me, "smoke record belongs to a different script version")
        need(re.search(r"(?m)^D66\b", (ROOT / "spec/decisions.md").read_text()), "D66 (settings of this arm) not committed")
        need(git("status", "--porcelain", "--", "scripts/epic_sensitivity.py", "spec") == "", "script or spec/ has uncommitted changes")
    ml = upstream()
    import sklearn, imblearn, tsfresh
    vers = {"sklearn": sklearn.__version__, "imblearn": imblearn.__version__, "tsfresh": tsfresh.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    say(f"   upstream ml.py = pinned blob, imported from {ml.__file__} | {vers}")
    ctxs = [plant_ctx(pn, manifest) for pn in PLANTS]
    for c in ctxs:
        say(f"   {c['pn']}: {len(c['R'])} real windows ({c['ntr']} in first half), {len(c['names'])} features, {sum(map(len, c['R']))} real rows")
    if a.smoke:
        ch, g = gate(ctxs, manifest); T, checks = {}, {}
        for c in ctxs:
            rec = job(ml, c, "C1-Full_s0", manifest)
            sd = OUT / "smoke" / c["pn"]; sd.mkdir(parents=True, exist_ok=True)
            (sd / "C1-Full_s0.json").write_text(json.dumps(dict(rec, script_sha256=me), indent=1))
            T[c["pn"]] = rec["seconds_frame"] + rec["seconds_rank"] + sum(v["seconds"] for v in rec["arms"].values())
            checks[c["pn"]] = {"top11": rec["top11"], "nan_p_values": rec["nan_p_values"]}
            say(f"   smoke {c['pn']} C1-Full_s0: frame {rec['seconds_frame']:.0f} s, ranking {rec['seconds_rank']:.0f} s, "
                + ", ".join(f"{k} {v['seconds']:.0f} s" for k, v in rec["arms"].items())
                + f" | 3 split calls per arm, {TOP_K} features, test set has real and decoy sources (values not shown)")
        n = len(METHODS) * len(SEEDS); proj = {pn: n * T[pn] for pn in PLANTS}
        say(f"   projection: {n} outputs per plant | hai ~{fmt(proj['hai'])} | qappd ~{fmt(proj['qappd'])} | total ~{fmt(sum(proj.values()))}")
        SMOKE_REC.write_text(json.dumps({"script_sha256": me, "classify_sha256": ch, "repo_head": head, "versions": vers,
                                         "gate": g, "checks": checks, "seconds_per_output": T, "projection_s": proj,
                                         "values_displayed": False}, indent=1))
        say(f"SENSITIVITY SMOKE PASSED -> {SMOKE_REC.relative_to(ROOT)} (projection ~{fmt(sum(proj.values()))})")
        return
    full = OUT / "full"; jobs = [(c, f"{m}_s{s}") for c in ctxs for m in METHODS for s in SEEDS]
    t0, k, done = time.time(), 0, 0
    try:
        for c, o in jobs:
            k += 1; p = full / c["pn"] / f"{o}.json"
            if p.exists() and json.loads(p.read_text()).get("script_sha256") == me: done += 1; continue
            t = time.time(); rec = job(ml, c, o, manifest); rec["script_sha256"] = me
            p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=1)); tmp.replace(p)
            ran = k - done; el = time.time() - t0
            say(f"   [{k}/{len(jobs)}] {c['pn']} {o}: {fmt(time.time() - t)} | elapsed {fmt(el)} | left ~{fmt(el / ran * (len(jobs) - k))}")
    except KeyboardInterrupt:
        say("INTERRUPTED: finished outputs are kept; rerun the same command to resume"); sys.exit(130)
    S = summary(full); sp = OUT / "summary.json"
    sp.write_text(json.dumps({"script_sha256": me, "repo_head": head, "versions": vers, "rows": S}, indent=1))
    say(f"SENSITIVITY FULL DONE in {fmt(time.time() - t0)} -> {sp.relative_to(ROOT)} sha256 {sha(sp)}")

if __name__ == "__main__":
    main()
