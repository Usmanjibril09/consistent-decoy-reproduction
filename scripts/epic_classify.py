#!/usr/bin/env python3
"""EPIC replication arm, upstream classifier part (D48(c),(e),(h); D49 r2-r6).
Arms U, U-split, U-rank, each with all real files and with EPIC6 removed, on the five frozen static
generations (D50). Upstream workflow_example.extract_features/train_models run unchanged in a fresh
interpreter per job. U-split replaces simprocess.ml.train_test_split at runtime (no upstream line changed);
U-rank uses a copy of ml.py whose only change is the ranking direction (diff-checked).
--smoke: static_s0 only (extraction, EPIC6 filter check, U twice, U-split, U-rank); prints no P(real)."""
import argparse, ast, csv, difflib, hashlib, json, math, os, re, shutil, statistics, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
UPREPO = HOME / "epic_work/upstream_7533f59"
UP = UPREPO / "Discriminator"
GEN = HOME / "epic_work/generation"
WORK = HOME / "epic_work/classify"
PY = ROOT / ".venv-epic/bin/python"
COMMIT = "7533f59e94e9f54d8e3f151cc53a02a5c6146b6f"
STATIC = [f"static_s{s}" for s in range(5)]
ARMS = ("U", "U-split", "U-rank")
SUBSETS = ("all", "no6")
DROP = "EPIC6.csv"                      # D48(c): every arm also run with EPIC6 removed
MODELS = ["RandomForestClassifier", "MLPClassifier", "LogisticRegression", "AdaBoostClassifier",
          "DecisionTreeClassifier", "KNeighborsClassifier"]   # upstream fast_mode=1 set; checked in ml.py
RANK_OLD = "df_feat = df_feat.sort_values(by='Importance', ascending=False)"
RANK_NEW = "df_feat = df_feat.sort_values(by='Importance', ascending=True)"
MIN_FREE_GB = 6
T0 = time.time()
LOCK = threading.Lock()
CFG, REF_COLS, GENERATORS = {}, [], []

RUNNER = r'''
import sys, os, json, math, atexit

def _main():
    mode, out, arg = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.argv = ["workflow_example.py", "--output-dir", out]
    import workflow_example as w
    if mode == "extract":
        w.ensure_dirs_exist()
        files = w.extract_features()
        json.dump([os.path.basename(f) for f in files], open(os.path.join(out, "feature_files.json"), "w"))
    elif mode == "extract_col":
        from simprocess.features import process_csv_files
        os.makedirs(w.FEATURES_DIR, exist_ok=True)
        ok = process_csv_files(data_directory=w.INPUT_DIR,
                               output_file=os.path.join(w.FEATURES_DIR, "combined_%s_features.csv" % arg),
                               target_column=arg, window_size=w.WINDOW_SIZE, extract_noise=True,
                               filter_type=w.FILTER_TYPE, epsilon=w.DEFAULT_EPSILON)
        sys.exit(0 if ok else 3)
    elif mode == "train":
        calls = [0]
        if arg == "U-split":
            import pandas as pd
            import simprocess.ml as ml
            purge = int(os.environ["EPIC_PURGE"]); halving = float(os.environ["EPIC_HALVING"])
            def ordered_split(X, y, test_size=None, shuffle=True, random_state=None, stratify=None):
                calls[0] += 1
                p = 0 if test_size == halving else purge
                src = y["source"].to_numpy()
                tr, te = [], []
                for s in pd.unique(src):
                    idx = y.index[src == s]
                    n_te = int(math.ceil(test_size * len(idx)))
                    n_tr = len(idx) - n_te
                    tr.extend(idx[:max(n_tr - p, 0)]); te.extend(idx[n_tr:])
                return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
            ml.train_test_split = ordered_split
            atexit.register(lambda: json.dump({"ordered_split_calls": calls[0]},
                                              open(os.path.join(out, "split_calls.json"), "w")))
        names = json.load(open(os.path.join(out, "feature_files.json")))
        os.makedirs(w.MODELS_DIR, exist_ok=True)
        trained = w.train_models([os.path.join(w.FEATURES_DIR, n) for n in names])
        json.dump([os.path.basename(m) for m in trained], open(os.path.join(out, "trained_models.json"), "w"))
    else:
        sys.exit(2)

if __name__ == "__main__":
    _main()
'''

def say(*a):
    with LOCK:
        print(*a, flush=True)

def stop(msg):
    say(f"STOP: {msg}")
    raise SystemExit(1)

def fmt(s):
    s = int(max(s, 0)); return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def feat_name(c):
    return "combined_all_columns_features.csv" if c == "all" else f"combined_{c}_features.csv"

def model_name(c):
    return "all_columns" if c == "all" else c

def tail_text(log, n=65536):
    with open(log, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(size - n, 0))
        return f.read().decode(errors="replace")

def last_step(log):
    L = tail_text(log).splitlines()
    def last(keys):
        for l in reversed(L):
            if any(k in l for k in keys):
                return l.split(" - ")[-1].strip()[:70]
        return ""
    a = last(("Extracting features for", "Training model for")); b = last(("Processing file", "Training data size"))
    return f"{a} | {b}" if (a or b) else "starting"

def run_job(label, mode, out, arg, cwd, code):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(WORK).free / 1e9
    if free < MIN_FREE_GB:
        stop(f"{label}: only {free:.1f} GB free (minimum {MIN_FREE_GB}); rerun resumes")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")}
    env.update(PYTHONPATH=str(code), EPIC_PURGE=str(CFG["purge"]), EPIC_HALVING=repr(CFG["halving"]))
    log = out / f"{mode}.log"
    t0 = time.time(); beat = t0
    with open(log, "w") as fh:
        p = subprocess.Popen([str(PY), "-u", str(WORK / "code/runner.py"), mode, str(out), arg],
                             cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(5)
            try:
                with open(log, "rb") as _f:
                    _f.seek(0, 2); _n = _f.tell(); _f.seek(max(0, _n - 65536)); _tail = _f.read()
            except OSError:
                _n, _tail = 0, b""
            if b"Traceback" in _tail or _n > 50_000_000:
                p.kill(); p.wait()
                raise SystemExit(f"STOP: {label}: " + ("traceback in log" if b"Traceback" in _tail else "log over 50 MB") + f" -> {log}")
            if time.time() - beat >= 60:
                beat = time.time(); say(f"      ... {label} {fmt(beat - t0)} | {last_step(log)}")
    if p.returncode != 0:
        stop(f"{label}: exit {p.returncode}; last lines of {log}:\n" + "\n".join(tail_text(log).splitlines()[-15:]))
    errs = [l.rstrip() for l in open(log, errors="replace") if " - ERROR - " in l]
    return time.time() - t0, errs

def preconditions():
    def git(*a):
        return subprocess.run(["git", "-C", str(UPREPO), *a], capture_output=True, text=True).stdout.strip()
    if git("rev-parse", "HEAD") != COMMIT: stop("upstream clone not at pinned commit")
    if git("status", "--porcelain"): stop("upstream clone not pristine")
    if not PY.exists(): stop(f"{PY} missing")
    dec = (ROOT / "spec/decisions.md").read_text()
    pur = set(re.findall(r"(\d+)-window purge", dec))
    if len(pur) != 1: stop(f"purge in D48 not unique: {pur}")
    d50 = [l for l in dec.splitlines() if l.startswith("D50 ")]
    m = re.search(r"epic_generation_manifest\.sha256[^;]*?sha256 ([0-9a-f]{64})", d50[0]) if len(d50) == 1 else None
    if not m: stop("D50 manifest sha256 not found")
    if sha(ROOT / "spec/epic_generation_manifest.sha256") != m.group(1): stop("manifest differs from D50")
    sh = re.findall(r">= (\d+)% of the primary rows", dec)
    if len(set(sh)) != 1: stop(f"D49 pass share not unique: {sh}")
    ml = (UP / "simprocess/ml.py").read_text()
    if ml.count(RANK_OLD) != 2: stop("ranking line not found exactly twice in ml.py")
    h = re.findall(r"train_test_split\(\s*X_test_non_real, y_test_non_real, test_size=([0-9.]+)", ml)
    sp = re.findall(r"^\s*split = ([0-9.]+)", ml, re.M)
    rt = re.findall(r"X_real, y_real, test_size=([0-9.]+)", ml)
    if len(h) != 1 or len(sp) != 1 or not rt: stop("split fractions not found in ml.py")
    if float(h[0]) in {float(sp[0])} | {float(x) for x in rt}: stop("halving fraction not distinguishable")
    for n in MODELS:
        if f"{n}()" not in ml: stop(f"{n} not in ml.py")
    wf = (UP / "workflow_example.py").read_text()
    keep = re.findall(r"features_to_keep=(\d+)", wf)
    if len(keep) != 1: stop("features_to_keep not found in workflow_example.py")
    p = ROOT / "spec/simprocess_reference.csv"
    lines = [l for l in p.read_text().splitlines() if l.strip() and not l.lstrip('"').startswith("#")]
    ref = list(csv.DictReader(lines))
    if not ref or not {"value", "column", "table2_noise", "table2_prob", "tol_floor", "primary"} <= set(ref[0]):
        stop("simprocess_reference.csv columns")
    return dict(purge=int(pur.pop()), halving=float(h[0]), share=int(sh[0]) / 100, keep=int(keep[0]),
                manifest_sha256=m.group(1), ref=ref, ref_sha256=sha(p),
                table2_sha256=sha(ROOT / "spec/simprocess_table2.csv"))

def tree_hashes(d):
    return {p.relative_to(d).as_posix(): sha(p) for p in sorted(d.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}

def setup_code():
    U, R = WORK / "code/U", WORK / "code/U-rank"
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    up = tree_hashes(UP)
    if not U.exists(): shutil.copytree(UP, U, ignore=ign)
    if tree_hashes(U) != up: stop(f"{U} differs from pinned upstream; remove it and rerun")
    if R.exists(): shutil.rmtree(R)
    shutil.copytree(UP, R, ignore=ign)
    b = (UP / "simprocess/ml.py").read_bytes()
    nb = b.replace(RANK_OLD.encode(), RANK_NEW.encode())
    (R / "simprocess/ml.py").write_bytes(nb)
    d = [l for l in difflib.unified_diff(b.decode().splitlines(), nb.decode().splitlines(), lineterm="", n=0)
         if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(d) != 4: stop(f"U-rank diff has {len(d)} changed lines, expected 4 (2 removed, 2 added)")
    rest = {k: v for k, v in tree_hashes(R).items() if k != "simprocess/ml.py"}
    if rest != {k: v for k, v in up.items() if k != "simprocess/ml.py"}: stop("U-rank copy differs beyond ml.py")
    (WORK / "code/runner.py").write_text(RUNNER)
    return {"upstream_files": len(up), "rank_diff": d, "runner_sha256": sha(WORK / "code/runner.py")}

def check_inputs():
    M = {}
    for l in (ROOT / "spec/epic_generation_manifest.sha256").read_text().splitlines():
        if l.strip():
            h, p = l.split(None, 1); M[p.strip()] = h
    INP, ref_real, sets = {}, None, set()
    for g in STATIC:
        real = sorted((GEN / g / "src/Discriminator/dataset").glob("EPIC*.csv"))
        if not real or not all(re.fullmatch(r"EPIC\d+\.csv", p.name) for p in real): stop(f"{g}: real files {real}")
        rh = {p.name: sha(p) for p in real}
        if ref_real is None: ref_real = rh
        elif rh != ref_real: stop(f"{g}: real EPIC copies differ from {STATIC[0]}")
        srcs = sorted(p for p in (GEN / g / "out").glob("*/*.csv") if "+" in p.name)
        names = [p.name for p in srcs]
        if len(set(names)) != len(names): stop(f"{g}: duplicate source names")
        for p in srcs:
            rel = p.relative_to(GEN).as_posix()
            if M.get(rel) != sha(p): stop(f"{rel}: not in D50 manifest or hash differs")
            if "EPIC" in p.name or "MORRIS" in p.name: stop(f"{p.name} would be labelled real by upstream")
        gn = {}
        for n in names:
            a, b = n[:-4].split("+", 1); gn.setdefault(a, set()).add(b)
        v = list(gn.values())
        if len(gn) < 2 or any(s != v[0] for s in v): stop(f"{g}: noise sets differ between generators")
        sets.add((tuple(sorted(gn)), tuple(sorted(v[0]))))
        INP[g] = {"real": real, "sources": srcs}
    if len(sets) != 1: stop("noise sets differ between generations")
    if DROP not in ref_real: stop(f"{DROP} not among real files")
    gens, noises = sets.pop()
    return INP, ref_real, list(gens), list(noises)

def extract(g, INP):
    d = WORK / g / "ext"; done = d / "DONE.json"; fd = d / "extraction/features"
    if done.exists():
        D = json.loads(done.read_text())
        if all((fd / n).exists() and sha(fd / n) == h for n, h in D["features"].items()):
            say(f"   {g}: extraction reused ({len(D['features'])} feature files, hashes verified; took {fmt(D['seconds'])})")
            return D
        stop(f"{g}: {done} does not verify; remove {d} and rerun")
    if d.exists(): shutil.rmtree(d)
    (d / "dataset").mkdir(parents=True)
    for p in INP[g]["real"] + INP[g]["sources"]: shutil.copy2(p, d / "dataset" / p.name)
    say(f"   {g}: extracting features from {len(INP[g]['real']) + len(INP[g]['sources'])} files (upstream extract_features)")
    secs, errs = run_job(f"{g} extract", "extract", d / "extraction", "-", d, WORK / "code/U")
    names = json.loads((d / "extraction/feature_files.json").read_text())
    miss = [feat_name(c) for c in REF_COLS if feat_name(c) not in names]
    if miss: stop(f"{g}: feature files missing for Table 2 columns: {miss}")
    D = {"features": {n: sha(fd / n) for n in names}, "seconds": secs, "n_errors": len(errs), "errors": errs[:10]}
    done.write_text(json.dumps(D, indent=1))
    say(f"   {g}: extraction done in {fmt(secs)} | {len(names)} feature files | {len(errs)} upstream ERROR lines")
    for e in errs[:5]: say(f"      {e[:160]}")
    return D

def make_no6(g, D):
    src = WORK / g / "ext/extraction/features"; dst = WORK / g / "ext_no6"; done = dst / "DONE.json"
    if done.exists():
        X = json.loads(done.read_text())
        if X["from"] == D["features"] and all(sha(dst / n) == h for n, h in X["features"].items()): return X
        stop(f"{g}: {done} does not verify; remove {dst} and rerun")
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True); removed = {}
    for n in D["features"]:
        r = 0
        with open(src / n, newline="") as fi, open(dst / n, "w", newline="") as fo:
            head = fi.readline(); fo.write(head)
            k = next(csv.reader([head])).index("source")
            for line in fi:
                if next(csv.reader([line]))[k] == DROP: r += 1
                else: fo.write(line)
        removed[n] = r
    X = {"from": D["features"], "features": {n: sha(dst / n) for n in D["features"]}, "removed_rows": removed}
    done.write_text(json.dumps(X, indent=1))
    return X

def v1_check(INP, col):
    import pandas as pd
    g = STATIC[0]; d = WORK / g / "v1check"
    if d.exists(): shutil.rmtree(d)
    (d / "dataset").mkdir(parents=True)
    for p in INP[g]["real"] + INP[g]["sources"]:
        if p.name != DROP: shutil.copy2(p, d / "dataset" / p.name)
    secs, _ = run_job(f"{g} {col} re-extraction without {DROP}", "extract_col", d / "extraction", col, d, WORK / "code/U")
    a = pd.read_csv(d / "extraction/features" / feat_name(col)); b = pd.read_csv(WORK / g / "ext_no6" / feat_name(col))
    a = a[[c for c in a.columns if not c.startswith("Unnamed")]].reset_index(drop=True)
    b = b[[c for c in b.columns if not c.startswith("Unnamed")]].reset_index(drop=True)
    same = list(a.columns) == list(b.columns) and a.equals(b)
    if not same:
        stop(f"{col}: removing {DROP} rows {b.shape} differs from re-extraction without {DROP} {a.shape}; decide before full run")
    return {"column": col, "identical": True, "shape": list(a.shape), "seconds": secs}

def ranking_source(log):
    cur, out = None, {}
    for l in open(log, errors="replace"):
        m = re.search(r"Training model for (\S+)", l)
        if m: cur = m.group(1); continue
        if cur and "Loading feature importance from" in l: out[cur] = "loaded"
        elif cur and "Calculating feature importance" in l: out.setdefault(cur, "computed")
    return out

def parse_reports(md):
    res = {}
    for c in REF_COLS:
        f = md / f"report_{model_name(c)}.csv"
        if not f.exists(): stop(f"missing {f}")
        rows = list(csv.DictReader(open(f, newline="")))
        got = sorted(r.get("model", "") for r in rows)
        if got != sorted(MODELS): stop(f"{f.name}: models {got}")
        res[c] = {}
        for r in rows:
            try:
                mp = {k: float(v) for k, v in ast.literal_eval(r["mean_probas"]).items()}
            except Exception:
                stop(f"{f.name}: cannot parse mean_probas {r['mean_probas'][:120]!r}")
            fs = int(float(r["features_size"]))
            if fs != CFG["keep"]: stop(f"{f.name} {r['model']}: {fs} features, expected {CFG['keep']}")
            res[c][r["model"]] = {"f1": float(r["f1"]), "accuracy": float(r["accuracy"]),
                                  "precision": float(r["precision"]), "recall": float(r["recall"]),
                                  "mean_probas": mp, "best_params": r.get("best_params"), "features_size": fs,
                                  "features_used": r.get("features_used"), "balancing": r.get("DatasetBalancing")}
    return res

def train(g, arm, sub, D, root, name):
    a = root / g / name; done = a / "DONE.json"
    if done.exists():
        R0 = json.loads(done.read_text())
        if R0.get("runner_sha256") == sha(WORK / "code/runner.py"): return R0
    if a.exists(): shutil.rmtree(a)
    (a / "features").mkdir(parents=True)
    src = WORK / g / ("ext/extraction/features" if sub == "all" else "ext_no6")
    for n in D["features"]: os.link(src / n, a / "features" / n)
    (a / "feature_files.json").write_text(json.dumps(list(D["features"])))
    code = WORK / "code" / ("U-rank" if arm == "U-rank" else "U")
    secs, errs = run_job(f"{g} {name}", "train", a, arm, a, code)
    trained = json.loads((a / "trained_models.json").read_text())
    rk = ranking_source(a / "train.log")
    bad = [c for c in REF_COLS if rk.get(model_name(c)) != "computed"]
    if bad: stop(f"{g} {name}: ranking not computed fresh for {bad} ({ {model_name(c): rk.get(model_name(c)) for c in bad} })")
    res = parse_reports(a / "models")
    calls = None
    if arm == "U-split":
        calls = json.loads((a / "split_calls.json").read_text())["ordered_split_calls"]
        want = 3 * len(MODELS) * len(trained)
        if calls != want: stop(f"{g} {name}: ordered split called {calls} times, expected {want}")
    mh = {}
    for p in sorted((a / "models").glob("*.joblib")):
        mh[p.name] = sha(p); p.unlink()
    R = {"runner_sha256": sha(WORK / "code/runner.py"), "gen": g, "arm": arm, "subset": sub, "seconds": secs, "n_errors": len(errs), "errors": errs[:10],
         "trained": len(trained), "split_calls": calls, "ranking_loaded_other": sorted(k for k, v in rk.items() if v == "loaded"),
         "model_sha256": mh, "results": res}
    done.write_text(json.dumps(R))
    return R

def determinism(A, B):
    worst = worst_rf = f1 = 0.0; n_diff = 0
    for c in A:
        for m in A[c]:
            pa, pb = A[c][m]["mean_probas"], B[c][m]["mean_probas"]
            if set(pa) != set(pb): stop(f"determinism: sources differ for {c} {m}")
            d = max(abs(pa[k] - pb[k]) for k in pa)
            worst = max(worst, d); n_diff += d > 0
            if m == MODELS[0]: worst_rf = max(worst_rf, d)
            f1 = max(f1, abs(A[c][m]["f1"] - B[c][m]["f1"]))
    return {"max_abs_diff_mean_p_real_all_models": worst, "max_abs_diff_mean_p_real_rf": worst_rf,
            "max_abs_diff_f1": f1, "column_model_pairs_differing": n_diff,
            "column_model_pairs": sum(len(A[c]) for c in A)}

def evaluate(R, noises):
    rf = MODELS[0]
    def p_of(res, col, noise, model):
        mp = res[col][model]["mean_probas"]
        v = [mp[k] for k in (f"{g}+{noise}.csv" for g in GENERATORS) if k in mp and mp[k] >= 0]
        return max(v) if v else None
    def st(vals):
        ok = [v for v in vals if v is not None]
        if not ok: return None, None, len(vals)
        return statistics.fmean(ok), (statistics.stdev(ok) if len(ok) > 1 else 0.0), len(vals) - len(ok)
    E = {}
    for arm in ARMS:
        for sub in SUBSETS:
            rows = []
            for r in CFG["ref"]:
                col, nz, t2 = r["column"], r["table2_noise"].lower(), float(r["table2_prob"])
                if nz not in noises: stop(f"Table 2 noise {nz} not among generated noises {noises}")
                rs = [R[(g, arm, sub)] for g in STATIC]
                m, sd, miss = st([p_of(x, col, nz, rf) for x in rs])
                mb, sdb, _ = st([p_of(x, col, nz, max(x[col], key=lambda k: x[col][k]["f1"])) for x in rs])
                nm = {n: st([p_of(x, col, n, rf) for x in rs])[0] for n in noises}
                nm = {n: v for n, v in nm.items() if v is not None}
                best = max(nm, key=nm.get) if nm else None
                tol = max(float(r["tol_floor"]), 2 * sd) if sd is not None else None
                rows.append(dict(value=r["value"], column=col, noise=nz, table2=t2, primary=r["primary"] == "1",
                                 mean=m, sd=sd, missing=miss, tol=tol,
                                 within=(m is not None and abs(t2 - m) <= tol), best_noise=best,
                                 best_matches=(best == nz), bestf1_mean=mb, bestf1_sd=sdb))
            E[f"{arm}__{sub}"] = rows
    return E

def main():
    global CFG, REF_COLS, GENERATORS
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--time-one", action="store_true"); a = ap.parse_args()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"EPIC classifier arm (D48/D49) | repo {head} | upstream {COMMIT[:7]} | {'SMOKE' if a.smoke else 'TIME-ONE' if a.time_one else 'FULL'}")
    CFG = preconditions()
    REF_COLS = [r["column"] for r in CFG["ref"]]
    WORK.mkdir(parents=True, exist_ok=True)
    code = setup_code()
    say(f"   preconditions OK | upstream pristine | D50 manifest verified | purge {CFG['purge']} windows | "
        f"halving split {CFG['halving']} (no purge) | top-{CFG['keep']} features | pass share {CFG['share']}")
    say(f"   code: {code['upstream_files']} upstream files copied unchanged; U-rank diff: {code['rank_diff']}")
    INP, real, gens, noises = check_inputs()
    GENERATORS = gens
    say(f"   inputs: {len(real)} real files identical in all {len(STATIC)} generations; "
        f"{len(INP[STATIC[0]]['sources'])} sources per generation ({', '.join(gens)} x {len(noises)} noises), all in D50 manifest")
    me = Path(__file__).resolve()
    if a.smoke or a.time_one:
        sm = WORK / "smoke"
        for d in (sorted(sm.glob("*/*")) if sm.exists() else []):
            if d.is_dir() and not (d / "DONE.json").exists(): shutil.rmtree(d)
        g = STATIC[0]
        D = extract(g, INP)
        X = make_no6(g, D)
        say(f"   {g}: {DROP} rows removed per feature file: min {min(X['removed_rows'].values())}, max {max(X['removed_rows'].values())}")
        if a.time_one:
            r = train(g, "U", "all", D, sm, "U_a")
            n_full = len(STATIC) * len(ARMS) * len(SUBSETS)
            say(f"   [time-one] {g} U_a: {fmt(r['seconds'])} | {r['trained']} models trained | {r['n_errors']} upstream ERROR lines")
            say(f"   projection: smoke needs 3 more trainings, about {fmt(3 * r['seconds'])} (+ a few minutes for the V1 gate)")
            say(f"   projection: FULL needs {n_full} trainings, about {fmt(n_full * r['seconds'])}, plus {len(STATIC) - 1} extractions, up to {fmt((len(STATIC) - 1) * D['seconds'])}")
            say("TIME-ONE DONE (no P(real) shown; nothing written to the repo)")
            return
        col = next(c for c in REF_COLS if c != "all")
        v1 = v1_check(INP, col)
        say(f"   gate: removing {DROP} rows equals re-extraction without {DROP} for {col} {tuple(v1['shape'])}")
        runs = {}
        for tag, arm in (("U_a", "U"), ("U_b", "U"), ("U-split", "U-split"), ("U-rank", "U-rank")):
            r = train(g, arm, "all", D, sm, tag); runs[tag] = r
            extra = f" | ordered split calls {r['split_calls']}" if r["split_calls"] is not None else ""
            say(f"   [smoke] {g} {tag}: {fmt(r['seconds'])} | {r['trained']} models trained | {r['n_errors']} ERROR lines{extra}")
        A = runs["U_a"]["results"]
        same = [c for c in REF_COLS if runs["U-rank"]["results"][c][MODELS[0]]["features_used"] == A[c][MODELS[0]]["features_used"]]
        if same: stop(f"U-rank selects the same features as U for {same}: ranking change had no effect")
        say(f"   gate: U-rank selects different features from U for all {len(REF_COLS)} Table 2 columns")
        det = determinism(A, runs["U_b"]["results"])
        say(f"   D48(h) nondeterminism, U twice on {g}: max |diff| mean P(real) RF {det['max_abs_diff_mean_p_real_rf']:.4f}, "
            f"all models {det['max_abs_diff_mean_p_real_all_models']:.4f}, F1 {det['max_abs_diff_f1']:.4f} "
            f"({det['column_model_pairs_differing']} of {det['column_model_pairs']} column-model pairs differ)")
        t_tr = statistics.fmean(r["seconds"] for r in runs.values())
        n_tr = len(STATIC) * len(ARMS) * len(SUBSETS)
        disk = sum(p.stat().st_size for p in (WORK / g).rglob("*") if p.is_file()) / 1e9
        proj = {"extraction_single_s": D["seconds"], "extractions_left": len(STATIC) - 1, "train_mean_s": t_tr,
                "trainings": n_tr, "train_total_s": n_tr * t_tr, "disk_per_generation_gb": disk}
        say(f"   projection FULL: {len(STATIC) - 1} more extractions, 4 in parallel: about {fmt(D['seconds'])} if parallel is free, "
            f"up to {fmt(D['seconds'] * (len(STATIC) - 1))} if serial (single run measured, parallel slowdown not)")
        say(f"   projection FULL: {n_tr} trainings x {fmt(t_tr)} = {fmt(n_tr * t_tr)} (sequential; upstream GridSearchCV uses all cores)")
        say(f"   projection disk: {disk:.2f} GB per generation x {len(STATIC)} = {disk * len(STATIC):.1f} GB; free now {shutil.disk_usage(WORK).free / 1e9:.1f} GB")
        out = {"stage": "epic_classify_smoke", "values_displayed": False, "repo_head": head,
               "script_sha256": sha(me), "runner_sha256": code["runner_sha256"], "upstream_commit": COMMIT,
               "rank_diff": code["rank_diff"], "purge_windows": CFG["purge"], "halving_fraction_no_purge": CFG["halving"],
               "features_to_keep": CFG["keep"], "pass_share": CFG["share"], "manifest_sha256": CFG["manifest_sha256"],
               "reference_sha256": CFG["ref_sha256"], "table2_sha256": CFG["table2_sha256"],
               "real_files": real, "generators": gens, "noises": noises,
               "extraction": {k: D[k] for k in ("seconds", "n_errors", "errors")} | {"feature_files": D["features"]},
               "no6_removed_rows": X["removed_rows"], "filter_check": v1,
               "trainings": {t: {k: r[k] for k in ("seconds", "n_errors", "errors", "trained", "split_calls", "ranking_loaded_other")}
                             for t, r in runs.items()},
               "determinism": det, "projection": proj, "seconds": time.time() - T0}
        (ROOT / "spec/epic_classify_smoke.json").write_text(json.dumps(out, indent=1))
        say(f"EPIC SMOKE PASSED in {fmt(time.time() - T0)} -> spec/epic_classify_smoke.json (no P(real) value shown)")
        return
    sj = ROOT / "spec/epic_classify_smoke.json"
    if not sj.exists(): stop("smoke record missing")
    if json.loads(sj.read_text())["script_sha256"] != sha(me): stop("script changed since the smoke run")
    with ThreadPoolExecutor(4) as ex:
        Ds = dict(zip(STATIC, ex.map(lambda g: extract(g, INP), STATIC)))
    for g in STATIC: make_no6(g, Ds[g])
    jobs = [(g, arm, sub) for g in STATIC for arm in ARMS for sub in SUBSETS]
    R, meta, times = {}, {}, []
    for k, (g, arm, sub) in enumerate(jobs, 1):
        name = f"{arm}__{sub}"
        pre = (WORK / "full" / g / name / "DONE.json").exists()
        r = train(g, arm, sub, Ds[g], WORK / "full", name)
        if not pre: times.append(r["seconds"])
        left = statistics.fmean(times) * (len(jobs) - k) if times else 0
        say(f"   [{k}/{len(jobs)}] {g} {name}: {fmt(r['seconds'])}{' (reused)' if pre else ''} | "
            f"elapsed {fmt(time.time() - T0)} | left ~{fmt(left)}")
        R[(g, arm, sub)] = r["results"]
        meta[f"{g}/{name}"] = {k2: r[k2] for k2 in ("seconds", "n_errors", "errors", "trained", "split_calls", "ranking_loaded_other", "model_sha256")}
    E = evaluate(R, noises)
    P = [r for r in E["U__all"] if r["primary"]]
    need = math.ceil(CFG["share"] * len(P)); npass = sum(r["within"] for r in P)
    say("== Table 2 vs arm U (RandomForest, mean over 5 generations; D49 rule) and other arms (no rule)")
    for i, r in enumerate(E["U__all"]):
        oth = " | ".join(f"{key.replace('__all', '').replace('__no6', '-noEPIC6')} {E[key][i]['mean']:.3f}"
                         for key in E if key != "U__all" and E[key][i]["mean"] is not None)
        say(f"   {r['value']:<15} {r['noise']:<16} T2 {r['table2']:.3f} | U {r['mean']:.3f} sd {r['sd']:.3f} tol {r['tol']:.3f} "
            f"{'within' if r['within'] else 'OUTSIDE'}{'' if r['primary'] else ' (not primary)'} | best {r['best_noise']} | {oth}")
    say(f"   U reproduces Table 2: {'YES' if npass >= need else 'NO'} ({npass} of {len(P)} primary rows within tolerance; rule >= {need})")
    say(f"   best-noise agreement (U, all rows): {sum(r['best_matches'] for r in E['U__all'])} of {len(E['U__all'])}")
    S = {"stage": "epic_classify_full", "repo_head": head, "script_sha256": sha(me), "upstream_commit": COMMIT,
         "rule": {"pass_share": CFG["share"], "needed": need, "passed": npass, "primary_rows": len(P), "reproduced": npass >= need},
         "evaluation": E, "jobs": meta, "extraction": {g: {k: Ds[g][k] for k in ("seconds", "n_errors")} for g in STATIC},
         "seconds": time.time() - T0}
    f = WORK / "full/summary.json"; f.write_text(json.dumps(S, indent=1))
    say(f"EPIC FULL DONE in {fmt(time.time() - T0)} -> {f} sha256 {sha(f)}")

if __name__ == "__main__":
    main()
