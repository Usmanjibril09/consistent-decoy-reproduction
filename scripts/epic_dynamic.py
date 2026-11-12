#!/usr/bin/env python3
"""EPIC dynamic test (D48; D49 r5). Exploratory, outside the Holm family.
Upstream detector exactly as arm U (workflow_example.train_models -> train_command.run -> ml.train_binary, unchanged),
with only dynamic=True per train call (= upstream CLI --dynamic): train on real + static simulations, test only on the
unseen dynamic_ sources. dynamic_s{100+s} paired with static_s{s}. Static rows = arm U's frozen feature files verbatim;
dynamic rows = upstream extract_features on 8 real + 22 dynamic files (real rows gated identical to arm U).
--smoke: dynamic_s100 only (extraction, gates, one training on 'all'); prints no P(real)."""
import argparse, csv, json, os, re, shutil, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import epic_classify as EC

WORK = EC.WORK
DYN = WORK / "dynamic"
RUNNER_PATH = WORK / "code/runner_dyn.py"
PAIRS = [(f"static_s{s}", f"dynamic_s{100 + s}") for s in range(5)]
SUBSETS = ("all", "no6")
ARMU_SUMMARY_SHA = "171102266ccef876f59ff8d7caf5ce1e02d87a487e3aeef142431a30b5697c59"
LOG_CAP = 500_000_000
T0 = time.time()
say, stop, fmt, sha = EC.say, EC.stop, EC.fmt, EC.sha

RUNNER = r'''
import sys, os, json

def _main():
    mode, out = sys.argv[1], sys.argv[2]
    sys.argv = ["workflow_example.py", "--output-dir", out]
    import workflow_example as w
    if mode == "extract":
        w.ensure_dirs_exist()
        files = w.extract_features()
        json.dump([os.path.basename(f) for f in files], open(os.path.join(out, "feature_files.json"), "w"))
    elif mode == "train":
        calls = []
        orig = w.train_run
        def dyn_run(a):
            a.dynamic = True
            calls.append(os.path.basename(a.input))
            return orig(a)
        w.train_run = dyn_run
        names = json.load(open(os.path.join(out, "feature_files.json")))
        os.makedirs(w.MODELS_DIR, exist_ok=True)
        trained = w.train_models([os.path.join(w.FEATURES_DIR, n) for n in names])
        json.dump({"trained": [os.path.basename(m) for m in trained], "dynamic_calls": calls},
                  open(os.path.join(out, "trained_models.json"), "w"))
    else:
        sys.exit(2)

if __name__ == "__main__":
    _main()
'''

def run_job(label, mode, out, cwd):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(WORK).free / 1e9
    if free < EC.MIN_FREE_GB:
        stop(f"{label}: only {free:.1f} GB free (minimum {EC.MIN_FREE_GB}); rerun resumes")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")}
    env["PYTHONPATH"] = str(WORK / "code/U")
    log = out / f"{mode}.log"
    t0 = beat = time.time()
    with open(log, "w") as fh:
        p = subprocess.Popen([str(EC.PY), "-u", str(RUNNER_PATH), mode, str(out)],
                             cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(5)
            try:
                n = log.stat().st_size; tail = EC.tail_text(log)
            except OSError:
                n, tail = 0, ""
            if "Traceback" in tail or n > LOG_CAP:
                p.kill(); p.wait()
                stop(f"{label}: " + ("traceback in log" if "Traceback" in tail else "log over 500 MB") + f" -> {log}")
            if time.time() - beat >= 60:
                beat = time.time(); say(f"      ... {label} {fmt(beat - t0)} | {EC.last_step(log)}")
    if p.returncode != 0:
        stop(f"{label}: exit {p.returncode}; last lines of {log}:\n" + "\n".join(EC.tail_text(log).splitlines()[-15:]))
    errs = [l.rstrip() for l in open(log, errors="replace") if " - ERROR - " in l]
    return time.time() - t0, errs

def inputs():
    INP, real, gens, noises = EC.check_inputs()
    M = {}
    for l in (ROOT / "spec/epic_generation_manifest.sha256").read_text().splitlines():
        if l.strip():
            h, p = l.split(None, 1); M[p.strip()] = h
    DI = {}
    for st, dy in PAIRS:
        srcs = sorted(p for p in (EC.GEN / dy / "out").glob("*/*.csv") if "+" in p.name)
        names = [p.name for p in srcs]
        if len(set(names)) != len(names): stop(f"{dy}: duplicate source names")
        if not names or not all(n.startswith("dynamic_") for n in names): stop(f"{dy}: source without dynamic_ prefix")
        for p in srcs:
            rel = p.relative_to(EC.GEN).as_posix()
            if M.get(rel) != sha(p): stop(f"{rel}: not in D50 manifest or hash differs")
            if "EPIC" in p.name or "MORRIS" in p.name: stop(f"{p.name} would be labelled real by upstream")
        if sorted(n[len("dynamic_"):] for n in names) != sorted(p.name for p in INP[st]["sources"]):
            stop(f"{dy}: sources differ from {st}")
        DI[dy] = {"real": INP[st]["real"], "sources": srcs}
    return DI, real, gens, noises

def p_of(res, col, nz, prefix):
    mp = res[col][EC.MODELS[0]]["mean_probas"]
    v = [mp[k] for k in (f"{prefix}{g}+{nz}.csv" for g in EC.GENERATORS) if k in mp and mp[k] >= 0]
    return max(v) if v else None

def ms(vals):
    ok = [v for v in vals if v is not None]
    if not ok: return None, None
    return statistics.fmean(ok), (statistics.stdev(ok) if len(ok) > 1 else 0.0)

def load_armU():
    f = WORK / "full/summary.json"
    if not f.exists() or sha(f) != ARMU_SUMMARY_SHA: stop("arm U summary missing or sha256 differs from D65")
    if ARMU_SUMMARY_SHA not in (ROOT / "spec/decisions.md").read_text(): stop("arm U summary sha256 not in decisions.md")
    S = json.loads(f.read_text())
    U = {}
    for st, _ in PAIRS:
        for sub in SUBSETS:
            d = WORK / "full" / st / f"U__{sub}"
            if not (d / "DONE.json").exists() or not (d / "train.log").exists():
                stop(f"arm U {st} U__{sub}: DONE.json or train.log missing")
            U[(st, sub)] = {"results": json.loads((d / "DONE.json").read_text())["results"], "log": d / "train.log"}
    for sub in SUBSETS:
        rows = S["evaluation"][f"U__{sub}"]
        if len(rows) != len(EC.CFG["ref"]): stop("arm U summary rows differ from reference rows")
        for r, row in zip(EC.CFG["ref"], rows):
            m, _ = ms([p_of(U[(st, sub)]["results"], r["column"], r["table2_noise"].lower(), "") for st, _ in PAIRS])
            if (m is None) != (row["mean"] is None) or (m is not None and abs(m - row["mean"]) > 1e-12):
                stop(f"arm U {sub} {r['value']}: recomputed {m} != summary {row['mean']}")
    return U

def published_delta():
    p = ROOT / "spec/simprocess_table2.csv"
    lines = [l for l in p.read_text().splitlines() if l.strip() and not l.lstrip('"').startswith("#")]
    rows = list(csv.DictReader(lines))
    dk = [k for k in (rows[0] if rows else {}) if k and "delta" in k.lower()]
    if len(dk) != 1: return {}, f"no unique Delta column in {p.name}: {list(rows[0]) if rows else []}"
    out = {}
    for r in EC.CFG["ref"]:
        hit = [t for t in rows if r["value"].strip() in [str(v).strip() for v in t.values()]]
        if len(hit) == 1:
            try: out[r["value"]] = float(hit[0][dk[0]])
            except (TypeError, ValueError): pass
    return out, None

def extract_dyn(dy, D):
    d = DYN / dy / "ext"; done = d / "DONE.json"; fd = d / "extraction/features"
    if done.exists():
        X = json.loads(done.read_text())
        if X.get("runner_sha256") == sha(RUNNER_PATH) and all((fd / n).exists() and sha(fd / n) == h for n, h in X["features"].items()):
            say(f"   {dy}: extraction reused ({len(X['features'])} feature files, hashes verified; took {fmt(X['seconds'])})")
            return X
        stop(f"{dy}: {done} does not verify; remove {d} and rerun")
    if d.exists(): shutil.rmtree(d)
    (d / "dataset").mkdir(parents=True)
    for p in D["real"] + D["sources"]: shutil.copy2(p, d / "dataset" / p.name)
    say(f"   {dy}: extracting features from {len(D['real'])} real + {len(D['sources'])} dynamic files (upstream extract_features)")
    secs, errs = run_job(f"{dy} extract", "extract", d / "extraction", d)
    names = json.loads((d / "extraction/feature_files.json").read_text())
    X = {"runner_sha256": sha(RUNNER_PATH), "features": {n: sha(fd / n) for n in names}, "seconds": secs,
         "n_errors": len(errs), "errors": errs[:10]}
    done.write_text(json.dumps(X, indent=1))
    say(f"   {dy}: extraction done in {fmt(secs)} | {len(names)} feature files | {len(errs)} upstream ERROR lines")
    return X

def combine(st, dy, X):
    import pandas as pd
    sx = json.loads((WORK / st / "ext/DONE.json").read_text()); sfd = WORK / st / "ext/extraction/features"
    nx = json.loads((WORK / st / "ext_no6/DONE.json").read_text()); nfd = WORK / st / "ext_no6"
    for n, h in sx["features"].items():
        if sha(sfd / n) != h: stop(f"{st}: arm U feature file {n} changed since extraction")
    for n, h in nx["features"].items():
        if sha(nfd / n) != h: stop(f"{st}: arm U no-EPIC6 file {n} changed")
    order = json.loads((WORK / st / "ext/extraction/feature_files.json").read_text())
    if sorted(order) != sorted(sx["features"]) or sorted(X["features"]) != sorted(order):
        stop(f"{dy}: feature file set differs from arm U {st}")
    base = DYN / dy; done = base / "COMBINED.json"
    frm = {"static": sx["features"], "dynamic": X["features"]}
    if done.exists():
        C = json.loads(done.read_text())
        if C["from"] == frm and all(sha(base / f"combined_{s}" / n) == h for s in SUBSETS for n, h in C[s].items()):
            return C
        stop(f"{done} does not verify; remove {base}/combined_* and COMBINED.json and rerun")
    for s in SUBSETS:
        if (base / f"combined_{s}").exists(): shutil.rmtree(base / f"combined_{s}")
        (base / f"combined_{s}").mkdir(parents=True)
    dfd = base / "ext/extraction/features"
    C = {"from": frm, "order": order, "all": {}, "no6": {}, "real_rows": {}, "dyn_rows": {}, "test_rows": {"all": {}, "no6": {}}}
    for n in order:
        sl = (sfd / n).read_text().splitlines(keepends=True)
        dl = (dfd / n).read_text().splitlines(keepends=True)
        if not sl or not dl or sl[0] != dl[0]: stop(f"{dy} {n}: header differs from arm U")
        hdr = next(csv.reader([sl[0]]))
        if any(h.startswith("Unnamed") for h in hdr): stop(f"{n}: index column in features file")
        ks, kr = hdr.index("source"), hdr.index("real")
        def sr(l):
            f = next(csv.reader([l])); return f[ks], float(f[kr])
        s_real, d_real, d_dyn = {}, {}, []
        for l in sl[1:]:
            s, r = sr(l)
            if s.startswith("dynamic_"): stop(f"{st} {n}: static file already has dynamic rows")
            if r == 1: s_real.setdefault(s, []).append(l)
        for l in dl[1:]:
            s, r = sr(l)
            if s.startswith("dynamic_"):
                if r != 0: stop(f"{dy} {n}: dynamic row labelled real")
                d_dyn.append(l)
            else:
                if r != 1: stop(f"{dy} {n}: non-dynamic, non-real row {s}")
                d_real.setdefault(s, []).append(l)
        if d_real != s_real: stop(f"{dy} {n}: real rows differ from arm U {st} (sources {sorted(d_real)} vs {sorted(s_real)})")
        if not d_dyn: stop(f"{dy} {n}: no dynamic rows extracted")
        if not sl[-1].endswith("\n"): sl[-1] += "\n"
        if not d_dyn[-1].endswith("\n"): d_dyn[-1] += "\n"
        (base / "combined_all" / n).write_text("".join(sl + d_dyn))
        no6_static = [sl[0]] + [l for l in sl[1:] if sr(l)[0] != EC.DROP]
        if "".join(no6_static).rstrip("\n") != (nfd / n).read_text().rstrip("\n"):
            stop(f"{dy} {n}: static part without {EC.DROP} differs from arm U ext_no6")
        (base / "combined_no6" / n).write_text("".join(no6_static + d_dyn))
        for s in SUBSETS:
            f = base / f"combined_{s}" / n
            C[s][n] = sha(f)
            df = pd.read_csv(f).dropna()
            k = int(df["source"].astype(str).str.startswith("dynamic_").sum())
            if k == 0: stop(f"{dy} {s} {n}: no dynamic test rows after dropna")
            C["test_rows"][s][n] = k
        C["real_rows"][n] = sum(len(v) for v in s_real.values()); C["dyn_rows"][n] = len(d_dyn)
    done.write_text(json.dumps(C, indent=1))
    return C

def sizes_by_column(log):
    parts = re.split(r"Training model for (\S+)", Path(log).read_text(errors="replace"))
    out = {}
    for i in range(1, len(parts), 2):
        out[parts[i]] = [(int(a), int(b), c) for a, b, c in re.findall(
            r"Training data size: (\d+) .*?Testing data size: (\d+), ratio: [^\n]*?, dynamic: (True|False)", parts[i + 1], re.S)]
    return out

def train_dyn(st, dy, sub, C, U):
    a = DYN / dy / f"U__{sub}"; done = a / "DONE.json"
    if done.exists():
        R0 = json.loads(done.read_text())
        if R0.get("runner_sha256") == sha(RUNNER_PATH) and R0.get("inputs") == C[sub]:
            R0["reused"] = True; return R0
    if a.exists(): shutil.rmtree(a)
    (a / "features").mkdir(parents=True)
    for n in C["order"]: shutil.copy2(DYN / dy / f"combined_{sub}" / n, a / "features" / n)
    (a / "feature_files.json").write_text(json.dumps(C["order"]))
    secs, errs = run_job(f"{dy} U__{sub}", "train", a, a)
    T = json.loads((a / "trained_models.json").read_text())
    if T["dynamic_calls"] != C["order"]: stop(f"{dy} {sub}: dynamic flag set on {len(T['dynamic_calls'])} of {len(C['order'])} calls")
    if len(T["trained"]) != len(C["order"]): stop(f"{dy} {sub}: {len(T['trained'])} of {len(C['order'])} models trained")
    rk = EC.ranking_source(a / "train.log")
    bad = [c for c in EC.REF_COLS if rk.get(EC.model_name(c)) != "computed"]
    if bad: stop(f"{dy} {sub}: ranking not computed fresh for {bad}")
    res = EC.parse_reports(a / "models")
    for c in EC.REF_COLS:
        rows = list(csv.DictReader(open(a / "models" / f"report_{EC.model_name(c)}.csv", newline="")))
        if any(r.get("dynamic") != "True" for r in rows): stop(f"{dy} {sub} {c}: report rows not dynamic=True")
        for m, v in res[c].items():
            nd = [k for k in v["mean_probas"] if not k.startswith("dynamic_")]
            if nd: stop(f"{dy} {sub} {c} {m}: test set has non-dynamic sources {nd[:3]}")
    got, ref = sizes_by_column(a / "train.log"), sizes_by_column(U[(st, sub)]["log"])
    for n in C["order"]:
        mn = n.replace("combined_", "").replace("_features.csv", "")
        g, r = got.get(mn, []), ref.get(mn, [])
        if len(g) != len(EC.MODELS) or len(r) != len(EC.MODELS): stop(f"{dy} {sub} {mn}: size lines {len(g)} / arm U {len(r)}")
        if [x[0] for x in g] != [x[0] for x in r]: stop(f"{dy} {sub} {mn}: training sizes {[x[0] for x in g]} != arm U {[x[0] for x in r]}")
        if any(x[1] != C["test_rows"][sub][n] for x in g): stop(f"{dy} {sub} {mn}: test sizes {[x[1] for x in g]} != {C['test_rows'][sub][n]}")
        if any(x[2] != "True" for x in g) or any(x[2] != "False" for x in r): stop(f"{dy} {sub} {mn}: dynamic flags in logs unexpected")
    mh = {}
    for p in sorted((a / "models").glob("*.joblib")):
        mh[p.name] = sha(p); p.unlink()
    R = {"runner_sha256": sha(RUNNER_PATH), "inputs": C[sub], "pair": [st, dy], "subset": sub, "seconds": secs,
         "n_errors": len(errs), "errors": errs[:10], "results": res, "sizes": got, "model_sha256": mh}
    done.write_text(json.dumps(R))
    return R

def evaluate(R, U, pub):
    E = {}
    rf = EC.MODELS[0]
    for sub in SUBSETS:
        rows = []
        for r in EC.CFG["ref"]:
            col, nz = r["column"], r["table2_noise"].lower()
            s = [p_of(U[(st, sub)]["results"], col, nz, "") for st, _ in PAIRS]
            d = [p_of(R[(dy, sub)]["results"], col, nz, "dynamic_") for _, dy in PAIRS]
            dl = [y - x for x, y in zip(s, d) if x is not None and y is not None]
            acc = [R[(dy, sub)]["results"][col][rf]["accuracy"] for _, dy in PAIRS]
            rows.append(dict(value=r["value"], column=col, noise=nz, primary=r["primary"] == "1",
                             static=ms(s), dynamic=ms(d), delta=ms(dl), n_pairs=len(dl),
                             published_delta=pub.get(r["value"]), rf_accuracy_dynamic=statistics.fmean(acc),
                             static_per_pair=s, dynamic_per_pair=d))
        E[sub] = rows
    return E

def f3(x):
    return "n/a" if x is None else f"{x:.3f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"EPIC dynamic test (D48; D49 r5; exploratory) | repo {head} | upstream {EC.COMMIT[:7]} | {'SMOKE' if a.smoke else 'FULL'}")
    EC.CFG = EC.preconditions()
    EC.REF_COLS = [r["column"] for r in EC.CFG["ref"]]
    if json.loads((ROOT / "spec/epic_classify_smoke.json").read_text())["script_sha256"] != sha(ROOT / "scripts/epic_classify.py"):
        stop("scripts/epic_classify.py differs from the version that produced arm U")
    if EC.tree_hashes(WORK / "code/U") != EC.tree_hashes(EC.UP): stop("code/U differs from pinned upstream")
    RUNNER_PATH.write_text(RUNNER)
    DI, real, gens, noises = inputs()
    EC.GENERATORS = gens
    U = load_armU()
    pub, note = published_delta()
    say(f"   preconditions OK | upstream pristine | D50 manifest verified (static + dynamic) | arm U summary = D65, "
        f"static values recomputed exactly | {len(real)} real files | {len(DI[PAIRS[0][1]]['sources'])} dynamic sources per generation")
    say(f"   published Delta found for {len(pub)} of {len(EC.CFG['ref'])} rows" + (f" ({note})" if note else ""))
    me = Path(__file__).resolve()
    if a.smoke:
        st, dy = PAIRS[0]
        X = extract_dyn(dy, DI[dy])
        C = combine(st, dy, X)
        say(f"   gate: real rows identical to arm U {st} in {len(C['order'])} of {len(C['order'])} files ({sum(C['real_rows'].values())} rows)")
        say(f"   {dy}: dynamic rows {sum(C['dyn_rows'].values())} | test rows after dropna {sum(C['test_rows']['all'].values())}")
        r = train_dyn(st, dy, "all", C, U)
        say(f"   [smoke] {dy} U__all: {fmt(r['seconds'])} | {len(r['model_sha256'])} models | {r['n_errors']} ERROR lines")
        say(f"   gate: training sizes = arm U for all {len(C['order'])} columns x {len(EC.MODELS)} models; test = dynamic rows only; dynamic: True in every fit")
        proj = (f"4 more extractions in parallel {fmt(X['seconds'])} to {fmt(4 * X['seconds'])}; "
                f"9 more trainings x {fmt(r['seconds'])} = {fmt(9 * r['seconds'])}")
        say(f"   projection FULL: {proj}")
        out = {"stage": "epic_dynamic_smoke", "values_displayed": False, "repo_head": head, "script_sha256": sha(me),
               "runner_sha256": sha(RUNNER_PATH), "epic_classify_sha256": sha(ROOT / "scripts/epic_classify.py"),
               "upstream_commit": EC.COMMIT, "pair": f"{dy} with {st}", "extraction_seconds": X["seconds"],
               "extraction_errors": X["n_errors"], "training_seconds": r["seconds"], "training_errors": r["n_errors"],
               "real_rows": C["real_rows"], "dyn_rows": C["dyn_rows"], "test_rows": C["test_rows"],
               "test_rows_total": sum(C["test_rows"]["all"].values()), "published_delta_rows": len(pub),
               "projection_text": proj, "seconds": time.time() - T0}
        (ROOT / "spec/epic_dynamic_smoke.json").write_text(json.dumps(out, indent=1))
        say(f"EPIC DYNAMIC SMOKE PASSED in {fmt(time.time() - T0)} -> spec/epic_dynamic_smoke.json (no P(real) value shown)")
        return
    sj = ROOT / "spec/epic_dynamic_smoke.json"
    if not sj.exists() or json.loads(sj.read_text())["script_sha256"] != sha(me): stop("smoke record missing or script changed since the smoke run")
    if not re.search(r"^D70 ", (ROOT / "spec/decisions.md").read_text(), re.M): stop("D70 not committed")
    with ThreadPoolExecutor(4) as ex:
        Xs = dict(zip([dy for _, dy in PAIRS], ex.map(lambda p: extract_dyn(p[1], DI[p[1]]), PAIRS)))
    Cs = {}
    for st, dy in PAIRS:
        Cs[dy] = combine(st, dy, Xs[dy])
        say(f"   {dy}: real rows identical to arm U {st} in all {len(Cs[dy]['order'])} files | test rows {sum(Cs[dy]['test_rows']['all'].values())}")
    jobs = [(st, dy, sub) for st, dy in PAIRS for sub in SUBSETS]
    R, times = {}, []
    for k, (st, dy, sub) in enumerate(jobs, 1):
        r = train_dyn(st, dy, sub, Cs[dy], U)
        if not r.get("reused"): times.append(r["seconds"])
        left = statistics.fmean(times) * (len(jobs) - k) if times else 0
        say(f"   [{k}/{len(jobs)}] {dy} U__{sub}: {fmt(r['seconds'])}{' (reused)' if r.get('reused') else ''} | "
            f"elapsed {fmt(time.time() - T0)} | left ~{fmt(left)}")
        R[(dy, sub)] = r
    E = evaluate(R, U, pub)
    for sub in SUBSETS:
        say(f"== dynamic test, {'all real files' if sub == 'all' else 'EPIC6 removed'}: RandomForest P(real) of the Table 2 noise, "
            "max over Panda/Mosaik, mean (sd) over 5 pairs; Delta = dynamic - static paired")
        for r in E[sub]:
            say(f"   {r['value']:<15} {r['noise']:<16} static {f3(r['static'][0])} | dynamic {f3(r['dynamic'][0])} (sd {f3(r['dynamic'][1])}) | "
                f"Delta {f3(r['delta'][0])} (sd {f3(r['delta'][1])}, n {r['n_pairs']}) | published {f3(r['published_delta'])} | "
                f"RF acc {f3(r['rf_accuracy_dynamic'])}{'' if r['primary'] else ' (not primary)'}")
    S = {"stage": "epic_dynamic_full", "repo_head": head, "script_sha256": sha(me), "upstream_commit": EC.COMMIT,
         "evaluation": E, "jobs": {f"{dy}/U__{sub}": {k2: R[(dy, sub)][k2] for k2 in ("seconds", "n_errors", "errors", "model_sha256")}
                                    for _, dy, sub in jobs},
         "extraction": {dy: {k2: Xs[dy][k2] for k2 in ("seconds", "n_errors")} for dy in Xs}, "seconds": time.time() - T0}
    f = DYN / "summary.json"; f.write_text(json.dumps(S, indent=1))
    say(f"EPIC DYNAMIC FULL DONE in {fmt(time.time() - T0)} -> {f} sha256 {sha(f)}")

if __name__ == "__main__":
    main()
