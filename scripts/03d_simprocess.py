"""Stage 3d-1 SimProcess (D45/D46): features + random forests on held-out normal and P-syn windows.
Reads results/phase3/decoy (stage 3c); writes results/phase3/simprocess. Chunks are atomic; rerun resumes.
--smoke: small subset into results/phase3/simprocess_smoke (deleted first); classifiers only real vs noisy-real."""
import sys, os, json, time, hashlib, shutil, argparse, subprocess, warnings, importlib, signal
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in ("src", "scripts", "third_party/simprocess_7533f59"):
    sys.path.insert(0, str(ROOT / _p))
sp = importlib.import_module("03d_sp_smoke")      # feature code exactly as smoke-tested (80b97be)

CFG_HASH = "59a11b235108"
VENDORED = {"core.py": "74439d1927b84f94c57548f128b469f81e7d9df2",
            "data.py": "a7ffae87350af79116468b2a96274b0eecebfab2",
            "features.py": "b9115c2afa16c65b40b1cddb0ff810da4bf52785"}
DECOY = ROOT / "results/phase3/decoy"
SEED = 4545
QUOTA = {"hai": {"hai-test1": 26, "hai-test2": 134}, "qappd": {"train_9": 80, "train_10": 80}}
PSYN_N = {"hai": 280, "qappd": None}               # None = all P-syn windows
CSEEDS = list(range(5))
N_OUT, NF_PER_TAG = 77, 62
CAP_GB, MIN_FREE_GB = 15.0, 5.0
SMOKE_N, SMOKE_PSYN = 8, 4
LOG = None

def say(msg):
    print(msg, flush=True)
    if LOG is not None:
        with open(LOG, "a") as fh: fh.write(msg + "\n")

def stop(msg):
    say("STOP: " + msg); sys.exit(1)

def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)

def fmt(s):
    s = int(s); return f"{s // 3600}h{s % 3600 // 60:02d}m"

# ---------- workers ----------
def _init_worker(cols):
    sp._init(cols)
    signal.signal(signal.SIGINT, signal.SIG_IGN)                 # parent handles Ctrl-C
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)                 # upstream warning flood off the console

def job(args):
    with warnings.catch_warnings(record=True) as rec:            # counted, not printed; values unaffected
        warnings.simplefilter("always")
        f = sp.feats(args)
    return f, dict(Counter(w.category.__name__ for w in rec))

# ---------- selection ----------
def halves(idx, L):
    by = {}
    for i, e in enumerate(idx): by.setdefault(e["file"], []).append((int(e["start"]), i))
    out = {}
    for f in sorted(by):
        v = sorted(by[f]); b = v[len(v) // 2][0]
        out[f] = {"boundary": b, "train": [i for s, i in v if s + L <= b], "test": [i for s, i in v if s >= b + L]}
    return out

def select(pn, idx_n, idx_p, L):
    rng = np.random.default_rng(SEED)
    hv = halves(idx_n, L)
    if sorted(hv) != sorted(QUOTA[pn]): stop(f"{pn} normal files {sorted(hv)} != {sorted(QUOTA[pn])}")
    sel = {"train": [], "test": []}
    for f in sorted(hv):
        for half in ("train", "test"):
            pool, q = hv[f][half], QUOTA[pn][f]
            if len(pool) < q: stop(f"{pn} {f} {half}: {len(pool)} windows < quota {q}")
            sel[half] += sorted(int(i) for i in rng.choice(pool, q, replace=False))
    strata = {}
    for i, e in enumerate(idx_p): strata.setdefault((e["set"], e["kind"], e["tag"]), []).append(i)
    keys, N = sorted(strata), PSYN_N[pn]
    if N is None:
        alloc = {k: len(strata[k]) for k in keys}
    else:
        exact = {k: N * len(strata[k]) / len(idx_p) for k in keys}
        alloc = {k: int(np.floor(exact[k])) for k in keys}
        for k in sorted(keys, key=lambda k: (alloc[k] - exact[k], k))[:N - sum(alloc.values())]: alloc[k] += 1
    ps = []
    for k in keys:
        ps += list(strata[k]) if alloc[k] == len(strata[k]) else [int(i) for i in rng.choice(strata[k], alloc[k], replace=False)]
    return {"L": L, "seed": SEED, "halves": hv, "normal_train": sel["train"], "normal_test": sel["test"],
            "p_syn": sorted(ps), "p_syn_alloc": {"|".join(map(str, k)): alloc[k] for k in keys}}

# ---------- context ----------
def context(pn, cfg, data, out, smoke):
    P = data.load_plant(cfg, pn); ncu = P.nc + P.ne
    allc = list(P.info["columns"]); cols = allc[ncu:]
    mu, sd = np.asarray(P.info["mu"])[ncu:], np.asarray(P.info["sd"])[ncu:]
    W, H = P.W, P.H; win = sp.WIN[pn] or H; eps = sp.EPS[pn]
    del P
    idx = {s: json.loads((DECOY / pn / s / "index.json").read_text()) for s in ("normal", "p_syn")}
    outs = {}
    for s in ("normal", "p_syn"):
        x = np.load(DECOY / pn / s / "X.npy", mmap_mode="r")
        if x.shape != (len(idx[s]), W + H, len(allc)): stop(f"{pn} {s} X.npy shape {x.shape}")
        outs[s] = sorted(p.name[:-4] for p in (DECOY / pn / s).glob("*.npy") if p.name != "X.npy")
        f64 = []
        for o in outs[s]:
            y = np.load(DECOY / pn / s / f"{o}.npy", mmap_mode="r")
            if y.shape != (len(idx[s]), H, len(cols)) or y.dtype not in (np.float32, np.float64):
                stop(f"{pn} {s} {o}.npy shape {y.shape} dtype {y.dtype}")
            if y.dtype == np.float64: f64.append(o)
        say(f"   {pn} {s}: {len(outs[s])} decoy outputs, shape ({len(idx[s])}, {H}, {len(cols)}), float64 (cast to float32): {f64}")
    if outs["normal"] != outs["p_syn"] or len(outs["normal"]) != N_OUT: stop(f"{pn}: decoy output lists differ or != {N_OUT}")
    sel = select(pn, idx["normal"], idx["p_syn"], W + H)
    np_ = out / f"feature_names_{pn}.json"
    ctx = dict(pn=pn, cols=cols, mu=mu, sd=sd, W=W, H=H, ncu=ncu, win=win, eps=eps, outs=outs["normal"], sel=sel,
               names_path=np_, names=json.loads(np_.read_text()) if np_.exists() else None)
    tr, te, ps = sel["normal_train"], sel["normal_test"], sel["p_syn"]
    if smoke:
        tr, te, ps = tr[:SMOKE_N], te[:SMOKE_N], ps[:SMOKE_PSYN]
        on, op = ["real", "NOISY", "C1_s0", "B2"], ["C1_s0", "B2"]
    else:
        on, op = ["real"] + ctx["outs"], ctx["outs"]
    ctx["plan"] = [("normal", o, tr + te) for o in on] + [("p_syn", o, ps) for o in op]
    ctx["ntr"] = len(tr)
    for f, h in sel["halves"].items():
        say(f"   {pn} {f}: {len(h['train'])} | {len(h['test'])} windows in first | second half after purge "
            f"(boundary row {h['boundary']}), draw {QUOTA[pn][f]} per half")
    say(f"   {pn} P-syn: {len(ps) if smoke else len(sel['p_syn'])} of {len(idx['p_syn'])} windows; by set "
        f"{dict(Counter(idx['p_syn'][i]['set'] for i in sel['p_syn']))}")
    return ctx

# ---------- features ----------
def source(ctx, s, o, ids):
    d = DECOY / ctx["pn"] / s
    if o in ("real", "NOISY"):
        f = d / "X.npy"; A = np.asarray(np.load(f, mmap_mode="r")[ids][:, ctx["W"]:, ctx["ncu"]:])
    else:
        f = d / f"{o}.npy"; A = np.asarray(np.load(f, mmap_mode="r")[ids])
    A = A.astype(np.float32).astype(np.float64) * ctx["sd"] + ctx["mu"]      # physical units (B2: float32 first, D42)
    if o == "NOISY":
        A = A + np.random.default_rng(7).normal(0, 1, A.shape) * np.diff(A, axis=1).std((0, 1))
    return A, sha256(f)

def to_rows(f, ctx):
    names = list(f.columns)
    if ctx["names"] is None:
        if len(names) != NF_PER_TAG * len(ctx["cols"]): stop(f"{ctx['pn']}: {len(names)} features, expected {NF_PER_TAG * len(ctx['cols'])}")
        ctx["names"] = names; wjson(ctx["names_path"], names)
    elif names != ctx["names"]:
        stop(f"{ctx['pn']}: feature columns differ from {ctx['names_path'].name}")
    with np.errstate(over="ignore", invalid="ignore"):
        a = f.to_numpy(np.float64).astype(np.float32)
    bad = ~np.isfinite(a); a[bad] = 0
    return a, int(bad.sum())

class Prog:
    def __init__(self, total):
        self.total, self.done, self.t0, self.last = max(total, 1), 0, time.time(), time.time()
    def skip(self, n): self.done += n
    def tick(self, label, j, n):
        self.done += 1; now = time.time()
        if now - self.last >= 60 or j == n:
            self.last = now; el = now - self.t0; r = el / self.done
            say(f"   {label} | window {j}/{n} | {r:.2f} s/window | elapsed {fmt(el)} | left ~{fmt(r * (self.total - self.done))}"
                f" | {100 * self.done / self.total:.1f}% of this session")

def disk_check(out):
    used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    free = shutil.disk_usage(out).free / 1e9
    if used > CAP_GB: stop(f"output {used:.2f} GB exceeds cap {CAP_GB} GB (finished chunks kept)")
    if free < MIN_FREE_GB: stop(f"only {free:.1f} GB free, minimum {MIN_FREE_GB} GB (finished chunks kept)")
    return used, free

def run_chunk(ex, ctx, out, s, o, ids, prog, label):
    d = out / "feat" / ctx["pn"] / s; npz, meta = d / f"{o}.npz", d / f"{o}.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        if m["ids"] == ids and npz.exists() and sha256(npz) == m["npz_sha256"]: return None
        stop(f"{meta} does not match its feature file or window ids; move both files away and rerun")
    t0 = time.time(); A, src = source(ctx, s, o, ids)
    fin = np.isfinite(A).all((1, 2))
    jobs = [(A[i], ctx["cols"], ctx["win"], ctx["eps"]) for i in np.flatnonzero(fin)]
    res = []
    for r in ex.map(job, jobs, chunksize=1):
        res.append(r); prog.tick(label, len(res), len(jobs))
    prog.skip(int((~fin).sum()))
    it = iter(res); F, rows, warn, bad = [], [], Counter(), 0
    for i in range(len(A)):
        if not fin[i]: rows.append(0); continue          # non-finite decoy window: no features = detected (D45)
        f, wc = next(it); warn.update(wc)
        exp = sp.expected_rows(A[i], ctx["win"], ctx["eps"])
        if len(f) != exp: stop(f"{label} window {ids[i]}: {len(f)} feature rows, independent filter count {exp}")
        rows.append(len(f))
        if len(f):
            a, b = to_rows(f, ctx); F.append(a); bad += b
    nf = len(ctx["names"]) if ctx["names"] else 0
    F = np.concatenate(F) if F else np.zeros((0, nf), np.float32)
    d.mkdir(parents=True, exist_ok=True); tmp = d / f"{o}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, F=F, rows=np.asarray(rows, np.int32), ids=np.asarray(ids, np.int64), nonfinite_window=~fin)
    os.replace(tmp, npz)
    m = {"plant": ctx["pn"], "set": s, "output": o, "ids": ids, "windows": len(ids), "rows_total": int(sum(rows)),
         "zero_pass": int(sum(1 for r, ok in zip(rows, fin) if ok and r == 0)), "nonfinite_windows": int((~fin).sum()),
         "nonfinite_values": bad, "warnings": dict(warn), "seconds": time.time() - t0,
         "source_sha256": src, "npz_sha256": sha256(npz)}
    wjson(meta, m)
    return m

def features(ctxs, out, procs):
    plan = [(ctx, s, o, ids) for ctx in ctxs for s, o, ids in ctx["plan"]]
    pend = [c for c in plan if not (out / "feat" / c[0]["pn"] / c[1] / f"{c[2]}.json").exists()]
    nwin = sum(len(c[3]) for c in pend)
    say(f"== features: {len(plan)} chunks, {len(plan) - len(pend)} already done, {nwin} windows to compute, {procs} processes")
    prog = Prog(nwin); k = 0
    for ctx in ctxs:
        ex = ProcessPoolExecutor(procs, initializer=_init_worker, initargs=(ctx["cols"],))
        try:
            for c in [c for c in plan if c[0] is ctx]:
                k += 1; used, free = disk_check(out)
                label = f"[{k}/{len(plan)}] {ctx['pn']} {c[1]} {c[2]}"
                m = run_chunk(ex, ctx, out, c[1], c[2], c[3], prog, label)
                if m:
                    say(f"   done {label}: {m['windows']} windows, rows {m['rows_total']}, zero-pass {m['zero_pass']}, "
                        f"non-finite windows {m['nonfinite_windows']}, values set to 0 {m['nonfinite_values']}, "
                        f"{m['seconds']:.0f} s | disk {used:.2f} GB used, {free:.0f} GB free")
        except KeyboardInterrupt:
            say("INTERRUPTED: finished chunks are kept; rerun the same command to resume")
            ex.shutdown(wait=False, cancel_futures=True); sys.exit(130)
        ex.shutdown(wait=True)
    return len(pend)

# ---------- classifiers ----------
def load_feat(out, pn, s, o):
    z = np.load(out / "feat" / pn / s / f"{o}.npz")
    F, rows = z["F"], z["rows"]; off = np.concatenate([[0], np.cumsum(rows)])
    return [F[off[i]:off[i + 1]] for i in range(len(rows))]

def classify(Rtr, Dtr, Rte, Dte, S, cseed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, GroupKFold
    X, y, g = [], [], []
    for i, (r, d) in enumerate(zip(Rtr, Dtr)):                  # group = parent window (real + decoy together)
        for A, lab in ((r, 1), (d, 0)):
            if len(A): X.append(A); y += [lab] * len(A); g += [i] * len(A)
    y = np.array(y); t = time.time(); res = {}
    if not (y == 1).any(): stop("no real training window passes the filter")
    if (y == 0).any():
        gs = GridSearchCV(RandomForestClassifier(random_state=cseed, n_jobs=-1), {"n_estimators": [50, 200]},
                          cv=GroupKFold(5), scoring="accuracy").fit(np.concatenate(X), y, groups=np.array(g))
        if list(gs.classes_) != [0, 1]: stop(f"classes {gs.classes_}")
        p = lambda A: float(gs.predict_proba(A)[:, 1].mean()) if len(A) else None
        res.update(best_n_estimators=int(gs.best_params_["n_estimators"]), cv_accuracy=float(gs.best_score_), one_class=False)
    else:
        p = lambda A: 1.0 if len(A) else None
        res.update(best_n_estimators=None, cv_accuracy=None, one_class=True)
    Pr, Pd, Ps = [p(A) for A in Rte], [p(A) for A in Dte], [p(A) for A in S]
    z = lambda v: 0.0 if v is None else v                       # zero-pass window: P(real) = 0 (D45)
    acc = (sum(z(v) > .5 for v in Pr) + sum(z(v) <= .5 for v in Pd)) / (len(Pr) + len(Pd))
    passing = [v for v in Pd if v is not None]
    res.update(fit_s=time.time() - t, train_rows=int(len(y)), accuracy=float(acc), m2=float(abs(acc - .5)),
               m2b=float(np.mean([z(v) for v in Pd])), m2b_passing_only=float(np.mean(passing)) if passing else None,
               decoy_pass_rate=len(passing) / len(Pd), real_pass_rate=float(np.mean([v is not None for v in Pr])),
               psyn_flag_rate=float(np.mean([z(v) <= .5 for v in Ps])) if Ps else None,
               psyn_pass_rate=float(np.mean([v is not None for v in Ps])) if Ps else None,
               p_real_test_real=Pr, p_real_test_decoy=Pd, p_real_psyn=Ps)
    return res

def rf(ctxs, out, smoke):
    for ctx in ctxs:
        for s, o, ids in ctx["plan"]:
            if not (out / "feat" / ctx["pn"] / s / f"{o}.json").exists(): stop(f"features missing: {ctx['pn']} {s} {o}")
    todo = [(ctx, o, c) for ctx in ctxs for o in (["NOISY"] if smoke else ctx["outs"]) for c in CSEEDS
            if not (out / "pred" / ctx["pn"] / f"{o}_c{c}.json").exists()]
    say(f"== random forests: {len(todo)} to fit")
    t0, n, cache = time.time(), 0, {}
    for ctx, o, c in todo:
        pn, ntr = ctx["pn"], ctx["ntr"]
        if cache.get("key") != (pn, o):
            R = cache["R"] if cache.get("pn") == pn else load_feat(out, pn, "normal", "real")
            D = load_feat(out, pn, "normal", o)
            S = load_feat(out, pn, "p_syn", o) if (out / "feat" / pn / "p_syn" / f"{o}.npz").exists() else []
            if len(D) != len(R): stop(f"{pn} {o}: {len(D)} decoy windows vs {len(R)} real")
            cache = {"key": (pn, o), "pn": pn, "R": R, "D": D, "S": S}
        R, D, S = cache["R"], cache["D"], cache["S"]
        r = classify(R[:ntr], D[:ntr], R[ntr:], D[ntr:], S, c); r.update(plant=pn, output=o, classifier_seed=c)
        wjson(out / "pred" / pn / f"{o}_c{c}.json", r); n += 1; el = time.time() - t0
        say(f"   rf [{n}/{len(todo)}] {pn} {o} c{c}: {r['best_n_estimators']} trees, {r['train_rows']} rows, {r['fit_s']:.0f} s"
            f" | elapsed {fmt(el)} | left ~{fmt(el / n * (len(todo) - n))}")

def summary(ctxs, out, smoke):
    S = {}
    for ctx in ctxs:
        pn = ctx["pn"]; say(f"== {pn}: mean over classifier seeds 0-4 (accuracy, M2, M2b, decoy pass rate, P-syn flag rate)")
        for o in (["NOISY"] if smoke else ctx["outs"]):
            R = [json.loads((out / "pred" / pn / f"{o}_c{c}.json").read_text()) for c in CSEEDS]
            mean = lambda k: (float(np.mean([r[k] for r in R])) if all(r[k] is not None for r in R) else None)
            S[f"{pn}/{o}"] = {k: mean(k) for k in ("accuracy", "m2", "m2b", "m2b_passing_only", "decoy_pass_rate",
                                                     "real_pass_rate", "psyn_flag_rate", "psyn_pass_rate")}
            v = S[f"{pn}/{o}"]; f = lambda x: "  n/a" if x is None else f"{x:.3f}"
            say(f"   {o:10s} acc {f(v['accuracy'])} | M2 {f(v['m2'])} | M2b {f(v['m2b']) if pn == 'hai' else '  n/a'}"
                f" | decoy pass {f(v['decoy_pass_rate'])} | P-syn flag {f(v['psyn_flag_rate'])}")
    wjson(out / "summary.json", S)
    return S

# ---------- main ----------
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", choices=["all", "features", "rf"], default="all")
    a = ap.parse_args()
    if not 1 <= a.procs <= (os.cpu_count() or 1): stop(f"--procs {a.procs}")
    from decoy import data
    if data.cfg_hash() != CFG_HASH: stop("config hash " + data.cfg_hash())
    V = ROOT / "third_party/simprocess_7533f59/simprocess"
    for f, b in VENDORED.items():
        if git("hash-object", str(V / f)) != b: stop(f"vendored {f} changed")
    if git("status", "--porcelain", "--", "scripts/03d_sp_smoke.py"): stop("scripts/03d_sp_smoke.py modified")
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith("results/")]
    if not a.smoke:
        if dirty: stop("tree not clean: " + "; ".join(dirty))
        if not any(l.startswith("D46 ") for l in (ROOT / "spec/decisions.md").read_text().splitlines()):
            stop("D46 not in spec/decisions.md")
    out = ROOT / ("results/phase3/simprocess_smoke" if a.smoke else "results/phase3/simprocess")
    if a.smoke: shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True); LOG = out / "progress.log"
    say(f"Stage 3d-1 SimProcess | commit {git('rev-parse', '--short=7', 'HEAD')}{' (dirty)' if dirty else ''} | config {CFG_HASH}"
        f" | {a.procs} processes | {'SMOKE' if a.smoke else 'FULL'} | out {out}")
    ident = {"script": git("hash-object", str(Path(__file__).resolve())),
             "smoke_script": git("hash-object", str(ROOT / "scripts/03d_sp_smoke.py")), "vendored": VENDORED,
             "config": CFG_HASH, "stage3c_sha256": sha256(ROOT / "results/phase3/stage3c.json"), "smoke": a.smoke}
    ip = out / "identity.json"
    if ip.exists() and json.loads(ip.read_text()) != ident:
        stop(f"experiment mismatch at {out} (code or inputs changed); existing results preserved")
    wjson(ip, ident)
    cfg = data.load_cfg()
    say("== decoy layout and window selection")
    ctxs = [context(pn, cfg, data, out, a.smoke) for pn in ("hai", "qappd")]
    selp, sel = out / "selection.json", {c["pn"]: c["sel"] for c in ctxs}
    if selp.exists() and json.loads(selp.read_text()) != json.loads(json.dumps(sel)): stop("selection differs from selection.json")
    wjson(selp, sel)
    fails = []
    if a.stage in ("all", "features"):
        features(ctxs, out, a.procs)
        if a.smoke:
            d = out / "feat/hai/normal"; z0 = dict(np.load(d / "C1_s0.npz"))
            (d / "C1_s0.npz").unlink(); (d / "C1_s0.json").unlink()
            say("== resume check: deleted hai normal C1_s0, rerunning (all other chunks must be skipped)")
            npend = features(ctxs, out, a.procs); z1 = dict(np.load(d / "C1_s0.npz"))
            same = npend == 1 and all(np.array_equal(z0[k], z1[k]) for k in z0)
            say(f"   resume: {npend} chunk recomputed, identical to first run: {same}")
            if not same: fails.append("resume check")
    if a.stage in ("all", "rf"):
        rf(ctxs, out, a.smoke); S = summary(ctxs, out, a.smoke)
        if a.smoke:
            for pn in ("hai", "qappd"):
                if S[f"{pn}/NOISY"]["accuracy"] < .9: fails.append(f"{pn} real-vs-noisy {S[f'{pn}/NOISY']['accuracy']:.3f}")
    if fails:
        say("SMOKE FAILED: " + "; ".join(fails)); sys.exit(1)
    say("SMOKE 3d-1 PASSED (no decoy output scored)" if a.smoke else f"STAGE 3d-1 DONE -> {out}")

if __name__ == "__main__":
    main()
