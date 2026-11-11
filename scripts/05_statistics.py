#!/usr/bin/env python
"""Stage 05: primary statistics, Holm family m = 17 (protocol section 10; D33(d), D45, D51, D60, D61).

Reads stored, hash-recorded stage 3d scores only; trains nothing and opens no data file.
--smoke: every gate and the full computation path with 200 resamples; prints no comparison value.
Full run: clean tree and a D61 line in spec/decisions.md naming this script's sha256.
"""
import argparse, collections, csv, hashlib, json, os, shutil, subprocess, sys, time, tomllib
import numpy as np
import scipy
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = os.path.join(ROOT, "results", "phase3")
FULL = "C1-Full"
BASELINES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
PLANTS = ("hai", "qappd")
RATE_HZ = {"hai": 1, "qappd": 100}
STRIDE = {"hai": 180, "qappd": 300}
CLFS = ("rnn", "mr", "sp")
ALPHA = 0.05
ORDER = [f"H1-{b}" for b in BASELINES] + ["H2-hai", "H2-qappd", "H3-hai", "H3-qappd",
         "H4-GRU-hai", "H4-TF-hai", "H4-GRU-qappd", "H4-TF-qappd"]


def stop(msg):
    print("STOP:", msg)
    sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def outs(names, m):
    o = sorted(n for n in names if n == m or (n.startswith(m + "_s") and n[len(m) + 2:].isdigit()))
    if len(o) != (1 if m in ("B2", "B3") else 5):
        stop(f"{m}: outputs {o}")
    return o


def pdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.isnan(a).any() or np.isnan(b).any():
        stop("NaN in paired values")
    with np.errstate(invalid="ignore"):
        return np.where(a == b, 0.0, a - b)          # equal values (incl. both -inf) -> 0; -inf ranks worst


def signed_rank(d):
    """One-sided exact signed-rank, alternative d > 0; zeros dropped; mid-ranks; null over all 2^n signs."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return dict(n_nonzero=0, w_plus=0.0, w_minus=0.0, p=1.0, rank_biserial=0.0)
    r = stats.rankdata(np.abs(nz))
    r2 = np.rint(2 * r).astype(np.int64)
    if not np.allclose(r2, 2 * r):
        stop("ranks not multiples of 1/2")
    tot = int(r2.sum())
    cnt = np.zeros(tot + 1)
    cnt[0] = 1.0
    for x in r2:
        nxt = cnt.copy()
        nxt[x:] += cnt[:tot + 1 - x]
        cnt = nxt
    p = float(cnt[int(r2[nz > 0].sum()):].sum() / 2.0 ** n)
    wp, wm = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    return dict(n_nonzero=n, w_plus=wp, w_minus=wm, p=min(1.0, p), rank_biserial=(wp - wm) / (wp + wm))


def brute_p(d):
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    r = stats.rankdata(np.abs(nz))
    S = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) @ r
    return float(np.mean(S >= r[nz > 0].sum() - 1e-9))


def holm(ps):
    ps = np.asarray(ps, float)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def median_ci(d, B, seed=2026):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = np.median(d[rng.integers(0, len(d), (B, len(d)))], axis=1)
    meth = "linear" if np.isfinite(d).all() else "nearest"
    return [float(np.percentile(m, 2.5, method=meth)), float(np.percentile(m, 97.5, method=meth))]


def mbb(blocksets, n, B, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((B, n))
    for _, pos, blocks in blocksets:
        ns, nb = len(pos), len(blocks)
        for b in range(B):
            got, tot = [], 0
            while tot < ns:
                k = int(rng.integers(nb))
                got.append(blocks[k])
                tot += len(blocks[k])
            np.add.at(W[b], np.concatenate(got)[:ns], 1.0)
    return W


def method_m2(W, n, G, sp_pos):
    parts = []
    for k in ("rnn", "mr"):
        parts.append(np.abs(W @ G[k].T / n - 0.5).mean(1))
    Ws = W[:, sp_pos]
    den = Ws.sum(1)
    if np.any(den == 0):
        stop("a resample contains no SimProcess window")
    parts.append(np.abs((Ws @ G["sp"].T) / den[:, None] - 0.5).mean(1))
    return np.mean(parts, axis=0), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    commit, dirty = git("rev-parse", "--short=7", "HEAD"), bool(git("status", "--porcelain"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from decoy import data
    cfg = data.cfg_hash()
    me = sha(os.path.abspath(__file__))
    print(f"Stage 05 statistics | commit {commit}{' (dirty)' if dirty else ''} | config {cfg} | scipy {scipy.__version__} "
          f"numpy {np.__version__} | {'SMOKE (no comparison value printed)' if a.smoke else 'FULL'}")
    if cfg != "59a11b235108":
        stop(f"config hash {cfg}")
    dec = open(os.path.join(ROOT, "spec", "decisions.md")).read()
    out = os.path.join(P3, "statistics")
    if not a.smoke:
        if dirty:
            stop("full run needs a clean tree")
        if not any(l.startswith("D61 ") and me in l for l in dec.splitlines()):
            stop("no D61 line naming this script's sha256")
        if os.path.exists(out):
            stop(f"{out} exists; not overwritten")
    ev = tomllib.load(open(os.path.join(ROOT, "configs", "evaluation.toml"), "rb"))["bootstrap"]
    if ev["resamples"] != 10000 or ev["seed"] != 4242:
        stop(f"evaluation.toml [bootstrap] {ev}")
    B = 200 if a.smoke else int(ev["resamples"])
    block_s = {"hai": ev["block_s_hai"], "qappd": ev["block_s_qappd"]}

    inputs = {}

    def rec(rel):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            stop(f"missing {rel}")
        h = sha(p)
        if h not in dec:
            stop(f"{rel} sha256 {h} not recorded in decisions.md")
        inputs[rel] = h
        print(f"   input {rel} {h[:12]} recorded")
        return p

    MJ = json.load(open(rec("results/phase3/metrics/metrics.json")))
    SJ = json.load(open(rec("results/phase3/classifiers/summary.json")))
    for nm, J in (("metrics", MJ), ("classifiers", SJ)):
        pv = J["provenance"]
        if pv.get("dirty") is not False or pv.get("smoke") is not False:
            stop(f"{nm} provenance {pv}")
    if SJ["provenance"]["evaluation_toml_sha256"] != sha(os.path.join(ROOT, "configs", "evaluation.toml")):
        stop("configs/evaluation.toml differs from the one used in stage 3d-3")
    CF = open(rec("results/phase3/classifiers/files.json")).read()
    man = rec("spec/phase3_simprocess_manifest.sha256")
    mh = {}
    for line in open(man):
        parts = line.split()
        if len(parts) >= 2:
            mh["/".join(parts[-1].split("/")[-3:])] = parts[0]
    rules = tomllib.load(open(rec("spec/reporting_rules.toml"), "rb"))
    attacks = list(rules["ablation"]["m1_attacks"])
    S3 = open(rec("results/phase3/stage3c.json")).read()
    selp = os.path.join(P3, "simprocess", "selection.json")
    if sha(selp) not in set(mh.values()):
        stop("selection.json not in the D47 manifest")
    sel = json.load(open(selp))

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        d = rng.normal(0.3, 1.0, int(rng.integers(5, 21)))
        worst = max(worst, abs(signed_rank(d)["p"] - stats.wilcoxon(d, alternative="greater", method="exact").pvalue))
    worst2 = 0.0
    for _ in range(100):
        d = rng.integers(-3, 4, int(rng.integers(4, 13))).astype(float)
        worst2 = max(worst2, abs(signed_rank(d)["p"] - brute_p(d)))
    if worst > 1e-12 or worst2 > 1e-12:
        stop(f"signed-rank check: vs scipy exact {worst:.2e}, vs brute force (ties) {worst2:.2e}")
    if not np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
        stop("Holm check")
    print(f"gate signed-rank OK (300 untied vs scipy exact max |dp| {worst:.1e}; 100 tied vs brute force {worst2:.1e}); Holm OK")

    comps, extra, info = {}, {}, collections.Counter()
    for p in PLANTS:
        Mp = MJ["plants"][p]["outputs"]
        F, D = Mp["fit"], Mp["detector"]
        ip = os.path.join(P3, "decoy", p, "normal", "index.json")
        if sha(ip) not in S3:
            stop(f"{p} normal/index.json not in stage3c.json")
        index = json.load(open(ip))
        TI, G, stored = None, {}, {}
        for m in [FULL] + BASELINES:
            G[m], st = {k: [] for k in CLFS}, {k: [] for k in CLFS}
            for o in outs(F, m):
                for clf in ("rnn", "mr"):
                    accs = []
                    for c in range(5):
                        f = os.path.join(P3, "classifiers", "jobs", p, clf, f"{o}_c{c}.npz")
                        if not os.path.exists(f) or sha(f) not in CF:
                            stop(f"{f} missing or not in classifiers/files.json")
                        Z = np.load(f)
                        meta = json.loads(Z["meta"].item())
                        t = float(meta["threshold"])
                        if TI is None:
                            TI = Z["test_index"].copy()
                        elif not np.array_equal(TI, Z["test_index"]):
                            stop(f"{f} test_index differs")
                        r, d = Z["test_real"], Z["test_decoy"]
                        if not (np.isfinite(r).all() and np.isfinite(d).all()):
                            stop(f"{f} non-finite scores")
                        g = ((r > t).astype(float) + (d <= t).astype(float)) / 2
                        if abs(g.mean() - meta["acc"]) > 1e-12:
                            stop(f"{f} accuracy {g.mean()} != stored {meta['acc']}")
                        G[m][clf].append(g)
                        st[clf].append(abs(meta["acc"] - 0.5))
                        accs.append(meta["acc"])
                    sm = SJ["plants"][p]["outputs"][o][clf]["m2"]
                    info[(p, clf, "per_fit")] += abs(sm - np.mean(np.abs(np.array(accs) - 0.5))) < 1e-9
                    info[(p, clf, "of_mean_acc")] += abs(sm - abs(np.mean(accs) - 0.5)) < 1e-9
                    info[(p, clf, "outputs")] += 1
                for c in range(5):
                    k = f"pred/{p}/{o}_c{c}.json"
                    f = os.path.join(P3, "simprocess", k)
                    if mh.get(k) != sha(f):
                        stop(f"{k} not as in D47 manifest")
                    J = json.load(open(f))
                    rr = np.array([0.0 if v is None else v for v in J["p_real_test_real"]], float)
                    dd = np.array([0.0 if v is None else v for v in J["p_real_test_decoy"]], float)
                    if len(rr) != 160 or len(dd) != 160:
                        stop(f"{k} list lengths {len(rr)}/{len(dd)}")
                    g = ((rr > 0.5).astype(float) + (dd <= 0.5).astype(float)) / 2
                    if abs(g.mean() - J["accuracy"]) > 1e-12:
                        stop(f"{k} accuracy {g.mean()} != stored {J['accuracy']}")
                    G[m]["sp"].append(g)
                    st["sp"].append(abs(J["accuracy"] - 0.5))
            G[m] = {k: np.array(v) for k, v in G[m].items()}
            stored[m] = {k: float(np.mean(v)) for k, v in st.items()}
        nfits = sum(len(G[m]["rnn"]) + len(G[m]["mr"]) + len(G[m]["sp"]) for m in G)
        print(f"gate {p}: {nfits} classifier fits, every stored accuracy reproduced from per-window scores")

        n = len(TI)
        pos = {int(w): i for i, w in enumerate(TI)}
        nt = [int(w) for w in sel[p]["normal_test"]]
        if len(set(nt)) != 160 or any(w not in pos for w in nt):
            stop(f"{p} SimProcess normal_test not a subset of test_index")
        sp_pos = np.array([pos[w] for w in nt])
        files = [index[int(i)]["file"] for i in TI]
        starts = np.array([index[int(i)]["start"] for i in TI])
        L = int(block_s[p] * RATE_HZ[p])
        blocksets = []
        for fname in dict.fromkeys(files):
            ps = np.array([k for k, x in enumerate(files) if x == fname])
            if np.any(np.diff(ps) != 1) or np.any(np.diff(starts[ps]) <= 0):
                stop(f"{p} {fname} test windows not contiguous / ordered")
            stp = starts[ps]
            dif = np.diff(stp)
            if collections.Counter(dif.tolist()).most_common(1)[0][0] != STRIDE[p]:
                stop(f"{p} {fname} start spacing {collections.Counter(dif.tolist()).most_common(3)}")
            allowed = [i for i in range(len(ps)) if stp[i] + L <= stp[-1] + STRIDE[p]]
            blocks = [ps[(stp >= stp[i]) & (stp < stp[i] + L)] for i in allowed] or [ps]
            blocksets.append((fname, ps, blocks))
            sz = [len(b) for b in blocks]
            print(f"   {p} {fname}: test windows {len(ps)} | max start gap {int(dif.max())} rows (stride {STRIDE[p]}) | "
                  f"blocks {len(blocks)} of {L} rows, windows per block {min(sz)}-{max(sz)}")
        one = np.ones((1, n))
        m2full = {}
        for m in G:
            tot, parts = method_m2(one, n, G[m], sp_pos)
            for k, v in zip(CLFS, parts):
                if abs(v[0] - stored[m][k]) > 1e-12:
                    stop(f"{p} {m} {k}: unresampled M2 {v[0]} != stored {stored[m][k]}")
            m2full[m] = dict(mean=float(tot[0]), **{k: float(v[0]) for k, v in zip(CLFS, parts)})
        print(f"gate {p}: bootstrap without resampling reproduces every stored M2")
        best = min(BASELINES, key=lambda b: (m2full[b]["mean"], BASELINES.index(b)))
        W = mbb(blocksets, n, B, int(ev["seed"]))
        if not np.allclose(W.sum(1), n) or any(not np.allclose(W[:, ps].sum(1), len(ps)) for _, ps, _ in blocksets):
            stop(f"{p} resample sizes")
        if a.smoke and not np.array_equal(W, mbb(blocksets, n, B, int(ev["seed"]))):
            stop(f"{p} bootstrap not repeatable")
        dstar = method_m2(W, n, G[FULL], sp_pos)[0] - method_m2(W, n, G[best], sp_pos)[0]
        del W
        W2 = mbb(blocksets, n, B, 2026)
        d2 = method_m2(W2, n, G[FULL], sp_pos)[0] - method_m2(W2, n, G[best], sp_pos)[0]
        del W2
        dobs = m2full[FULL]["mean"] - m2full[best]["mean"]
        comps[f"H2-{p}"] = dict(hypothesis="H2", plant=p, compare=f"{FULL} - {best}", predicted="< 0",
                                test=f"one-sided moving-block bootstrap, p = (1 + #(d* >= 0)) / (B + 1), B {B}, seed {ev['seed']}",
                                unit=f"test-half windows, blocks {block_s[p]} s", n_units=n, n_nonzero=None, w_plus=None,
                                w_minus=None, p=float((1 + np.sum(dstar >= 0)) / (B + 1)), median_diff=float(dobs),
                                ci95=[float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))], rank_biserial=None,
                                best_baseline=best)
        extra[f"m2_{p}"] = m2full

        def rates(m):
            o = outs(D, m)
            keys = sorted(D[o[0]]["p_syn"]["oor"]["per_base"], key=int)
            A = []
            for x in o:
                pb = D[x]["p_syn"]["oor"]["per_base"]
                v = np.array([pb[k] for k in keys], float) if sorted(pb, key=int) == keys else None
                if v is None or len(keys) != 20 or not (np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1):
                    stop(f"{p} {x} per_base malformed")
                A.append(v)
            return keys, np.mean(A, 0), float(np.mean([D[x]["p_syn"]["oor"]["rate"] for x in o]))

        want = {"hai": {"C1-Full": 0.157, "C1": 0.095, "C1-A2": 0.094, "B8": 0.269, "B8-L": 0.157},
                "qappd": {"C1-Full": 0.994, "C1": 1.0, "C1-A2": 1.0}}[p]
        R = {m: rates(m) for m in ("C1-Full", "C1", "C1-A2", "B8", "B8-L")}
        for m, w in want.items():
            if round(R[m][2], 3) != w:
                stop(f"{p} {m} out-of-range rate {R[m][2]:.4f} != recorded {w}")
        print(f"gate {p}: out-of-range detector rates reproduce D54/D58 ({', '.join(f'{m} {w}' for m, w in want.items())})")

        def rank_cmp(hyp, an, bn, av, bv, unit, labels):
            d = pdiff(av, bv)
            r = signed_rank(d)
            return dict(hypothesis=hyp, plant=p, compare=f"{an} - {bn}", predicted="> 0",
                        test="one-sided exact Wilcoxon signed-rank (mid-ranks, zeros dropped, p = 1 if all zero)",
                        unit=unit, n_units=int(len(d)), n_nonzero=r["n_nonzero"], w_plus=r["w_plus"], w_minus=r["w_minus"],
                        p=r["p"], median_diff=float(np.median(d)), ci95=median_ci(d, B), rank_biserial=r["rank_biserial"],
                        units={str(k): [float(x), float(y), float(z)] for k, x, y, z in zip(labels, av, bv, d)})

        keys = R["C1-Full"][0]
        comps[f"H3-{p}"] = rank_cmp("H3", "C1-A2", "C1-Full", R["C1-A2"][1], R["C1-Full"][1], "P-syn base window", keys)
        comps[f"H4-GRU-{p}"] = rank_cmp("H4", "C1", "C1-Full", R["C1"][1], R["C1-Full"][1], "P-syn base window", keys)
        comps[f"H4-TF-{p}"] = rank_cmp("H4", "B8", "B8-L", R["B8"][1], R["B8-L"][1], "P-syn base window", keys)

        if p == "hai":
            def m1(m):
                o = outs(F, m)
                for x in o:
                    if sorted(F[x]["p_real_m1"]) != sorted(attacks):
                        stop(f"{x} p_real_m1 attacks {sorted(F[x]['p_real_m1'])}")
                return np.array([np.mean([F[x]["p_real_m1"][t] for x in o]) for t in attacks])
            for m, w, tol in (("C1-Full", 37.1858, 1e-3), ("C1", 21.8317, 1e-3), ("B2", 20.34, 0.006),
                              ("B8", -12.00, 0.006), ("B8-L", 30.74, 0.006)):
                if abs(m1(m).mean() - w) > tol:
                    stop(f"M1 {m} {m1(m).mean():.4f} != recorded {w}")
            print("gate hai: M1 reproduces (C1-Full 37.1858, C1 21.8317, B2 20.34, B8 -12.00, B8-L 30.74)")
            for b in BASELINES:
                comps[f"H1-{b}"] = rank_cmp("H1", FULL, b, m1(FULL), m1(b), "P-real attack", attacks)
            oc = ["A103", "A105", "A110", "A205"]
            extra["h3_m1_per_attack"] = {t: {"C1-A2": float(m1("C1-A2")[attacks.index(t)]),
                                             "C1-Full": float(m1(FULL)[attacks.index(t)])} for t in oc}

    for (p, clf, k), v in sorted(info.items()):
        if k != "outputs":
            print(f"   info {p}/{clf}: summary.json m2 equals '{k}' for {v} of {info[(p, clf, 'outputs')]} outputs used here")
    if sorted(comps) != sorted(ORDER):
        stop(f"comparisons {sorted(comps)}")
    rows = [dict(id=i, **comps[i]) for i in ORDER]
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
        r["reject_h0"] = bool(q <= ALPHA)
    print(f"gate family: {len(rows)} comparisons (m = 17), Holm at {ALPHA}")
    if len(rows) != 17:
        stop("family size")
    blob = json.dumps(rows, sort_keys=True)
    if a.smoke:
        print(f"   hidden results: {len(rows)} comparisons computed, all p in [0,1]: "
              f"{all(0 <= r['p'] <= 1 for r in rows)} | sha256 {hashlib.sha256(blob.encode()).hexdigest()[:16]}")
        print(f"SMOKE 05 PASSED | resamples {B} | no comparison value printed | {time.time() - t0:.0f} s")
        return

    print(f"== comparisons (one-sided; Holm m = 17, alpha {ALPHA}); value = median paired difference (H2: M2 difference)")
    for r in rows:
        rb = "" if r["rank_biserial"] is None else f" | r_rb {r['rank_biserial']:+.3f}"
        nz = "" if r["n_nonzero"] is None else f" | nonzero {r['n_nonzero']} W+ {r['w_plus']:.1f}"
        print(f"   {r['id']:<13} {r['compare']:<18} pred {r['predicted']} | n {r['n_units']}{nz} | p {r['p']:.5f} | "
              f"p_holm {r['p_holm']:.5f} | {'REJECT H0' if r['reject_h0'] else 'not rejected'} | value {r['median_diff']:+.4f} "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]{rb}")
    for p in PLANTS:
        print(f"== {p} full-sample M2 (mean over fits; method = mean of RNN, MiniRocket, SimProcess)")
        for m, v in extra[f"m2_{p}"].items():
            print(f"   {m:<8} mean {v['mean']:.4f} | rnn {v['rnn']:.4f} | mr {v['mr']:.4f} | sp {v['sp']:.4f}")
    print("== H3 per-attack M1, out-of-coverage P-real (descriptive, no test)")
    for t, v in extra["h3_m1_per_attack"].items():
        print(f"   {t}: C1-A2 {v['C1-A2']:.2f} | C1-Full {v['C1-Full']:.2f}")
    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    json.dump(dict(stage="05_statistics", provenance=dict(commit=commit, dirty=dirty, smoke=False, config_sha12=cfg,
              script_sha256=me, scipy=scipy.__version__, numpy=np.__version__), inputs=inputs,
              settings=dict(alpha=ALPHA, m=17, resamples=B, bootstrap_seed=ev["seed"], ci_seed=2026, block_s=block_s),
              comparisons=rows, extra=extra, seconds=time.time() - t0), open(os.path.join(tmp, "statistics.json"), "w"), indent=1)
    cols = ["id", "hypothesis", "plant", "compare", "predicted", "n_units", "n_nonzero", "w_plus", "p", "p_holm",
            "reject_h0", "median_diff", "ci95", "rank_biserial"]
    with open(os.path.join(tmp, "comparisons.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r[c], float) else r[c] for c in cols])
    fl = {f: sha(os.path.join(tmp, f)) for f in ("statistics.json", "comparisons.tsv")}
    json.dump(fl, open(os.path.join(tmp, "files.json"), "w"), indent=1)
    os.rename(tmp, out)
    k = sum(r["reject_h0"] for r in rows)
    print(f"STAGE 05 DONE -> results/phase3/statistics | statistics.json sha256 {fl['statistics.json']} | "
          f"comparisons.tsv sha256 {fl['comparisons.tsv']} | H0 rejected {k} of 17 | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
