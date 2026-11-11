#!/usr/bin/env python
"""Stage 04: ablation tables (protocol sections 4 and 6; D27(f), D28(1), D29(8), D37, D41, D51(a)).

Reads only stored, hash-recorded stage 3d scores. Trains nothing and opens no data file.
Mapping: Full = C1-Full; A1 = C1-A1 (no C2); A2 = C1-A2 (no C3); A3 = A3-L (retrained without
command conditioning, C2/C3 applied), bare A3 listed as A3-bare; A4 = C1-A4 (C2 limits only, D29(8));
A5 = C1 (C1 only; former B9 per D27(f); C1-A5 identical to C1, stored once, D41).
Every numeric leaf of every stored score is tabulated over training seeds 0-4; delta = variant - Full,
paired by training seed. Descriptive only: H3/H4 are tested in 05_statistics.py.
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = os.path.join(ROOT, "results", "phase3")
VARIANTS = [
    ("Full", "C1-Full", "C1 + C2 + C3"),
    ("A1", "C1-A1", "without C2"),
    ("A2", "C1-A2", "without C3"),
    ("A3", "A3-L", "without command conditioning (retrained), C2/C3 applied"),
    ("A3-bare", "A3", "without command conditioning (retrained), no C2/C3"),
    ("A4", "C1-A4", "C2 with limits only (range + symmetric rate, D29(8))"),
    ("A5", "C1", "C1 only, no C2/C3 (former B9, D27(f); C1-A5 identical to C1, D41)"),
]
PLANTS = ("hai", "qappd")
SEEDS = range(5)
SKIP = {"fit_s", "seconds", "seconds_total", "train_rows", "best_n_estimators", "classifier_seed", "one_class"}
COLS = ["plant", "variant", "method", "metric", "full_mean", "full_sd", "variant_mean", "variant_sd",
        "delta_mean", "delta_sd", "n_seeds"]


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


def flat(x, pre, out):
    if x is None or isinstance(x, (bool, str)):
        return
    if isinstance(x, (int, float)):
        out[pre] = float(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP:
                flat(v, f"{pre}/{k}", out)
    elif isinstance(x, list):
        if x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
            out[pre + "[mean]"] = float(np.mean(np.asarray(x, float)))
        else:
            for i, v in enumerate(x):
                flat(v, f"{pre}[{i}]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    commit = git("rev-parse", "--short=7", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from decoy import data
    cfg = data.cfg_hash()
    print(f"Stage 04 ablation | commit {commit}{' (dirty)' if dirty else ''} | config {cfg} | "
          f"{'SMOKE' if a.smoke else 'FULL'}")
    if cfg != "59a11b235108":
        stop(f"config hash {cfg}")
    if not a.smoke and dirty:
        stop("full run needs a clean tree")
    out = "/tmp/consistent_decoy_ablation_smoke" if a.smoke else os.path.join(P3, "ablation")
    if os.path.exists(out) and not a.smoke:
        stop(f"{out} exists; not overwritten")

    dec = open(os.path.join(ROOT, "spec", "decisions.md")).read()
    inputs = {}

    def recorded(p):
        h = sha(p)
        if h in dec:
            via = "decisions.md"
        else:
            fj = os.path.join(os.path.dirname(p), "files.json")
            if os.path.exists(fj) and h in open(fj).read() and sha(fj) in dec:
                via = "files.json (its hash in decisions.md)"
            else:
                stop(f"{os.path.relpath(p, ROOT)} sha256 {h} not recorded")
        inputs[os.path.relpath(p, ROOT)] = h
        print(f"   input {os.path.relpath(p, ROOT)} {h[:12]} recorded via {via}")

    src = {}
    for key, rel in (("metrics", "metrics/metrics.json"), ("seam", "seam/seam.json"),
                     ("clf", "classifiers/summary.json")):
        p = os.path.join(P3, rel)
        if not os.path.exists(p):
            stop(f"missing {p}")
        recorded(p)
        J = json.load(open(p))
        pv = J.get("provenance", {})
        if pv.get("dirty") is not False or pv.get("smoke") is not False:
            stop(f"{rel} provenance dirty={pv.get('dirty')} smoke={pv.get('smoke')}")
        src[key] = J["plants"]
    man = os.path.join(ROOT, "spec", "phase3_simprocess_manifest.sha256")
    recorded(man)
    mh = {}
    for line in open(man):
        parts = line.split()
        if len(parts) >= 2:
            mh["/".join(parts[-1].split("/")[-3:])] = parts[0]

    vals, nsp = {}, 0
    for p in PLANTS:
        M = src["metrics"][p]["outputs"]
        for _, meth, _ in VARIANTS:
            for s in SEEDS:
                n, d = f"{meth}_s{s}", {}
                for sec in ("fit", "detector", "dtw", "cfid"):
                    if n not in M[sec]:
                        stop(f"{p} metrics/{sec} lacks {n}")
                    flat(M[sec][n], f"metrics/{sec}", d)
                for key in ("seam", "clf"):
                    O = src[key][p]["outputs"]
                    if n not in O:
                        stop(f"{p} {key} lacks {n}")
                    flat(O[n], key, d)
                cs = []
                for c in range(5):
                    k = f"pred/{p}/{n}_c{c}.json"
                    f = os.path.join(P3, "simprocess", k)
                    if not os.path.exists(f):
                        stop(f"missing {f}")
                    if mh.get(k) != sha(f):
                        stop(f"{k} sha256 not as in D47 manifest")
                    e = {}
                    flat(json.load(open(f)), "simprocess", e)
                    cs.append(e)
                    nsp += 1
                for k in set.intersection(*(set(e) for e in cs)):
                    d[k] = float(np.mean([e[k] for e in cs]))
                vals[(p, meth, s)] = d
    print(f"   SimProcess prediction files checked against manifest: {nsp}")

    rows, means = [], {}
    for p in PLANTS:
        fullp = set.intersection(*(set(vals[(p, "C1-Full", s)]) for s in SEEDS))
        for lab, meth, _ in VARIANTS:
            vp = set.intersection(*(set(vals[(p, meth, s)]) for s in SEEDS))
            paths = sorted(fullp & vp)
            print(f"   {p} {lab:<7} {meth:<8} metrics {len(paths)} | only in Full {len(fullp - vp)} | only in variant {len(vp - fullp)}")
            for m in paths:
                f = np.array([vals[(p, "C1-Full", s)][m] for s in SEEDS])
                v = np.array([vals[(p, meth, s)][m] for s in SEEDS])
                with np.errstate(all="ignore"):
                    dl = v - f
                    r = dict(plant=p, variant=lab, method=meth, metric=m, full_mean=f.mean(), full_sd=f.std(ddof=1),
                             variant_mean=v.mean(), variant_sd=v.std(ddof=1), delta_mean=dl.mean(),
                             delta_sd=dl.std(ddof=1), n_seeds=5)
                rows.append(r)
                means[(p, meth, m)] = r["variant_mean"]

    bad = [r["metric"] for r in rows if r["variant"] == "Full" and np.isfinite(r["delta_mean"]) and r["delta_mean"] != 0]
    if bad:
        stop(f"Full-vs-Full delta not zero: {bad[:5]}")
    print("gate Full-vs-Full OK")

    def m1(p, meth):
        ks = sorted(k for k in vals[(p, meth, 0)] if k.startswith("metrics/fit/p_real_m1/") and k.count("/") == 3)
        return ks, float(np.mean([[vals[(p, meth, s)][k] for k in ks] for s in SEEDS]))
    ks, fm1 = m1("hai", "C1-Full")
    _, cm1 = m1("hai", "C1")
    if len(ks) != 10:
        stop(f"p_real_m1 has {len(ks)} attack entries, expected 10: {ks[:12]}")
    print(f"gate H1 M1 (HAI, mean over 10 attacks): C1-Full {fm1:.4f} (recorded 37.19) | C1 {cm1:.4f} (recorded 21.83)")
    if abs(fm1 - 37.19) > 0.006 or abs(cm1 - 21.83) > 0.006:
        stop("M1 does not reproduce the recorded stage 3d-2 means")
    hits = [m for (p, meth, m), v in means.items() if p == "hai" and meth == "C1-Full" and m.startswith("metrics/detector")
            and round(v, 3) == 0.157 and round(means.get(("hai", "C1", m), -1), 3) == 0.095
            and round(means.get(("hai", "C1-A2", m), -1), 3) == 0.094]
    print(f"info D54 cross-check (C1-Full 0.157, C1 0.095, C1-A2 0.094): {hits if hits else 'no path with all three (aggregation may differ; not a gate)'}")

    print("== metric inventory (per source, HAI)")
    inv = {}
    for r in rows:
        if r["plant"] == "hai" and r["variant"] == "Full":
            g = "/".join(r["metric"].split("/")[:2]) if r["metric"].startswith("metrics") else r["metric"].split("/")[0]
            inv.setdefault(g, []).append(r["metric"])
    for g, ms in inv.items():
        print(f"   {g}: {len(ms)} metrics")
        for m in ms[:30]:
            print(f"      {m}")
        if len(ms) > 30:
            print(f"      ... {len(ms) - 30} more")

    head = ["metrics/fit/normal", "metrics/cfid", "simprocess/accuracy", "simprocess/m2", "simprocess/m2b"]
    for p in PLANTS:
        print(f"== {p} headline (mean over training seeds; M1 = mean over 10 attacks, HAI only)")
        for lab, meth, _ in VARIANTS:
            parts = []
            if p == "hai":
                parts.append(f"M1 {m1(p, meth)[1]:8.3f}")
            for m in head:
                if (p, meth, m) in means:
                    parts.append(f"{m.split('/')[-1]} {means[(p, meth, m)]:.4f}")
            print(f"   {lab:<7} {meth:<8} " + " | ".join(parts))

    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    if a.smoke:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "ablation.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[c] if c in ("plant", "variant", "method", "metric", "n_seeds") else repr(float(r[c])) for c in COLS])
    meta = dict(stage="04_ablation", provenance=dict(commit=commit, dirty=dirty, smoke=a.smoke, config_sha12=cfg,
                script_sha256=sha(os.path.abspath(__file__))), inputs=inputs,
                variants=[dict(label=l, method=m, removes=d) for l, m, d in VARIANTS],
                delta="variant - Full, paired by training seed", n_rows=len(rows), seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(tmp, "ablation.json"), "w"), indent=1)
    fl = {f: sha(os.path.join(tmp, f)) for f in ("ablation.tsv", "ablation.json")}
    json.dump(fl, open(os.path.join(tmp, "files.json"), "w"), indent=1)
    os.rename(tmp, out)
    ht, hj = fl["ablation.tsv"], fl["ablation.json"]
    if a.smoke:
        print(f"SMOKE 04 PASSED | {len(rows)} rows | ablation.tsv sha256 {ht} | {time.time() - t0:.0f} s")
    else:
        print(f"STAGE 04 DONE -> results/phase3/ablation | ablation.tsv sha256 {ht} | ablation.json sha256 {hj} | "
              f"{len(rows)} rows | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
