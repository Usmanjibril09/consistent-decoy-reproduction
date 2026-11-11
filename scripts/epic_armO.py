#!/usr/bin/env python3
"""D48(c) Arm O on EPIC: our D45/D46 classifier (classify() verbatim from scripts/03d_simprocess.py, sha256-gated
against the version that reproduced stored HAI/QAPPD results, D66(c)) on the frozen upstream-extracted EPIC features
used by arm U: 5 static generations x {all, no6} x 11 columns; one classifier per (generation, subset, column,
simulated source, classifier seed 0-4). Settings: D68. --smoke prints counts and timings only, no result value."""
import argparse, ast, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / "epic_work/classify"
UPML = Path.home() / "epic_work/upstream_7533f59/Discriminator/simprocess/ml.py"
OUT = ROOT / "results/phase3/epic_armO"
GENS = (0, 1, 2, 3, 4)
SUBS = {"all": "ext/extraction/features", "no6": "ext_no6"}
COLS = ("C1", "C2", "C3", "V1", "V2", "V3", "frequency", "power_apparent", "power_reactive", "power_real", "all_columns")
CSEEDS = (0, 1, 2, 3, 4)
BLOCK, PURGE, MIN_ROWS = 5, 4, 9      # block = one window (20 samples at stride 4); 4 rows dropped -> no shared sample
SMOKE_REC = ROOT / "spec/epic_armO_smoke.json"
SENS_REC = ROOT / "spec/epic_sensitivity_smoke.json"
LOG = [None]

def say(m):
    print(m, flush=True)
    if LOG[0]:
        with open(LOG[0], "a") as f: f.write(m + "\n")

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

def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj)); os.replace(tmp, p)

def our_classify():
    src = (ROOT / "scripts/03d_simprocess.py").read_text()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "classify"]
    need(len(fn) == 1, "classify() not found exactly once in scripts/03d_simprocess.py")
    code = ast.get_source_segment(src, fn[0]); g = {"np": np, "time": time, "stop": stop}
    exec(code, g)
    return g["classify"], hashlib.sha256(code.encode()).hexdigest()

def fpath(g, sub, col):
    return WORK / f"static_s{g}" / SUBS[sub] / f"combined_{col}_features.csv"

def load(g, sub, col):
    tag = f"s{g}/{sub}/{col}"
    df = pd.read_csv(fpath(g, sub, col))
    need({"real", "source"} <= set(df.columns), f"{tag}: no real/source columns")
    s0 = df["source"].astype(str)
    need(int((s0 != s0.shift()).sum()) == s0.nunique(), f"{tag}: a source's rows are not contiguous")
    mask = df.isna().any(axis=1)
    dropped = {str(k): int(v) for k, v in df.loc[mask, "source"].value_counts().items()}
    n0 = len(df); df = df[~mask]                                   # rows with NaN dropped, as upstream ml.py
    s = df["source"].astype(str)
    feats = [c for c in df.columns if c not in ("real", "source")]
    X = df[feats].to_numpy(dtype=np.float64).astype(np.float32)
    bad = ~np.isfinite(X); nf = int(bad.sum()); X[bad] = 0.0      # D46(d): after the cast, set to 0 and count
    real = df["real"].to_numpy(); rows = {}
    for name in pd.unique(s):
        need(name.endswith(".csv"), f"{tag}: source {name} is not a .csv name")
        m = (s == name).to_numpy(); lab = set(real[m].tolist())
        need(lab == ({1} if "EPIC" in name else {0}), f"{tag} {name}: labels {lab}")
        rows[name] = X[m]
    return rows, len(feats), dict(rows_in=n0, nan_rows_dropped=dropped, nonfinite_after_cast=nf)

def halves(A):
    h = len(A) // 2
    return A[:h], A[h + PURGE:]

def blocks(A):
    return [A[i:i + BLOCK] for i in range(0, len(A), BLOCK)]

def eligible(rows):
    sims = sorted(k for k in rows if "EPIC" not in k)
    ok = [k for k in sims if len(rows[k]) >= MIN_ROWS]
    return ok, [k for k in sims if k not in ok]

def build(rows, src, F):
    Rtr, Rte, tonly = [], [], []
    for name in sorted(k for k in rows if "EPIC" in k):
        a, b = halves(rows[name]); Rtr += blocks(a); Rte += blocks(b)
        if not len(b): tonly.append(name)
    a, b = halves(rows[src]); Dtr, Dte = blocks(a), blocks(b)
    need(len(Rte) > 0 and len(Dte) > 0, f"{src}: empty test half")
    n = max(len(Rtr), len(Dtr)); need(n >= 5, f"{src}: {n} training groups, fewer than 5 folds")
    E = np.empty((0, F), np.float32)                              # padding: classify() pairs by index (zip)
    return Rtr + [E] * (n - len(Rtr)), Dtr + [E] * (n - len(Dtr)), Rte, Dte, tonly

def run_one(classify, rows, F, src, seed):
    Rtr, Dtr, Rte, Dte, tonly = build(rows, src, F)
    t = time.time(); r = classify(Rtr, Dtr, Rte, Dte, [], seed)
    o = {k: r[k] for k in ("accuracy", "cv_accuracy", "best_n_estimators", "train_rows", "one_class",
                           "p_real_test_real", "p_real_test_decoy")}
    o.update(p_real=r["m2b"], seconds=time.time() - t, test_blocks_real=len(Rte), test_blocks_sim=len(Dte),
             real_train_only=tonly, classifier_seed=seed, source=src)
    return o

def inventory():
    inv, plan = {}, []
    for g in GENS:
        for sub in SUBS:
            for col in COLS:
                p = fpath(g, sub, col); need(p.exists(), f"missing {p}")
                rows, F, info = load(g, sub, col); ok, sk = eligible(rows)
                real = sorted(k for k in rows if "EPIC" in k)
                tonly = [k for k in real if len(rows[k]) - len(rows[k]) // 2 - PURGE <= 0]
                key = f"s{g}/{sub}/{col}"; nd = info["nan_rows_dropped"]
                dr = sum(v for k, v in nd.items() if "EPIC" in k); ds = sum(v for k, v in nd.items() if "EPIC" not in k)
                inv[key] = dict(sha256=sha(p), features=F, simulated=len(ok), skipped=sk, real_files=real,
                                real_train_only=tonly, **info)
                plan += [(g, sub, col, s) for s in ok]
                for s in ok: build(rows, s, F)  # pre-flight: every planned job has both halves and >= 5 groups
                say(f"   {key:24s} feats {F:4d} | simulated {len(ok):2d}, skipped {len(sk)} | real files {len(real)}, "
                    f"train-only {len(tonly)} | NaN rows dropped: real {dr}, simulated {ds} | "
                    f"non-finite after cast {info['nonfinite_after_cast']}")
    return inv, plan

def jp(P, g, sub, col, s, c):
    return P / f"s{g}" / sub / col / f"{s[:-4]}_c{c}.json"

def noise(s):
    return s[:-4].split("+", 1)[1]

def summary(P, plan):
    unit = {}
    for (g, sub, col, s) in plan:
        R = [json.loads(jp(P, g, sub, col, s, c).read_text()) for c in CSEEDS]
        unit[(g, sub, col, s)] = (float(np.mean([r["p_real"] for r in R])), float(np.mean([r["accuracy"] for r in R])))
    table, acc = {}, {}
    for sub in SUBS:
        for col in COLS:
            keys = [k for k in unit if k[1:3] == (sub, col)]
            if not keys: continue
            for nz in sorted({noise(k[3]) for k in keys}):
                per_g = []
                for g in GENS:
                    v = [unit[k][0] for k in keys if k[0] == g and noise(k[3]) == nz]
                    if v: per_g.append(max(v))                     # D49(r2): max over Panda and Mosaik
                table[f"{sub}/{col}/{nz}"] = dict(p_real_mean=float(np.mean(per_g)), n_generations=len(per_g),
                                                  p_real_sd=float(np.std(per_g, ddof=1)) if len(per_g) > 1 else None,
                                                  per_generation=per_g)
            acc[f"{sub}/{col}"] = float(np.mean([unit[k][1] for k in keys]))
    units = {f"s{k[0]}/{k[1]}/{k[2]}/{k[3]}": dict(p_real_seed_mean=v[0], accuracy_seed_mean=v[1]) for k, v in unit.items()}
    return dict(rule="P(real) per source = mean over 5 classifier seeds; per noise = max over Panda/Mosaik (D49 r2); "
                     "mean (sd) over generations", table=table, accuracy=acc, units=units)

def cell(S, sub, col, nz):
    k = f"{sub}/{col}/{nz}"
    return f"{S['table'][k]['p_real_mean']:11.3f}" if k in S["table"] else f"{'n/a':>11s}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOG[0] = OUT / ("smoke.log" if a.smoke else "run.log")
    me = sha(Path(__file__)); head = git("rev-parse", "--short", "HEAD"); t0 = time.time()
    say(f"D48(c) Arm O on EPIC | repo {head} | {'SMOKE' if a.smoke else 'FULL'} | script {me[:12]}")
    paths = ["spec", "scripts/03d_simprocess.py"] + ([] if a.smoke else ["scripts/epic_armO.py"])
    st = [l for l in git("status", "--porcelain", "--", *paths).splitlines()
          if not (a.smoke and l.strip() == "?? spec/epic_armO_smoke.json")]
    need(not st, f"uncommitted changes: {st}")
    classify, ch = our_classify()
    need(ch == json.loads(SENS_REC.read_text())["classify_sha256"],
         "classify() differs from the version that reproduced the stored HAI/QAPPD results (D66(c))")
    say(f"   gate: classify() sha256 {ch[:12]} = version that reproduced stored HAI/QAPPD results exactly (D66(c))")
    dl = [l.strip() for l in UPML.read_text().splitlines() if "dropna(" in l]
    need(dl, "upstream ml.py has no dropna(); the D68 row rule would not match upstream")
    say(f"   upstream ml.py NaN handling: {dl}")
    if not a.smoke:
        rec = json.loads(SMOKE_REC.read_text())
        need(rec["script_sha256"] == me, "smoke record belongs to a different script version")
        need(re.search(r"^D68[ (]", (ROOT / "spec/decisions.md").read_text(), re.M), "D68 not in spec/decisions.md")
    say("== inputs: 110 frozen upstream feature files (counts only)")
    inv, plan = inventory(); hashes = {k: v["sha256"] for k, v in inv.items()}
    d65t = int(git("log", "-1", "--format=%ct", "daebd39"))
    late = [f"s{g}/{sub}/{col}" for g in GENS for sub in SUBS for col in COLS if fpath(g, sub, col).stat().st_mtime > d65t]
    need(not late, f"feature files modified after the D65 commit: {late}")
    say("   provenance: 110 feature files not modified after the D65 commit daebd39 (sha256 frozen in the smoke record)")
    if not a.smoke: need(hashes == rec["inputs"], "input feature files differ from the smoke record")
    jobs = [(g, sub, col, s, c) for (g, sub, col, s) in plan for c in CSEEDS]
    say(f"   {len(plan)} sources x {len(CSEEDS)} classifier seeds = {len(jobs)} jobs | inputs read in {fmt(time.time() - t0)}")
    if a.smoke:
        T = {}
        for col in COLS:
            rows, F, _ = load(0, "all", col)
            cand = [s for (g, sub, cc, s) in plan if (g, sub, cc) == (0, "all", col)]
            need(cand, f"no eligible source for s0/all/{col}")
            o = run_one(classify, rows, F, cand[0], 0); wjson(OUT / "smoke" / f"{col}.json", o); T[col] = o["seconds"]
            say(f"   smoke {col:15s} {cand[0][:-4]:24s}: {o['train_rows']} training rows, "
                f"{o['test_blocks_real']} + {o['test_blocks_sim']} test blocks, {o['seconds']:.1f} s")
            if col == "C1":
                o2 = run_one(classify, rows, F, cand[0], 0)
                need(all(o[k] == o2[k] for k in ("accuracy", "p_real", "p_real_test_real", "p_real_test_decoy",
                                                  "best_n_estimators")), "C1 repeated with the same seed differs")
                say("   repeat C1 seed 0: identical result (deterministic)")
        proj = {col: T[col] * sum(1 for j in jobs if j[2] == col) for col in COLS}
        SMOKE_REC.write_text(json.dumps(dict(script_sha256=me, classify_sha256=ch, repo_head=head, inputs=hashes,
                             inventory=inv, jobs=len(jobs), seconds_per_fit=T, projection_seconds=sum(proj.values()),
                             values_displayed=False), indent=1))
        for col in COLS: say(f"   projection {col:15s} {fmt(proj[col])}")
        say(f"ARM O SMOKE PASSED in {fmt(time.time() - t0)} -> {SMOKE_REC.relative_to(ROOT)} | "
            f"projected full run ~{fmt(sum(proj.values()))} (one timed fit per column x job count) | no result value shown")
        return
    P = OUT / "pred"; done0 = sum(1 for j in jobs if jp(P, *j).exists()); todo = len(jobs) - done0; n = 0; t1 = time.time()
    say(f"   resume: {done0} of {len(jobs)} done, {todo} to run")
    for g in GENS:
        for sub in SUBS:
            for col in COLS:
                pend = [(s, c) for (gg, ss, cc, s) in plan if (gg, ss, cc) == (g, sub, col) for c in CSEEDS
                        if not jp(P, g, sub, col, s, c).exists()]
                if not pend: continue
                rows, F, _ = load(g, sub, col)
                for s, c in pend:
                    o = run_one(classify, rows, F, s, c); o.update(generation=g, subset=sub, column=col)
                    wjson(jp(P, g, sub, col, s, c), o); n += 1
                    if n % 25 == 0 or n == todo:
                        el = time.time() - t1
                        say(f"   [{done0 + n}/{len(jobs)}] s{g} {sub} {col} {s[:-4]} c{c} | elapsed {fmt(el)} | "
                            f"left ~{fmt(el / n * (todo - n))}")
    S = summary(P, plan); wjson(OUT / "summary.json", S); hs = sha(OUT / "summary.json")
    for sub in SUBS:
        nzs = sorted({k.split("/")[2] for k in S["table"] if k.startswith(sub + "/")})
        say(f"== P(real), {sub}: max over Panda/Mosaik per noise, mean over generations (D49 r2); last column accuracy")
        say("   " + f"{'column':15s}" + "".join(f"{z[:10]:>11s}" for z in nzs) + f"{'accuracy':>11s}")
        for col in COLS:
            if f"{sub}/{col}" in S["accuracy"]:
                say(f"   {col:15s}" + "".join(cell(S, sub, col, z) for z in nzs) + f"{S['accuracy'][sub + '/' + col]:11.3f}")
    say(f"ARM O FULL DONE in {fmt(time.time() - t0)} -> {(OUT / 'summary.json').relative_to(ROOT)} sha256 {hs}")

if __name__ == "__main__":
    main()
