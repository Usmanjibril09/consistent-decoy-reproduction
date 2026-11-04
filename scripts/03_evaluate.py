#!/usr/bin/env python
"""Phase 3 (PROTOCOL v1.5): one-time held-out evaluation. One script, run by stage.

Stage 3a - gate before any held-out value is read.
  (1) Provenance: protocol-v1.5 is an ancestor of HEAD; Phase 2 inference code and config are identical to tag
      phase2-done; config hash equals the Phase 2 record; processed train/val hashes equal those used on Kaggle.
  (2) Held-out manifest: SHA-256, split label, column/role identity of every held-out file. Only the 'split',
      'columns' and 'roles' arrays are read; X and labels are not.
  (3) D36 reproduction (CPU) of recorded validation FIT%: deterministic methods (B1 B2 B3 B8 B9 C1 A3, C1 layer
      variants, A3 layered) within 0.01; stochastic B4-B7 within max(2.0, 3 x sd of the 5 recorded seeds, ddof=1).
      B8-L (B8 + C2/C3, D33) has no Phase 2 record: its validation FIT is computed and recorded.
  --smoke: seed 0 only, writes nothing. Full run: all seeds, writes results/phase3/stage3a.json.
New Phase 3 code lives in this script or new modules; Phase 2 modules stay frozen.
"""
import argparse, json, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoy import baselines, constraints, data, detector, metrics, models  # noqa: E402

PHASE2 = ROOT / "results/phase2_kaggle/extracted/phase2"
LAYERS = ROOT / "results/phase2_d32"
OUT = ROOT / "results/phase3"
PLANTS = ("hai", "qappd")
STOCHASTIC = ("B4", "B5", "B6", "B7")
NEURAL = ("B4", "B5", "B6", "B7", "B8", "B9", "C1", "A3")
VARIANTS = ("Full", "A1", "A2", "A4", "A5")
DET_TOL = 0.01
THIS = "scripts/03_evaluate.py"
MINE = {THIS, "src/decoy/detector.py"}
FROZEN = ["src/decoy/data.py", "src/decoy/models.py", "src/decoy/baselines.py", "src/decoy/constraints.py",
          "src/decoy/metrics.py", "src/decoy/attacker.py", "scripts/02_train.py", "configs/experiment.toml"]
DEV = torch.device("cpu")


def say(*a):
    print(*a, flush=True)


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def sh(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def rjson(p):
    if not p.exists():
        stop(f"missing {p.relative_to(ROOT)}")
    return json.loads(p.read_text())


def _default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def jsonable(x):
    return json.loads(json.dumps(x, default=_default))


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(p)


def same(a, b, path):
    """First path where two JSON trees differ (numbers compared with rtol 1e-9), else None."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return path
        for k in a:
            r = same(a[k], b[k], f"{path}.{k}")
            if r:
                return r
        return None
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return path
        for i, (x, y) in enumerate(zip(a, b)):
            r = same(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    if any(isinstance(v, (bool, str)) or v is None for v in (a, b)):
        return None if a == b else path
    return None if np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12, equal_nan=True) else path


# ---------------------------------------------------------------- provenance
def provenance(smoke):
    st = [l for l in sh("git", "status", "--porcelain").stdout.splitlines() if l]
    other = [l for l in st if l[3:] not in MINE]
    if other:
        stop(f"uncommitted changes other than {THIS}: {other}")
    if st and not smoke:
        stop(f"full run needs the smoke-tested {THIS} committed; git status: {st}")
    if sh("git", "merge-base", "--is-ancestor", "protocol-v1.5", "HEAD").returncode != 0:
        stop("tag protocol-v1.5 is not an ancestor of HEAD")
    changed = sh("git", "diff", "--name-only", "phase2-done", "--", *FROZEN).stdout.split()
    if changed:
        stop(f"Phase 2 inference code/config changed since phase2-done: {changed}")
    rec = rjson(LAYERS / "relayer_summary.json")["config_sha12"]
    if data.cfg_hash() != rec:
        stop(f"config hash {data.cfg_hash()} != Phase 2 record {rec}")
    return {"commit": sh("git", "rev-parse", "HEAD").stdout.strip(), "dirty": bool(st), "smoke": smoke,
            "config_sha12": rec, "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": str(torch.__version__), "threads": torch.get_num_threads(),
            "device": "cpu"}


# ---------------------------------------------------------------- held-out manifest (no values read)
def manifest(cfg):
    proc = ROOT / cfg["paths"]["processed"]
    groups = {"hai": ("hai", list(cfg["hai"]["test"])),
              "qappd": ("qappd", [f"train_{i}" for i in cfg["qappd"]["heldout_normal"]]
                        + [f"test_{i}" for i in cfg["qappd"]["test"]]),
              "hai2204": ("hai", list(cfg["hai_crosstime"]["files"]))}
    out, bad = {}, []
    for folder, (plant, files) in groups.items():
        T = cfg["train"][plant]
        with np.load(proc / plant / f"{T['files_train'][0]}.npz", allow_pickle=False) as z:
            rc, rr = z["columns"].astype(str), z["roles"].astype(str)
        dev_files = set(T["files_train"]) | set(T["files_val"])
        for f in files:
            if folder == plant and f in dev_files:
                stop(f"{folder}/{f} is listed as held-out but used for training/validation")
            p = proc / folder / f"{f}.npz"
            if not p.exists():
                stop(f"missing {p.relative_to(ROOT)}")
            with np.load(p, allow_pickle=False) as z:
                r = {"sha256": data.file_sha(p), "split": str(z["split"]),
                     "columns_match": bool(np.array_equal(z["columns"].astype(str), rc)),
                     "roles_match": bool(np.array_equal(z["roles"].astype(str), rr))}
            out[f"{folder}/{f}"] = r
            if not (r["columns_match"] and r["roles_match"]):
                bad.append(f"{folder}/{f}")
            say(f"  {folder:7s} {f:10s} split {r['split']:9s} columns {'=' if r['columns_match'] else 'DIFF'}"
                f"  roles {'=' if r['roles_match'] else 'DIFF'}  sha {r['sha256'][:12]}")
    if bad:
        stop(f"held-out files whose columns/roles differ from training: {bad}")
    return out


# ---------------------------------------------------------------- Phase 2 records
def selection(pn, method):
    base = "C1" if method == "A3" else method          # A3 reuses C1's configuration (D28)
    sel = rjson(PHASE2 / pn / base / "selection.json")
    if "best" not in sel or "hp" not in sel:
        stop(f"{pn}/{base}/selection.json has no best/hp (keys {sorted(sel)})")
    return int(sel["best"]), sel["hp"]


def run_dir(pn, method, k, seed):
    return PHASE2 / pn / method / f"cfg{k:02d}_seed{seed}"


def done(pn, method, k, hp, seed):
    r = rjson(run_dir(pn, method, k, seed) / "done.json")
    for key, want in (("method", method), ("plant", pn), ("cfg", k), ("seed", seed), ("hp", hp)):
        if r.get(key) != want:
            stop(f"{pn}/{method}/cfg{k:02d}_seed{seed}/done.json: {key}={r.get(key)!r}, expected {want!r}")
    return r


def recorded_b1(pn):
    """B1 has no done.json. Its seed FITs are stored in the Phase 2 run summaries at
    plants.<plant>.results.B1.seed_fit. Only non-smoke runs with the Phase 2 config hash count; all must agree."""
    want = rjson(LAYERS / "relayer_summary.json")["config_sha12"]
    comp = rjson(PHASE2 / f"COMPLETE_{pn}.json")
    if comp.get("smoke") is not False or "B1" not in comp.get("methods", []):
        stop(f"COMPLETE_{pn}.json is not a completed non-smoke run including B1")
    hits = []
    for f in sorted(PHASE2.glob(f"run_{pn}_*.json")):
        j = json.loads(f.read_text())
        pv = j.get("provenance", {})
        if pv.get("smoke") or pv.get("config_sha12") != want:
            say(f"  {pn:5s} B1: skipped {f.name} (smoke={pv.get('smoke')}, config={pv.get('config_sha12')})")
            continue
        sf = j.get("plants", {}).get(pn, {}).get("results", {}).get("B1", {}).get("seed_fit")
        if isinstance(sf, list):
            hits.append((f, sf))
    if not hits:
        stop(f"no qualifying run_{pn}_*.json with plants.{pn}.results.B1.seed_fit")
    if len({json.dumps(h[1]) for h in hits}) > 1:
        stop(f"conflicting B1 records for {pn}: {[(f.name, v) for f, v in hits]}")
    say(f"  {pn:5s} B1 record: plants.{pn}.results.B1.seed_fit in {', '.join(f.name for f, _ in hits)}"
        f" (mean {np.mean(hits[0][1]):.2f})")
    return [float(v) for v in hits[0][1]]


# ---------------------------------------------------------------- frozen models
def load_model(pn, method, k, hp, seed, P):
    m = models.build(method, P, hp).to(DEV)
    sd = torch.load(run_dir(pn, method, k, seed) / "model.pt", map_location=DEV, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        stop(f"{pn}/{method} seed {seed}: weights missing from model.pt: {missing}")
    if method == "B4":
        m.fit_noise(P.tr, DEV)      # Phase 2 train_one: fit_noise after training, before generate
    if method == "B7":
        m.fit_scale(P.tr)           # restores the scale if it is not stored in the state dict
    late = [key for key in unexpected if key not in m.state_dict()]
    if late:
        stop(f"{pn}/{method} seed {seed}: model.pt keys the model never creates: {late}")
    m.load_state_dict(sd, strict=True)  # saved values take precedence wherever they exist
    return m


def output(method, m, P, seed):
    if method in STOCHASTIC:        # Phase 2 scoring path (generator seed + 30000)
        return models.generate(m, P.va, DEV, seed)[..., P.nc + P.ne:]
    return models.predict(m, P.va, DEV)


# ---------------------------------------------------------------- D36 reproduction
def reproduce(cfg, pn, seeds_all, run_seeds):
    P = data.load_plant(cfg, pn)
    ncu = P.nc + P.ne
    y = P.va[:, P.W:, ncu:]
    k1, hp1 = selection(pn, "C1")
    kfiles = rjson(run_dir(pn, "C1", k1, seeds_all[0]) / "identity.json").get("files") or {}
    for f, i in P.info["files"].items():
        if kfiles.get(f, {}).get("sha256") != i["sha256"]:
            stop(f"{pn}/{f}: processed file differs from the one used on Kaggle")
    rows, secs, preds = [], {}, {"C1": {}, "A3": {}, "B8": {}}

    def check(method, seed, yh, rec, tol, new=False):
        if yh.shape != y.shape:
            stop(f"{pn} {method} seed {seed}: output shape {yh.shape} != {y.shape}")
        got = metrics.fit_percent(y, yh)[1]
        ok = bool(np.isfinite(got)) and (new or abs(got - rec) <= tol)
        rows.append({"plant": pn, "method": method, "seed": seed, "val_fit": got, "recorded": rec,
                     "abs_diff": None if new else abs(got - rec), "tol": tol, "pass": ok, "new": new})
        s = "-" if seed is None else seed
        if new:
            say(f"  {pn:5s} {method:8s} seed {s}  val FIT {got:9.4f}  (new, no Phase 2 record)  {'OK' if ok else 'FAIL'}")
        else:
            say(f"  {pn:5s} {method:8s} seed {s}  reloaded {got:9.4f} | recorded {rec:9.4f} | "
                f"|d| {abs(got - rec):7.4f} <= {tol:.4f}  {'OK' if ok else 'FAIL'}")

    t0 = time.time()
    b1 = recorded_b1(pn)
    if len(b1) != len(seeds_all):
        stop(f"{pn}: B1 record has {len(b1)} seeds, expected {len(seeds_all)}")
    for s in run_seeds:
        check("B1", s, baselines.b1_replay(P.tr, P.va, P.nc, P.ne, P.W, s), b1[seeds_all.index(s)], DET_TOL)
    secs["B1"] = time.time() - t0

    t0 = time.time()
    k, hp = selection(pn, "B2")
    d2 = PHASE2 / pn / "B2" / f"cfg{k:02d}"
    r2 = rjson(d2 / "done.json")
    if r2.get("hp") != hp or r2.get("cfg") != k:
        stop(f"{pn}/B2: done.json does not match selection.json")
    th = np.load(d2 / "theta.npy", allow_pickle=False)
    check("B2", None, baselines.arx_sim(th, P.va, P.nc, P.ne, P.W, hp["na"], hp["nb"]), r2["val_fit"], DET_TOL)
    secs["B2"] = time.time() - t0

    t0 = time.time()
    grey = rjson(PHASE2 / pn / "B3" / "parameters.json")
    r3 = rjson(PHASE2 / pn / "B3" / "done.json")
    if not (isinstance(r3.get("seed_fit"), list) and len(r3["seed_fit"]) == 1):
        stop(f"{pn}/B3/done.json has no single seed_fit (keys {sorted(r3)})")
    check("B3", None, baselines.grey_sim(grey, P.va, P.W, ncu), r3["seed_fit"][0], DET_TOL)
    secs["B3"] = time.time() - t0

    for method in NEURAL:
        t0 = time.time()
        k, hp = selection(pn, method)
        recs = [done(pn, method, k, hp, s)["val_fit"] for s in seeds_all]
        tol = max(2.0, 3.0 * float(np.std(recs, ddof=1))) if method in STOCHASTIC else DET_TOL
        for s in run_seeds:
            m = load_model(pn, method, k, hp, s, P)
            yh = output(method, m, P, s)
            check(method, s, yh, recs[seeds_all.index(s)], tol)
            if method in preds:
                preds[method][s] = yh
            del m
        secs[method] = time.time() - t0

    t0 = time.time()
    c = constraints.fit(P, cfg["train"]["c2"])
    where = same(jsonable(c), rjson(LAYERS / pn / "layers" / "parameters.json"), "c2")
    if where:
        stop(f"{pn}: refitted C2/C3 parameters differ from results/phase2_d32 at {where}")
    if np.any(constraints.coverage(c, P.tr) > 0):
        stop(f"{pn}: C3 active on training windows")
    lv = rjson(LAYERS / pn / "layers" / "validation.json")
    for s in run_seeds:
        for v in VARIANTS:
            check(f"C1-{v}", s, constraints.apply(c, grey, P.va, preds["C1"][s], v),
                  lv["C1"][str(s)][v]["val_fit"], DET_TOL)
        check("A3-L", s, constraints.apply(c, grey, P.va, preds["A3"][s], "A3"),
              lv["A3"][str(s)]["A3"]["val_fit"], DET_TOL)
        check("B8-L", s, constraints.apply(c, grey, P.va, preds["B8"][s], "Full"), None, None, new=True)
    secs["layers"] = time.time() - t0

    return {"rows": rows, "seconds": secs, "best": {m: selection(pn, m)[0] for m in ("B2",) + NEURAL},
            "train_val_sha256": {f: i["sha256"] for f, i in P.info["files"].items()},
            "val_windows_c3_active": float((constraints.coverage(c, P.va) > 0).mean())}


# ---------------------------------------------------------------- physics-rule detector gate (D35)
D35_CHECK = {  # D35 text (val R2, k, gains, false alarms, kept) + D35 check-run log (lags, train R2, thresholds)
    "hai": {"lag": {"P1_FT01": 0, "P1_FT01Z": 1, "P1_FT02": 0, "P1_FT02Z": 1, "P1_FT03": 30, "P1_FT03Z": 20},
            "tr_r2": {"P1_FT01": .920, "P1_FT01Z": .925, "P1_FT02": .964, "P1_FT02Z": .972, "P1_FT03": .625, "P1_FT03Z": .619},
            "va_r2": {"P1_FT01": .887, "P1_FT01Z": .893, "P1_FT02": .949, "P1_FT02Z": .966, "P1_FT03": .623, "P1_FT03Z": .599},
            "thr": {"P1_FT01": 69.48, "P1_FT01Z": 269.3, "P1_FT02": 901.0, "P1_FT02Z": 827.1, "P1_FT03": 46.8, "P1_FT03Z": 93.19},
            "va_fa": {"P1_FT01": 0.0, "P1_FT01Z": 0.0, "P1_FT02": 0.0, "P1_FT02Z": 0.0, "P1_FT03": 0.0},
            "kept": ["P1_FT01Z", "P1_FT02Z", "P1_FT03"]},
    "qappd": {"k": {"PitchDot": 10, "YawDot": 10}, "gain": {"PitchDot": .998, "YawDot": 1.000},
              "va_r2": {"PitchDot": .969, "YawDot": .994}, "va_fa": {"PitchDot": .005, "YawDot": .000},
              "kept": ["PitchDot", "YawDot"]}}
DET_TOL = {"r2": 0.002, "gain": 0.002, "fa": 0.002, "thr_rel": 0.01}


def detector_gate(cfg, prov, smoke):
    out = OUT / "detector.json"
    if not smoke and out.exists():
        stop(f"{out.relative_to(ROOT)} exists; detector gate already passed")
    res, fails = {"stage": "det", "provenance": prov, "check": D35_CHECK, "tolerance": DET_TOL, "plants": {}}, []
    for pn in PLANTS:
        P = data.load_plant(cfg, pn)
        det = detector.fit(P, int(cfg[pn]["fs_hz"]))
        ck = D35_CHECK[pn]
        say(f"== {pn}: windows train {len(P.tr)} val {len(P.va)} (training/validation only) ==")
        for r in det["candidates"]:
            t, key = r["target"], ("lag" if r["kind"] == "flow" else "k")
            bad = []

            def cmp(name, got, want, tol, rel=False):
                if want is None:
                    return
                d = abs(got / want - 1) if rel else abs(got - want)
                if d > tol:
                    bad.append(f"{name} {got:.4g} vs D35 {want:.4g}")
            if r[key] != ck[key][t]:
                bad.append(f"{key} {r[key]} vs D35 {ck[key][t]}")
            cmp("train R2", r["tr_r2"], ck.get("tr_r2", {}).get(t), DET_TOL["r2"])
            cmp("val R2", r["va_r2"], ck["va_r2"][t], DET_TOL["r2"])
            cmp("threshold", r["thr"], ck.get("thr", {}).get(t), DET_TOL["thr_rel"], rel=True)
            cmp("gain", r["coef"][0], ck.get("gain", {}).get(t), DET_TOL["gain"])
            cmp("val false alarm", r["va_fa"], ck["va_fa"].get(t), DET_TOL["fa"])
            g = f"  gain {r['coef'][0]:.3f}" if r["kind"] == "rate" else ""
            say(f"  {t:9s} {key} {r[key]:>2}{g}  tr R2 {r['tr_r2']:.3f}  va R2 {r['va_r2']:.3f}  thr {r['thr']:.4g}"
                f"  va FA {r['va_fa']:.4f}  {'KEEP' if r['kept'] else 'drop'}  {'OK' if not bad else 'MISMATCH: ' + '; '.join(bad)}")
            fails += [f"{pn} {t}: {b}" for b in bad]
        kept = [r["target"] for r in det["rules"]]
        if kept != ck["kept"]:
            fails.append(f"{pn}: kept rules {kept} vs D35 {ck['kept']}")
        res["plants"][pn] = {"detector": det, "train_val_sha256": {f: i["sha256"] for f, i in P.info["files"].items()}}
    gate = "FAIL" if fails else "PASS"
    res["gate"] = gate
    if not smoke:
        wjson(out if gate == "PASS" else OUT / f"detector_FAILED_{time.strftime('%Y%m%d-%H%M%S')}.json", res)
    if fails:
        say("== MISMATCHES ==")
        for f in fails:
            say("  " + f)
        stop(f"detector gate FAILED ({len(fails)}); no held-out value has been read")
    say("SMOKE DETECTOR GATE PASSED" if smoke else "DETECTOR GATE PASSED -> results/phase3/detector.json")


def main():
    ap = argparse.ArgumentParser(description="Phase 3 held-out evaluation")
    ap.add_argument("--stage", required=True, choices=["3a", "det"])
    ap.add_argument("--smoke", action="store_true", help="seed 0 only; writes nothing")
    a = ap.parse_args()
    t_all = time.time()
    prov = provenance(a.smoke)
    say(f"Phase 3 stage {a.stage} | commit {prov['commit'][:7]}{' (dirty: ' + THIS + ')' if prov['dirty'] else ''}"
        f" | config {prov['config_sha12']} | torch {prov['torch']} numpy {prov['numpy']}"
        f" | {prov['threads']} threads | {'SMOKE (seed 0)' if a.smoke else 'FULL'}")
    cfg = data.load_cfg()
    if a.stage == "det":
        return detector_gate(cfg, prov, a.smoke)
    T = cfg["train"]
    seeds_all = [int(T["seed_select"])] + [int(s) for s in T["seeds_final"]]
    run_seeds = seeds_all[:1] if a.smoke else seeds_all
    out = OUT / "stage3a.json"
    if not a.smoke and out.exists():
        stop(f"{out.relative_to(ROOT)} exists; stage 3a already passed")
    say("== held-out manifest (split/columns/roles only; no values read) ==")
    man = manifest(cfg)
    say("== D36 reproduction on validation (CPU) ==")
    plants = {pn: reproduce(cfg, pn, seeds_all, run_seeds) for pn in PLANTS}
    rows = [r for p in plants.values() for r in p["rows"]]
    fails = [r for r in rows if not r["pass"]]
    say("== timing (s) ==")
    for pn, p in plants.items():
        say(f"  {pn:5s} " + "  ".join(f"{k} {v:.0f}" for k, v in p["seconds"].items()))
    if a.smoke:
        n = len(seeds_all)
        proj = sum(v * (1 if k in ("B2", "B3") else n) for p in plants.values() for k, v in p["seconds"].items())
        say(f"  projected full run: about {proj / 60:.0f} min")
    gate = "FAIL" if fails else "PASS"
    if not a.smoke:
        res = {"stage": "3a", "gate": gate, "provenance": prov,
               "rule": "D36: deterministic |d| <= 0.01; B4-B7 |d| <= max(2.0, 3*sd(5 recorded seeds, ddof=1))",
               "heldout_manifest": man, "plants": plants, "seconds_total": time.time() - t_all}
        wjson(out if gate == "PASS" else OUT / f"stage3a_FAILED_{time.strftime('%Y%m%d-%H%M%S')}.json", res)
    if fails:
        say("== FAILURES ==")
        for r in fails:
            say(f"  {r['plant']} {r['method']} seed {r['seed']}: {r['val_fit']} vs {r['recorded']} (tol {r['tol']})")
        stop(f"D36 reproduction gate FAILED ({len(fails)} of {len(rows)} checks); no held-out value has been read")
    say(f"{'SMOKE 3a PASSED' if a.smoke else 'STAGE 3a PASSED -> results/phase3/stage3a.json'}: "
        f"{len(rows)} checks, {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
