#!/usr/bin/env python
"""D52 exploratory seam diagnostic for the physics-rule detector (not in the Holm family).
Reads stage 3c decoy outputs, detector.json (D38) and stage 3d-2 results (D53); writes results/phase3/seam.
Primary scoring (E6) starts at window step W; a rule with look-back b (flow: lag, rate: k) mixes real warm-up and decoy
values for its first b scored steps. Interior scoring starts at W + b with frozen thresholds. Gates: primary per-rule hits
recomputed here = results/phase3/metrics hits exactly; real normal primary rate = metrics.json; interior hits are a subset
of primary hits. --smoke: C1-Full_s0, B1_s0, B2 only; writes nothing."""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoy import data, detector  # noqa: E402

P3 = ROOT / "results/phase3"
DECOY, MET, OUT = P3 / "decoy", P3 / "metrics", P3 / "seam"
CFG_HASH = "59a11b235108"
DET_SHA = "d50949f5ccc6d0ceab711753b5accdbb9826114ceaba39dc50f4d821b435a67f"
N_OUT = 77
SETS = {"hai": ("normal", "p_real", "p_syn"), "qappd": ("normal", "p_anom", "p_syn")}
SMOKE_OUTS = ("C1-Full_s0", "B1_s0", "B2")
BS = 256                      # same batch size as 03d_metrics.det_wm: residuals computed identically
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def stop(m):
    say(f"STOP: {m}")
    sys.exit(1)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def el():
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def method_of(o):
    m = re.fullmatch(r"(.+)_s(\d)", o)
    return (m.group(1), int(m.group(2))) if m else (o, None)


def per_step(det, cols, XP, Y, W, ncu, mu, sd):
    """Per kept rule: |residual| (N, S_rule) over the primary scoring steps (from W), built exactly as det_wm builds windows."""
    parts = [[] for _ in det["rules"]]
    for i in range(0, len(XP), BS):
        Xd = XP[i:i + BS].copy()
        if Y is not None:
            Xd[:, W:, ncu:] = np.asarray(Y[i:i + BS], np.float64) * sd[ncu:] + mu[ncu:]
        for j, r in enumerate(det["rules"]):
            A, y = detector._io(r, cols, Xd, det["W"])
            parts[j].append(np.abs(y - A @ np.asarray(r["coef"])))
    return [np.concatenate(p) for p in parts]


def score(res, thr, lbs):
    prim = np.stack([e.max(1) for e in res], 1)
    intr = np.stack([e[:, lb:].max(1) for e, lb in zip(res, lbs)], 1)
    hp = (prim > thr) | ~np.isfinite(prim)
    hi = (intr > thr) | ~np.isfinite(intr)
    first = np.full(hp.shape, -1)
    for j, (e, t) in enumerate(zip(res, thr)):
        c = (e > t) | ~np.isfinite(e)
        first[:, j] = np.where(c.any(1), c.argmax(1), -1)
    return hp, hi, first


def summ(hp, hi, first, m, lbs):
    a, b, f = hp[m].any(1), hi[m].any(1), first[m]
    med, within = [], []
    for j, lb in enumerate(lbs):
        v = f[:, j][f[:, j] >= 0]
        med.append(float(np.median(v)) if len(v) else None)
        within.append(float((v < lb).mean()) if len(v) else None)
    return {"windows": int(m.sum()), "primary": float(a.mean()), "interior": float(b.mean()),
            "seam_only": float((a & ~b).mean()), "per_rule_primary": hp[m].mean(0).tolist(),
            "per_rule_interior": hi[m].mean(0).tolist(), "first_step_median": med, "first_within_lookback": within}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    say(f"Stage 3d-2 seam diagnostic (D52) | commit {commit[:7]}{' (dirty)' if dirty else ''} | numpy {np.__version__} | "
        f"{'SMOKE' if args.smoke else 'FULL'}")
    if not args.smoke:
        if dirty:
            stop("full run needs a clean, committed tree")
        if OUT.exists():
            stop(f"{OUT.relative_to(ROOT)} exists; not overwriting")
    if data.cfg_hash() != CFG_HASH:
        stop(f"training config hash {data.cfg_hash()}")
    if sha256(P3 / "detector.json") != DET_SHA:
        stop("detector.json differs from D38")
    fj = json.loads((MET / "files.json").read_text())
    bad = [n for n, h in fj.items() if sha256(MET / n) != h]
    if bad:
        stop(f"stage 3d-2 result files differ from files.json: {bad}")
    mj = json.loads((MET / "metrics.json").read_text())
    if mj["provenance"]["dirty"] or mj["provenance"]["smoke"]:
        stop("metrics.json not from a clean full run")
    dets = json.loads((P3 / "detector.json").read_text())["plants"]
    say(f"gate OK: config, detector.json = D38, {len(fj)} stage 3d-2 files = files.json")
    cfg = data.load_cfg()
    res = {"stage": "3d-2 seam (D52)", "provenance": {"commit": commit, "dirty": dirty, "smoke": args.smoke,
           "script_sha256": sha256(Path(__file__)), "metrics_json_sha256": sha256(MET / "metrics.json")}, "plants": {}}
    ihits = {}
    for pn in ("hai", "qappd"):
        P = data.load_plant(cfg, pn)
        cols = list(P.info["columns"])
        ncu, W = P.nc + P.ne, P.W
        mu, sd = (np.asarray(P.info[k], np.float64) for k in ("mu", "sd"))
        det = dets[pn]["detector"]
        if det["W"] != W:
            stop(f"{pn}: detector W {det['W']} != plant W {W}")
        names = [r["target"] for r in det["rules"]]
        if names != mj["plants"][pn]["rules"]:
            stop(f"{pn}: rules {names} differ from metrics.json")
        thr = np.array([r["thr"] for r in det["rules"]])
        lbs = [int(detector._first(r)) for r in det["rules"]]
        outs = sorted(mj["plants"][pn]["outputs"]["detector"])
        if len(outs) != N_OUT:
            stop(f"{pn}: {len(outs)} outputs in metrics.json")
        todo = list(SMOKE_OUTS) if args.smoke else outs
        idx = {s: json.loads((DECOY / pn / s / "index.json").read_text()) for s in SETS[pn]}
        XP = {s: np.asarray(np.load(DECOY / pn / s / "X.npy", mmap_mode="r"), np.float64) * sd + mu for s in SETS[pn]}
        saved = {s: np.load(MET / f"{pn}_detector_hits_{s}.npz") for s in SETS[pn]}
        for s in SETS[pn]:
            if sorted(saved[s].files) != outs:
                stop(f"{pn} {s}: saved hit outputs differ from metrics.json")

        def groups(s):
            if s == "p_syn":
                st = np.array([e["set"] for e in idx[s]])
                return {"p_syn_in": st == "in", "p_syn_oor": st == "oor"}
            return {s: np.ones(len(idx[s]), bool)}

        say(f"== {pn}: rules " + ", ".join(f"{n} (look-back {b})" for n, b in zip(names, lbs))
            + " | " + ", ".join(f"{s} {len(idx[s])}" for s in SETS[pn]))
        real = {}
        for s in SETS[pn]:
            if s == "p_syn":
                continue
            hp, hi, first = score(per_step(det, cols, XP[s], None, W, ncu, mu, sd), thr, lbs)
            if (hi & ~hp).any():
                stop(f"{pn} real {s}: interior hit without primary hit")
            for g, m in groups(s).items():
                real[g] = summ(hp, hi, first, m, lbs)
        want = mj["plants"][pn]["real_detector"]["normal"]["rate"]
        if abs(real["normal"]["primary"] - want) > 1e-12:
            stop(f"{pn}: real normal primary {real['normal']['primary']} != metrics.json {want}")
        say("   real windows (primary -> interior): "
            + " | ".join(f"{g} {v['primary']:.4f} -> {v['interior']:.4f}" for g, v in real.items()))
        R, ih = {}, {s: {} for s in SETS[pn]}
        for k, o in enumerate(todo, 1):
            R[o] = {}
            for s in SETS[pn]:
                Y = np.load(DECOY / pn / s / f"{o}.npy")
                hp, hi, first = score(per_step(det, cols, XP[s], Y, W, ncu, mu, sd), thr, lbs)
                if not np.array_equal(hp, saved[s][o]):
                    stop(f"{pn} {s} {o}: recomputed primary hits differ from results/phase3/metrics "
                         f"({int((hp != saved[s][o]).sum())} cells)")
                if (hi & ~hp).any():
                    stop(f"{pn} {s} {o}: interior hit without primary hit")
                ih[s][o] = hi
                for g, m in groups(s).items():
                    R[o][g] = summ(hp, hi, first, m, lbs)
                del Y
            say(f"   [{k}/{len(todo)}] {pn} {o:11s} " + " | ".join(
                f"{g} {v['primary']:.3f} -> {v['interior']:.3f}" for g, v in R[o].items()) + f" | {el()}")
        say(f"== {pn}: mean over training seeds, primary -> interior (seam-only)")
        gm = {}
        for o in todo:
            gm.setdefault(method_of(o)[0], []).append(o)
        for m, os_ in gm.items():
            cells = []
            for g in R[os_[0]]:
                mean = lambda key: float(np.mean([R[o][g][key] for o in os_]))
                cells.append(f"{g} {mean('primary'):.3f} -> {mean('interior'):.3f} ({mean('seam_only'):.3f})")
            say(f"   {m:8s} n={len(os_)} | " + " | ".join(cells))
        res["plants"][pn] = {"rules": names, "lookback": lbs, "thresholds": thr.tolist(), "real": real, "outputs": R}
        ihits[pn] = ih
        del XP
    H, Q = res["plants"]["hai"], res["plants"]["qappd"]
    b1 = [H["outputs"][o]["normal"]["interior"] for o in H["outputs"] if method_of(o)[0] == "B1"]
    c1 = [Q["outputs"][o]["normal"]["interior"] for o in Q["outputs"] if method_of(o)[0] == "C1-Full"]
    say(f"D52(e) B1 HAI normal interior {np.mean(b1):.4f} (seam explanation predicts <= 0.02; real interior "
        f"{H['real']['normal']['interior']:.4f})")
    say(f"D52(e) C1-Full QAPPD normal interior {np.mean(c1):.4f} (high = decoy rates inconsistent with its own Pitch/Yaw; "
        f"real interior {Q['real']['normal']['interior']:.4f})")
    if args.smoke:
        say("SMOKE SEAM PASSED (nothing written)")
        return
    part = OUT.with_name("seam.partial")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    for pn, ih in ihits.items():
        for s, d in ih.items():
            np.savez_compressed(part / f"{pn}_interior_hits_{s}.npz", **d)
    res["seconds_total"] = time.time() - T0
    (part / "seam.json").write_text(json.dumps(res, indent=1))
    (part / "files.json").write_text(json.dumps({p.name: sha256(p) for p in sorted(part.iterdir())}, indent=1))
    os.replace(part, OUT)
    say(f"SEAM DONE -> results/phase3/seam | seam.json sha256 {sha256(OUT / 'seam.json')} | "
        f"files.json sha256 {sha256(OUT / 'files.json')} | {el()}")


if __name__ == "__main__":
    main()
