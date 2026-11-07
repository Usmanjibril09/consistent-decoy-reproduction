#!/usr/bin/env python
"""Stage 3d-2 (D51): FIT / M1 (H1), physics-rule detector on decoy windows (H3/H4), DTW, Context-FID.
Reads results/phase3/decoy (stage 3c), detector.json (D38), encoder (D44); writes results/phase3/metrics.
Nothing is trained; the frozen TS2Vec encoder runs forward only. Gates before any held-out metric: hashes of config,
metrics.toml, detector.json, encoder summary; decoy files = stage3c.json sha256; detector on real held-out windows =
stage3b.json; DTW unit tests; FIT in z units = FIT in physical units; cross-time sum formula = direct FIT; encoder
sanity FID = D44. --smoke: outputs C1-Full_s0, B1_s0, B2 on full sets; writes nothing; projects the full run."""
import argparse, csv, hashlib, json, math, os, re, shutil, subprocess, sys, time, tomllib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoy import cfid, data, detector, metrics  # noqa: E402

P3 = ROOT / "results/phase3"
DECOY = P3 / "decoy"
OUT = P3 / "metrics"
CFG_HASH = "59a11b235108"
MET_SHA12 = "a1a995717a64"
DET_SHA = "d50949f5ccc6d0ceab711753b5accdbb9826114ceaba39dc50f4d821b435a67f"
ENC_SHA = "72c0822696b84b14b8018ede870f0988b59bbb72d78ebc147e9fb00eee98ef7c"
N_OUT = 77
SETS = {"hai": ("normal", "p_real", "p_syn"), "qappd": ("normal", "p_anom", "p_syn")}
DTW_SETS = ("normal", "p_real", "p_anom")
SMOKE_OUTS = ("C1-Full_s0", "B1_s0", "B2")
AMB = 1e-4                     # relative threshold margin treated as float32 round-trip ambiguous (gate 4 only)
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


def el(s=None):
    s = int(time.time() - T0 if s is None else s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, np.ndarray):
        return js(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def method_of(o):
    m = re.fullmatch(r"(.+)_s(\d)", o)
    return (m.group(1), int(m.group(2))) if m else (o, None)


# ---------------------------------------------------------------- DTW (D51 d)
def dtw(a, b, w):
    """a, b: (N, T, R); Sakoe-Chiba band |i-j| <= w; local cost squared Euclidean; returns sqrt(total cost), (N,)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    N, T, _ = a.shape
    prev = np.full((T + 1, N), np.inf)
    prev[0] = 0.0
    for i in range(1, T + 1):
        cur = np.full((T + 1, N), np.inf)
        lo, hi = max(1, i - w), min(T, i + w)
        c = np.square(a[:, i - 1, None, :] - b[:, lo - 1:hi, :]).sum(-1).T
        for j in range(lo, hi + 1):
            cur[j] = c[j - lo] + np.minimum(np.minimum(prev[j], cur[j - 1]), prev[j - 1])
        prev = cur
    d = np.sqrt(prev[T])
    d[~np.isfinite(d)] = np.inf
    return d


def dtw_ref(a, b, w):
    T = len(a)
    D = [[math.inf] * (T + 1) for _ in range(T + 1)]
    D[0][0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            if abs(i - j) > w:
                continue
            c = sum((x - y) ** 2 for x, y in zip(a[i - 1], b[j - 1]))
            D[i][j] = c + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    return math.sqrt(D[T][T])


def dtw_tests():
    rs = np.random.RandomState(0)
    for n in range(200):
        T, R = int(rs.randint(2, 9)), int(rs.randint(1, 4))
        w = int(rs.randint(0, T))
        a, b = rs.randn(T, R), rs.randn(T, R)
        d = dtw(a[None], b[None], w)[0]
        if abs(d - dtw_ref(a.tolist(), b.tolist(), w)) > 1e-9:
            stop(f"DTW test {n}: differs from the pure-Python reference")
        if abs(d - dtw(b[None], a[None], w)[0]) > 1e-9:
            stop(f"DTW test {n}: not symmetric")
        if dtw(a[None], a[None], w)[0] != 0.0:
            stop(f"DTW test {n}: d(x, x) != 0")
        if abs(dtw(a[None], b[None], T - 1)[0] - dtw_ref(a.tolist(), b.tolist(), 10 ** 9)) > 1e-9:
            stop(f"DTW test {n}: open band differs from unbanded DTW")
    a, b = np.array([[[0.0], [0.0], [1.0]]]), np.array([[[0.0], [1.0], [1.0]]])
    if dtw(a, b, 0)[0] != 1.0 or dtw(a, b, 1)[0] != 0.0:
        stop("DTW hand example: expected 1.0 (band 0) and 0.0 (band 1)")
    A, B = rs.randn(5, 40, 3), rs.randn(5, 40, 3)
    if np.abs(dtw(A, B, 4) - np.array([dtw(A[i:i + 1], B[i:i + 1], 4)[0] for i in range(5)])).max() > 1e-12:
        stop("DTW batch differs from per-window")


# ---------------------------------------------------------------- FIT
def fit_sums(sse, s, ss, n):
    """Per-tag FIT from SSE, real sum and real sum of squares over n samples (identical rules to metrics.fit_percent)."""
    sse, s, ss = (np.asarray(v, np.float64) for v in (sse, s, ss))
    with np.errstate(all="ignore"):
        den = np.sqrt(np.maximum(ss - s * s / n, 0.0))
        f = 100.0 * (1.0 - np.sqrt(sse) / den)
    f[~np.isfinite(sse)] = -np.inf
    f[den < 1e-12] = np.nan
    ok = ~np.isnan(f)
    return f, (float(f[ok].mean()) if ok.any() else float("nan"))


# ---------------------------------------------------------------- detector
def det_wm(det, cols, XP, Y, W, ncu, mu, sd, bs=256):
    out = []
    for i in range(0, len(XP), bs):
        Xd = XP[i:i + bs].copy()
        if Y is not None:
            Xd[:, W:, ncu:] = np.asarray(Y[i:i + bs], np.float64) * sd[ncu:] + mu[ncu:]
        out.append(detector.window_max(det, cols, Xd))
    return np.concatenate(out)


def hits_of(wm, thr):
    return (wm > thr) | ~np.isfinite(wm)


def rate(h):
    return {"windows": int(len(h)), "rate": float(h.any(1).mean()), "per_rule": h.mean(0).tolist()}


def real_gate(wm, thr, want):
    g = rate(hits_of(wm, thr))
    amb = int((np.abs(wm - thr) <= AMB * thr).any(1).sum())
    n = g["windows"]
    ok = (n == want["windows"] and abs(g["rate"] - want["rate"]) * n <= amb + 1e-9
          and np.all(np.abs(np.array(g["per_rule"]) - np.array(want["per_rule"])) * n <= amb + 1e-9))
    return ok, g, amb


def psyn_summary(h, idx):
    any_ = h.any(1)
    st = np.array([e["set"] for e in idx])
    kd = np.array([e["kind"] for e in idx])
    bs = np.array([int(e["base"]) for e in idx])
    out = {}
    for name in ("in", "oor"):
        m = st == name
        out[name] = {"windows": int(m.sum()), "rate": float(any_[m].mean()), "per_rule": h[m].mean(0).tolist(),
                     "by_kind": {k: float(any_[m & (kd == k)].mean()) for k in sorted(set(kd[m].tolist()))},
                     "per_base": {str(b): float(any_[m & (bs == b)].mean()) for b in sorted(set(bs[m].tolist()))}}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    say(f"Stage 3d-2 metrics | commit {commit[:7]}{' (dirty)' if dirty else ''} | torch {torch.__version__} "
        f"numpy {np.__version__} | threads {torch.get_num_threads()} | {'SMOKE' if args.smoke else 'FULL'}")
    if not args.smoke:
        if dirty:
            stop("full run needs a clean, committed tree")
        if OUT.exists():
            stop(f"{OUT.relative_to(ROOT)} exists; not overwriting")

    # gate 1: fixed inputs
    if data.cfg_hash() != CFG_HASH:
        stop(f"training config hash {data.cfg_hash()} != {CFG_HASH}")
    mtp = ROOT / "configs/metrics.toml"
    if sha256(mtp)[:12] != MET_SHA12:
        stop("configs/metrics.toml differs from D51")
    MT = tomllib.loads(mtp.read_text())
    if sha256(P3 / "detector.json") != DET_SHA:
        stop("results/phase3/detector.json differs from the sha256 logged in D38")
    if sha256(P3 / "encoder/summary.json") != ENC_SHA:
        stop("results/phase3/encoder/summary.json differs from the sha256 logged in D44")
    st3b = json.loads((P3 / "stage3b.json").read_text())
    st3c = json.loads((P3 / "stage3c.json").read_text())
    for n, j in (("stage3b", st3b), ("stage3c", st3c)):
        if j["provenance"].get("dirty") or j["provenance"].get("smoke"):
            stop(f"{n}.json was not written by a clean full run")
    if st3c["stage3b_sha256"] != sha256(P3 / "stage3b.json"):
        stop("stage3b.json differs from the file stage 3c used")
    dets = json.loads((P3 / "detector.json").read_text())["plants"]
    enc = json.loads((P3 / "encoder/summary.json").read_text())["plants"]
    sel = json.loads((P3 / "simprocess/selection.json").read_text())
    att_tag = {r["attack"]: r["response"] for r in csv.DictReader(open(ROOT / "spec/hai2305_command_attacks.csv"))}
    say("gate 1 OK: config 59a11b235108, metrics.toml a1a995717a64 (D51), detector.json = D38, encoder summary = D44, "
        "stage 3b/3c clean, 3c used this 3b")

    # gate 2: DTW unit tests
    t = time.time()
    dtw_tests()
    say(f"gate 2 OK: DTW unit tests (200 random cases vs pure-Python reference, symmetry, identity, open band, "
        f"hand example, batch) {time.time() - t:.1f} s")

    # gate 3: decoy files = stage3c.json
    files = st3c["files"]
    disk = sorted(str(p.relative_to(DECOY)) for p in DECOY.rglob("*") if p.is_file() and not p.name.startswith("."))
    if disk != sorted(files):
        ex, mi = sorted(set(disk) - set(files)), sorted(set(files) - set(disk))
        stop(f"decoy folder differs from stage3c.json: extra {ex[:5]} missing {mi[:5]}")
    if args.smoke:
        check = [k for k in files if k.endswith(("/X.npy", "/index.json", "/real_sum.npy", "/real_sumsq.npy"))
                 or any(k.rsplit("/", 1)[-1].startswith(o + ".") for o in SMOKE_OUTS)]
    else:
        check = sorted(files)
    t = time.time()
    bad = [k for k in check if sha256(DECOY / k) != files[k]["sha256"]]
    if bad:
        stop(f"{len(bad)} decoy files differ from stage3c.json, e.g. {bad[:5]}")
    say(f"gate 3 OK: {len(check)} of {len(files)} decoy files = stage3c.json sha256 ({time.time() - t:.0f} s)")

    cfg = data.load_cfg()
    res = {"stage": "3d-2", "provenance": {
        "commit": commit, "dirty": dirty, "smoke": args.smoke, "config_sha12": CFG_HASH,
        "metrics_toml_sha256": sha256(mtp), "script_sha256": sha256(Path(__file__)),
        "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads()}, "plants": {}}
    store, proj = {}, 0.0
    for pn in ("hai", "qappd"):
        tp = time.time()
        P = data.load_plant(cfg, pn)
        cols = list(P.info["columns"])
        ncu, W, H = P.nc + P.ne, P.W, P.H
        rcols = cols[ncu:]
        mu, sd = (np.asarray(P.info[k], np.float64) for k in ("mu", "sd"))
        det = dets[pn]["detector"]
        thr = np.array([r["thr"] for r in det["rules"]])
        idx = {s: json.loads((DECOY / pn / s / "index.json").read_text()) for s in SETS[pn]}
        Xz = {s: np.load(DECOY / pn / s / "X.npy", mmap_mode="r") for s in SETS[pn]}
        outs = {s: sorted(p.name[:-4] for p in (DECOY / pn / s).glob("*.npy") if p.name != "X.npy") for s in SETS[pn]}
        ref = outs["normal"]
        if len(ref) != N_OUT or any(v != ref for v in outs.values()):
            stop(f"{pn}: output lists differ between sets or != {N_OUT}")
        for s in SETS[pn]:
            if Xz[s].shape != (len(idx[s]), W + H, len(cols)):
                stop(f"{pn} {s}: X.npy shape {Xz[s].shape}")
        cross = []
        if pn == "hai":
            cross = sorted(p.name for p in (DECOY / pn).glob("crosstime_*") if p.is_dir())
            if len(cross) != 6:
                stop(f"hai: {len(cross)} cross-time folders, expected 6")
            for c in cross:
                ci = json.loads((DECOY / pn / c / "index.json").read_text())
                names = sorted(p.name[:-8] for p in (DECOY / pn / c).glob("*.sse.npy"))
                if names != ref:
                    stop(f"hai {c}: SSE outputs differ from the window-set outputs")
                for f in ("real_sum.npy", "real_sumsq.npy"):
                    if np.load(DECOY / pn / c / f, mmap_mode="r").shape != (len(ci), P.nr):
                        stop(f"hai {c}/{f}: shape differs from ({len(ci)}, {P.nr})")
        say(f"== {pn}: {N_OUT} outputs | sets " + ", ".join(f"{s} {len(idx[s])}" for s in SETS[pn])
            + (f" | cross-time {len(cross)} files" if cross else "") + f" | rules {[r['target'] for r in det['rules']]}")
        XP = {s: np.asarray(Xz[s], np.float64) * sd + mu for s in SETS[pn]}
        RY = {s: np.asarray(Xz[s][:, W:, ncu:], np.float64) for s in SETS[pn]}

        # gate 4: detector on real held-out windows = stage3b
        b3 = st3b["plants"][pn]
        chk = [("normal", det_wm(det, cols, XP["normal"], None, W, ncu, mu, sd), b3["normal"]["false_alarm"])]
        attacks = []
        if pn == "hai":
            attacks = list(dict.fromkeys(e["attack"] for e in idx["p_real"]))
            if len(attacks) != 10 or sorted(attacks) != sorted(e["attack"] for e in b3["p_real"]):
                stop(f"hai: P-real attacks {attacks} differ from stage3b.json")
            for a_ in attacks:
                if att_tag.get(a_) not in rcols:
                    stop(f"{a_}: response tag {att_tag.get(a_)} not among response columns")
            wr = det_wm(det, cols, XP["p_real"], None, W, ncu, mu, sd)
            for e in b3["p_real"]:
                m = np.array([i["attack"] == e["attack"] for i in idx["p_real"]])
                chk.append((f"P-real {e['attack']}", wr[m], e["detector"]))
        else:
            chk.append(("P-anom", det_wm(det, cols, XP["p_anom"], None, W, ncu, mu, sd), b3["p_anom"]["detector"]))
        amb_total, real_det = 0, {}
        for name, wm, want in chk:
            ok, g, amb = real_gate(wm, thr, want)
            amb_total += amb
            real_det[name] = g
            if not ok:
                stop(f"{pn} {name}: detector on real windows {g} differs from stage3b.json {want} (ambiguous {amb})")
        say(f"   gate 4 OK: detector on real held-out windows = stage3b.json ({len(chk)} checks; normal false alarm "
            f"{real_det['normal']['rate']:.4f}; windows within {AMB:g} of a threshold: {amb_total})")

        # gates 5-6: FIT units and cross-time formula (C1-Full_s0, normal set)
        Y0 = np.load(DECOY / pn / "normal" / "C1-Full_s0.npy").astype(np.float64)
        fz = metrics.fit_percent(RY["normal"], Y0)[0]
        fp = metrics.fit_percent(RY["normal"] * sd[ncu:] + mu[ncu:], Y0 * sd[ncu:] + mu[ncu:])[0]
        if not np.allclose(fz, fp, rtol=0, atol=1e-6, equal_nan=True):
            stop(f"{pn}: FIT in z units differs from physical units: {np.nanmax(np.abs(fz - fp))}")
        sse = np.square(RY["normal"] - Y0).sum(1)
        fs_ = fit_sums(sse.sum(0), RY["normal"].sum((0, 1)), np.square(RY["normal"]).sum((0, 1)), sse.shape[0] * H)[0]
        if not np.allclose(fs_, fz, rtol=0, atol=1e-4, equal_nan=True):
            stop(f"{pn}: cross-time sum formula differs from direct FIT: {np.nanmax(np.abs(fs_ - fz))}")
        fin = np.isfinite(fz) & np.isfinite(fp) & np.isfinite(fs_)
        say(f"   gates 5-6 OK: FIT z = physical (max |d| {np.abs(fz - fp)[fin].max():.1e}); sum formula = direct "
            f"(max |d| {np.abs(fs_ - fz)[fin].max():.1e}) on C1-Full_s0 normal, 18/11 tags")
        del Y0

        # gate 7: encoder
        ep = P3 / "encoder" / pn / "encoder.pt"
        if sha256(ep) != enc[pn]["weights_sha256"]:
            stop(f"{pn}: encoder weights differ from D44")
        E = cfid.load_encoder(ep, P.nr)
        t = time.time()
        eva = cfid.embed(E, np.ascontiguousarray(P.va[:, W:, ncu:]))
        fv, want = cfid.fid(eva[0::2], eva[1::2]), enc[pn]["sanity_fid"]["val_even_vs_odd"]
        if abs(fv / want - 1) > 1e-4:
            stop(f"{pn}: encoder sanity FID {fv} differs from D44 {want}")
        er = cfid.embed(E, np.ascontiguousarray(Xz["normal"][:, W:, ncu:]))
        hv = sel[pn]["halves"]
        ia = sorted(i for f in hv for i in hv[f]["train"])
        ib = sorted(i for f in hv for i in hv[f]["test"])
        cref = cfid.fid(er[ia], er[ib])
        say(f"   gate 7 OK: encoder = D44, sanity FID {fv:.4f} (D44 {want:.4f}); real first vs second half "
            f"Context-FID {cref:.4f} ({len(ia)} vs {len(ib)} windows) {time.time() - t:.0f} s")
        del eva
        fixed = time.time() - tp

        # metrics
        w = int(round(float(MT["dtw"]["band_frac"]) * H))
        todo = list(SMOKE_OUTS) if args.smoke else ref
        if any(o not in ref for o in todo):
            stop(f"{pn}: smoke outputs missing")
        R = {"fit": {}, "detector": {}, "dtw": {}, "cfid": {}}
        hs = {s: {} for s in SETS[pn]}
        ds = {s: {} for s in SETS[pn] if s in DTW_SETS}
        tsec = {"fit": 0.0, "detector": 0.0, "dtw": 0.0, "cfid": 0.0}
        tl = time.time()
        for k, o in enumerate(todo, 1):
            Y = {s: np.load(DECOY / pn / s / f"{o}.npy") for s in SETS[pn]}
            t = time.time()
            F = {}
            pt, F["normal"] = metrics.fit_percent(RY["normal"], Y["normal"])
            F["normal_per_tag"] = pt.tolist()
            if pn == "hai":
                F["p_real_m1"], F["p_real_alltag"] = {}, {}
                for a_ in attacks:
                    ii = [i for i, e in enumerate(idx["p_real"]) if e["attack"] == a_]
                    r = rcols.index(att_tag[a_])
                    F["p_real_m1"][a_] = float(metrics.fit_percent(RY["p_real"][ii][..., r:r + 1],
                                                                   Y["p_real"][ii][..., r:r + 1])[0][0])
                    F["p_real_alltag"][a_] = metrics.fit_percent(RY["p_real"][ii], Y["p_real"][ii])[1]
                F["p_real_pooled"] = metrics.fit_percent(RY["p_real"], Y["p_real"])[1]
                ct, acc = {}, [0.0, 0.0, 0.0, 0]
                for c in cross:
                    d = DECOY / pn / c
                    e_, s1, s2 = (np.load(d / f) for f in (f"{o}.sse.npy", "real_sum.npy", "real_sumsq.npy"))
                    ct[c.replace("crosstime_", "")] = fit_sums(e_.sum(0), s1.sum(0), s2.sum(0), len(e_) * H)[1]
                    acc = [acc[0] + e_.sum(0), acc[1] + s1.sum(0), acc[2] + s2.sum(0), acc[3] + len(e_) * H]
                ct["pooled"] = fit_sums(*acc)[1]
                F["crosstime"] = ct
            else:
                F["p_anom"] = metrics.fit_percent(RY["p_anom"], Y["p_anom"])[1]
            R["fit"][o] = F
            tsec["fit"] += time.time() - t
            t = time.time()
            D = {}
            for s in SETS[pn]:
                h = hits_of(det_wm(det, cols, XP[s], Y[s], W, ncu, mu, sd), thr)
                hs[s][o] = h
                D[s] = psyn_summary(h, idx[s]) if s == "p_syn" else rate(h)
            R["detector"][o] = D
            tsec["detector"] += time.time() - t
            t = time.time()
            Dt = {}
            for s in ds:
                dd = dtw(RY[s], Y[s], w)
                ds[s][o] = dd
                Dt[s] = float(np.median(dd))
            R["dtw"][o] = Dt
            tsec["dtw"] += time.time() - t
            t = time.time()
            Yn = np.ascontiguousarray(Y["normal"], dtype=np.float32)
            R["cfid"][o] = cfid.fid(er, cfid.embed(E, Yn)) if np.isfinite(Yn).all() else float("inf")
            tsec["cfid"] += time.time() - t
            per = (time.time() - tl) / k
            m1 = f" | M1 mean {np.mean(list(F['p_real_m1'].values())):9.2f}" if pn == "hai" else f" | P-anom {F['p_anom']:8.2f}"
            say(f"   [{k}/{len(todo)}] {pn} {o:11s} normal FIT {F['normal']:8.2f}{m1} | det oor "
                f"{D['p_syn']['oor']['rate']:.3f} in {D['p_syn']['in']['rate']:.3f} | DTW {Dt['normal']:8.2f} | "
                f"CFID {R['cfid'][o]:9.3f} | {el()} | left ~{el(per * (len(todo) - k))}")
            del Y
        groups = {}
        for o in todo:
            groups.setdefault(method_of(o)[0], []).append(o)
        say(f"== {pn}: mean over training seeds")
        for m, os_ in groups.items():
            g = lambda f: float(np.mean([f(o) for o in os_]))
            line = f"   {m:8s} n={len(os_)} | normal FIT {g(lambda o: R['fit'][o]['normal']):8.2f}"
            if pn == "hai":
                m1 = float(np.mean([np.mean([R["fit"][o]["p_real_m1"][a_] for o in os_]) for a_ in attacks]))
                line += f" | M1 {m1:9.2f} | cross-time {g(lambda o: R['fit'][o]['crosstime']['pooled']):9.2f}"
            else:
                line += f" | P-anom {g(lambda o: R['fit'][o]['p_anom']):8.2f}"
            line += (f" | det oor {g(lambda o: R['detector'][o]['p_syn']['oor']['rate']):.3f}"
                     f" in {g(lambda o: R['detector'][o]['p_syn']['in']['rate']):.3f}"
                     f" normal {g(lambda o: R['detector'][o]['normal']['rate']):.3f}"
                     f" | DTW {g(lambda o: R['dtw'][o]['normal']):8.2f} | CFID {g(lambda o: R['cfid'][o]):9.3f}")
            say(line)
        res["plants"][pn] = {"rules": [r["target"] for r in det["rules"]], "dtw_band": w, "attacks": attacks,
                             "out_of_coverage": sorted({e["attack"] for e in idx.get("p_real", []) if e["out_of_coverage"]}),
                             "real_detector": real_det, "cfid_reference_real_halves": cref, "outputs": R,
                             "seconds": dict(tsec, fixed=fixed)}
        store[pn] = (hs, ds)
        per_out = sum(tsec.values()) / len(todo)
        proj += fixed + per_out * N_OUT
        say(f"   {pn}: {per_out:.1f} s per output (" + ", ".join(f"{k_} {v / len(todo):.1f}" for k_, v in tsec.items())
            + f"), fixed {fixed:.0f} s")
        del XP, RY, E
    if args.smoke:
        say(f"projected full run: about {proj / 60:.0f} min (plus about 1 min hashing all decoy files)")
        say("SMOKE 3d-2 PASSED (nothing written)")
        return
    part = OUT.with_name("metrics.partial")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    for pn, (hs, ds) in store.items():
        for s, d in hs.items():
            np.savez_compressed(part / f"{pn}_detector_hits_{s}.npz", **d)
        for s, d in ds.items():
            np.savez_compressed(part / f"{pn}_dtw_{s}.npz", **d)
    res["seconds_total"] = time.time() - T0
    (part / "metrics.json").write_text(json.dumps(js(res), indent=1))
    res_files = {p.name: sha256(p) for p in sorted(part.iterdir())}
    (part / "files.json").write_text(json.dumps(res_files, indent=1))
    os.replace(part, OUT)
    say(f"STAGE 3d-2 DONE -> results/phase3/metrics | metrics.json sha256 {sha256(OUT / 'metrics.json')} | "
        f"files.json sha256 {sha256(OUT / 'files.json')} | {el()}")


if __name__ == "__main__":
    main()
