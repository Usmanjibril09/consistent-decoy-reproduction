#!/usr/bin/env python
"""Stage 3d-3 (D51, D55): post-hoc RNN and MiniRocket K-test classifiers (M2 = H2), P-syn flag rates, TSTR.
Reads results/phase3/decoy (stage 3c) and the stage 3d-1 halves (selection.json); writes results/phase3/classifiers.
Gates: config and metrics.toml hashes, toml settings = implemented settings, selection.json = D47 manifest, decoy files =
stage3c.json, halves sizes. Jobs run in 4 processes (1 thread each), one result file per job, resumable under the same
commit and script. --smoke: C1-Full_s0, B1_s0, B2 with classifier seed 0 on both plants, an RNN determinism rerun; writes
nothing; projects the full run."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import argparse, hashlib, json, re, shutil, subprocess, sys, time, tomllib
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoy import data  # noqa: E402

P3 = ROOT / "results/phase3"
DECOY, OUT, PART = P3 / "decoy", P3 / "classifiers", P3 / "classifiers.partial"
SELP = P3 / "simprocess/selection.json"
CFG_HASH, MET_SHA12 = "59a11b235108", "a1a995717a64"
N_OUT, SEEDS, PROCS = 77, list(range(5)), 4
SMOKE_OUTS = ("C1-Full_s0", "B1_s0", "B2")
HALF_N = {"hai": (610, 608), "qappd": (602, 602)}
ITER_RNN, ITER_TSTR, BATCH, NK = 2000, 5000, 128, 10000
PRIO = {"tstr": 0, "trtr": 0, "rnn": 1, "mr": 2}
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


def method_of(o):
    m = re.fullmatch(r"(.+)_s(\d)", o)
    return (m.group(1), int(m.group(2))) if m else (o, None)


def tseed(o):
    s = method_of(o)[1]
    return 0 if s is None else s


def settings():
    ev = tomllib.loads((ROOT / "configs/evaluation.toml").read_text())
    mt = tomllib.loads((ROOT / "configs/metrics.toml").read_text())
    want = [("yoon.cell", ev["yoon"]["cell"], "gru"), ("yoon.layers", ev["yoon"]["layers"], 2),
            ("yoon.hidden", ev["yoon"]["hidden"], "dim_div_2"), ("yoon.iterations", ev["yoon"]["iterations"], ITER_RNN),
            ("yoon.batch", ev["yoon"]["batch"], BATCH), ("yoon.threshold", ev["yoon"]["threshold"], 0.5),
            ("minirocket.num_kernels", ev["minirocket"]["num_kernels"], NK),
            ("minirocket.classifier", ev["minirocket"]["classifier"], "RidgeClassifierCV"),
            ("minirocket.alphas", ev["minirocket"]["alphas"], "logspace(-3,3,10)"),
            ("split.classifier_seeds", ev["split"]["classifier_seeds"], len(SEEDS)),
            ("rnn.scale", mt["rnn"]["scale"], "minmax_train_half_both_classes"),
            ("rnn.halves", mt["rnn"]["halves"], "results/phase3/simprocess/selection.json"),
            ("minirocket.scale", mt["minirocket"]["scale"], "standard_scaler_on_features_train_half"),
            ("tstr.cell", mt["tstr"]["cell"], "gru"), ("tstr.layers", mt["tstr"]["layers"], 2),
            ("tstr.hidden", mt["tstr"]["hidden"], "int(dim_in/2)"), ("tstr.iterations", mt["tstr"]["iterations"], ITER_TSTR),
            ("tstr.batch", mt["tstr"]["batch"], BATCH), ("tstr.output", mt["tstr"]["output"], "linear"),
            ("tstr.scale", mt["tstr"]["scale"], "minmax_phase2_training_windows"),
            ("tstr.metric", mt["tstr"]["metric"], "mae_mean_over_tags"),
            ("tstr.seeds_per_training_seed", mt["tstr"]["seeds_per_training_seed"], 1),
            ("psyn_flag.windows", mt["psyn_flag"]["windows"], "all_p_syn")]
    bad = [(k, a, b) for k, a, b in want if a != b]
    if bad:
        stop(f"toml settings differ from the implementation: {bad}")


# ---------------------------------------------------------------- worker side
_C = {}


def _ctx(pn):
    if pn not in _C:
        cfg = data.load_cfg()
        P = data.load_plant(cfg, pn)
        W, H, ncu = P.W, P.H, P.nc + P.ne
        tr_w = np.asarray(P.tr)
        lo, hi = tr_w.min((0, 1)).astype(np.float64), tr_w.max((0, 1)).astype(np.float64)
        d = hi - lo
        d[d < 1e-12] = 1.0
        del tr_w, P
        hv = json.loads(SELP.read_text())[pn]["halves"]
        tr = np.array(sorted(i for f in hv for i in hv[f]["train"]))
        te = np.array(sorted(i for f in hv for i in hv[f]["test"]))
        X = np.load(DECOY / pn / "normal" / "X.npy", mmap_mode="r")
        R = np.asarray(X[:, W:, ncu:], np.float32)
        CE = np.asarray(X[:, W:, :ncu], np.float32)
        _C[pn] = dict(W=W, H=H, ncu=ncu, nr=R.shape[-1], tr=tr, te=te, R=R, CE=CE, tlo=lo, td=d, N=len(R))
    return _C[pn]


def _dec(c, pn, s, o):
    y = np.load(DECOY / pn / s / f"{o}.npy").astype(np.float32)
    if y.shape[1:] != (c["H"], c["nr"]) or (s == "normal" and len(y) != c["N"]):
        raise ValueError(f"{pn} {s} {o}: decoy shape {y.shape}")
    return y


class GRUHead(nn.Module):
    def __init__(self, din, h, layers, dout, seq):
        super().__init__()
        self.seq = seq
        self.gru = nn.GRU(din, h, layers, batch_first=True)
        self.lin = nn.Linear(h, dout)

    def forward(self, x):
        o, hN = self.gru(x)
        return self.lin(o) if self.seq else self.lin(hN[-1]).squeeze(-1)


def _clf_out(c, sr, sd_, sp, fin, thr, extra):
    acc = (np.sum(sr > thr) + np.sum(~(sd_ > thr))) / (len(sr) + len(sd_))
    p = np.full(len(fin), np.nan)
    p[fin] = sp
    flag = ~fin
    flag[fin] = ~(sp > thr)
    return dict(acc=float(acc), m2=float(abs(acc - 0.5)), threshold=thr, psyn_flag_rate=float(flag.mean()),
                psyn_nonfinite=int((~fin).sum()), test_real=np.asarray(sr, np.float64), test_decoy=np.asarray(sd_, np.float64),
                test_index=c["te"], p_syn=p, psyn_flag=flag, **extra)


def rnn(c, Yn, Yp, seed):
    tr, te, R = c["tr"], c["te"], c["R"]
    Rtr, Dtr = R[tr], Yn[tr]
    lo = np.minimum(Rtr.min((0, 1)), Dtr.min((0, 1)))
    d = np.maximum(Rtr.max((0, 1)), Dtr.max((0, 1))) - lo
    d[d < 1e-12] = 1.0
    f = lambda a: torch.from_numpy(((a - lo) / d).astype(np.float32))
    xr, xd = f(Rtr), f(Dtr)
    dim = R.shape[-1]
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    m = GRUHead(dim, dim // 2, 2, 1, False)
    opt = torch.optim.Adam(m.parameters())
    bce = nn.functional.binary_cross_entropy_with_logits
    for _ in range(ITER_RNN):
        a_ = m(xr[torch.from_numpy(rng.choice(len(xr), BATCH, replace=False))])
        b_ = m(xd[torch.from_numpy(rng.choice(len(xd), BATCH, replace=False))])
        loss = bce(a_, torch.ones_like(a_)) + bce(b_, torch.zeros_like(b_))
        if not torch.isfinite(loss):
            raise FloatingPointError("RNN loss non-finite")
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()

    def p(a):
        with torch.no_grad():
            return torch.sigmoid(torch.cat([m(b) for b in f(a).split(512)])).numpy().astype(np.float64)
    fin = np.isfinite(Yp).all((1, 2))
    sp = p(Yp[fin]) if fin.any() else np.zeros(0)
    return _clf_out(c, p(R[te]), p(Yn[te]), sp, fin, 0.5, {"final_loss": float(loss)})


def mrocket(c, Yn, Yp, seed):
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler
    tr, te, R = c["tr"], c["te"], c["R"]
    T = lambda a: np.ascontiguousarray(np.transpose(a, (0, 2, 1)), dtype=np.float64)
    mr = MiniRocketMultivariate(num_kernels=NK, random_state=seed, n_jobs=1)
    Ftr = np.asarray(mr.fit_transform(T(np.concatenate([R[tr], Yn[tr]]))), np.float64)
    y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    sc = StandardScaler().fit(Ftr)
    Ztr = sc.transform(Ftr)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Ztr, y)
    if not np.array_equal(clf.predict(Ztr), (clf.decision_function(Ztr) > 0).astype(y.dtype)):
        raise ValueError("RidgeClassifierCV decision rule is not 'real if decision > 0'")
    g = lambda a: clf.decision_function(sc.transform(np.asarray(mr.transform(T(a)), np.float64)))
    fin = np.isfinite(Yp).all((1, 2))
    sp = g(Yp[fin]) if fin.any() else np.zeros(0)
    return _clf_out(c, g(R[te]), g(Yn[te]), sp, fin, 0.0, {"alpha": float(clf.alpha_), "n_features": int(Ftr.shape[1])})


def tstr(c, resp_tr, seed):
    tr, te, ncu = c["tr"], c["te"], c["ncu"]
    f = lambda a: torch.from_numpy(((a.astype(np.float64) - c["tlo"]) / c["td"]).astype(np.float32))
    ztr = f(np.concatenate([c["CE"][tr], resp_tr], -1))
    zte = f(np.concatenate([c["CE"][te], c["R"][te]], -1))
    C = ztr.shape[-1]
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    m = GRUHead(C, int(C / 2), 2, C - ncu, True)
    opt = torch.optim.Adam(m.parameters())
    for _ in range(ITER_TSTR):
        xb = ztr[torch.from_numpy(rng.choice(len(ztr), BATCH, replace=False))]
        loss = (m(xb[:, :-1]) - xb[:, 1:, ncu:]).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("TSTR loss non-finite")
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        err = torch.cat([(m(b[:, :-1]) - b[:, 1:, ncu:]).abs() for b in zte.split(128)])
    per = err.mean(dim=(0, 1)).numpy().astype(np.float64)
    return {"mae": float(per.mean()), "per_tag": per, "final_loss": float(loss), "hidden": int(C / 2)}


def run_job(j):
    torch.set_num_threads(1)
    pn, kind, o, s = j
    t = time.time()
    c = _ctx(pn)
    if kind == "trtr":
        r = tstr(c, c["R"][c["tr"]], s)
    else:
        Yn = _dec(c, pn, "normal", o)
        if not np.isfinite(Yn).all():
            raise ValueError(f"{pn} {o}: non-finite decoy values in normal windows")
        if kind == "tstr":
            r = tstr(c, Yn[c["tr"]], s)
        else:
            Yp = _dec(c, pn, "p_syn", o)
            r = (rnn if kind == "rnn" else mrocket)(c, Yn, Yp, s)
    r["seconds"] = time.time() - t
    return j, r


# ---------------------------------------------------------------- main side
def jpath(j):
    pn, kind, o, s = j
    return PART / "jobs" / pn / kind / f"{o}_c{s}.npz"


def save(j, r):
    p = jpath(j)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arr)
    os.replace(tmp, p)


def jobs(outs, smoke):
    J = []
    for pn in ("hai", "qappd"):
        os_ = list(SMOKE_OUTS) if smoke else outs[pn]
        cs = [0] if smoke else SEEDS
        J += [(pn, "tstr", o, tseed(o)) for o in os_] + [(pn, "trtr", "real", s) for s in cs]
        J += [(pn, "rnn", o, s) for o in os_ for s in cs] + [(pn, "mr", o, s) for o in os_ for s in cs]
    if smoke:
        J.append(("hai", "rnn", "C1-Full_s0", 0))
    return sorted(J, key=lambda j: PRIO[j[1]])


def describe(j, r):
    pn, kind, o, s = j
    if kind in ("rnn", "mr"):
        return f"acc {r['acc']:.3f} M2 {r['m2']:.3f} P-syn flag {r['psyn_flag_rate']:.3f} (non-finite {r['psyn_nonfinite']})"
    return f"MAE {r['mae']:.4f} (final train loss {r['final_loss']:.4f})"


def aggregate(outs):
    S = {}
    for pn in ("hai", "qappd"):
        ip = json.loads((DECOY / pn / "p_syn" / "index.json").read_text())
        st, kd = np.array([e["set"] for e in ip]), np.array([e["kind"] for e in ip])
        O = {}
        for o in outs[pn]:
            e = {}
            for kind in ("rnn", "mr"):
                zs = [np.load(jpath((pn, kind, o, s))) for s in SEEDS]
                ms = [json.loads(z["meta"].item()) for z in zs]
                fl = np.stack([z["psyn_flag"] for z in zs]).astype(float)
                if fl.shape[1] != len(ip):
                    stop(f"{pn} {kind} {o}: P-syn flags {fl.shape[1]} != {len(ip)} windows")
                e[kind] = {"acc": float(np.mean([m["acc"] for m in ms])), "m2": float(np.mean([m["m2"] for m in ms])),
                           "acc_per_seed": [m["acc"] for m in ms], "psyn_nonfinite": ms[0]["psyn_nonfinite"],
                           "psyn_flag": {"all": float(fl.mean()), "in": float(fl[:, st == "in"].mean()),
                                         "oor": float(fl[:, st == "oor"].mean()),
                                         "by_kind": {k: float(fl[:, kd == k].mean()) for k in sorted(set(kd.tolist()))}}}
            e["tstr_mae"] = json.loads(np.load(jpath((pn, "tstr", o, tseed(o))))["meta"].item())["mae"]
            O[o] = e
        trm = [json.loads(np.load(jpath((pn, "trtr", "real", s)))["meta"].item())["mae"] for s in SEEDS]
        S[pn] = {"outputs": O, "trtr_mae": trm}
        say(f"== {pn}: mean over training seeds (classifier seeds 0-4 each) | TRTR MAE {np.mean(trm):.4f} +- {np.std(trm, ddof=1):.4f}")
        gm = {}
        for o in outs[pn]:
            gm.setdefault(method_of(o)[0], []).append(o)
        for m, os_ in gm.items():
            g = lambda fn: float(np.mean([fn(O[o]) for o in os_]))
            say(f"   {m:8s} n={len(os_)} | RNN acc {g(lambda e: e['rnn']['acc']):.3f} M2 {g(lambda e: e['rnn']['m2']):.3f} | "
                f"MiniRocket acc {g(lambda e: e['mr']['acc']):.3f} M2 {g(lambda e: e['mr']['m2']):.3f} | P-syn flag oor "
                f"RNN {g(lambda e: e['rnn']['psyn_flag']['oor']):.3f} MR {g(lambda e: e['mr']['psyn_flag']['oor']):.3f} in "
                f"RNN {g(lambda e: e['rnn']['psyn_flag']['in']):.3f} MR {g(lambda e: e['mr']['psyn_flag']['in']):.3f} | "
                f"TSTR MAE {g(lambda e: e['tstr_mae']):.4f}")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    import sktime
    say(f"Stage 3d-3 classifiers | commit {commit[:7]}{' (dirty)' if dirty else ''} | torch {torch.__version__} numpy "
        f"{np.__version__} sktime {sktime.__version__} | {PROCS} processes x 1 thread | {'SMOKE' if a.smoke else 'FULL'}")
    if not a.smoke:
        if dirty:
            stop("full run needs a clean, committed tree")
        if OUT.exists():
            stop(f"{OUT.relative_to(ROOT)} exists; not overwriting")
    if data.cfg_hash() != CFG_HASH:
        stop(f"training config hash {data.cfg_hash()}")
    if sha256(ROOT / "configs/metrics.toml")[:12] != MET_SHA12:
        stop("configs/metrics.toml differs from D51")
    settings()
    st3c = json.loads((P3 / "stage3c.json").read_text())
    if st3c["provenance"].get("dirty") or st3c["provenance"].get("smoke"):
        stop("stage3c.json was not written by a clean full run")
    man = [l for l in (ROOT / "spec/phase3_simprocess_manifest.sha256").read_text().splitlines() if l.strip().endswith("selection.json")]
    hx = [re.search(r"\b[0-9a-f]{64}\b", l) for l in man]
    if len(man) != 1 or not hx[0] or hx[0].group(0) != sha256(SELP):
        stop("selection.json differs from the D47 manifest entry")
    say("gate 1 OK: config 59a11b235108, metrics.toml a1a995717a64, toml settings = implementation, selection.json = D47 manifest")
    sel = json.loads(SELP.read_text())
    outs = {}
    for pn in ("hai", "qappd"):
        idx_n = json.loads((DECOY / pn / "normal" / "index.json").read_text())
        idx_p = json.loads((DECOY / pn / "p_syn" / "index.json").read_text())
        names = {s: sorted(p.name[:-4] for p in (DECOY / pn / s).glob("*.npy") if p.name != "X.npy") for s in ("normal", "p_syn")}
        if len(names["normal"]) != N_OUT or names["normal"] != names["p_syn"]:
            stop(f"{pn}: decoy output lists differ or != {N_OUT}")
        outs[pn] = names["normal"]
        hv = sel[pn]["halves"]
        tr = [i for f in hv for i in hv[f]["train"]]
        te = [i for f in hv for i in hv[f]["test"]]
        if ((len(tr), len(te)) != HALF_N[pn] or len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) & set(te)
                or max(tr + te) >= len(idx_n)):
            stop(f"{pn}: halves {len(tr)}/{len(te)} invalid (expected {HALF_N[pn]})")
        say(f"   {pn}: halves {len(tr)} train / {len(te)} test of {len(idx_n)} normal windows | P-syn {len(idx_p)} | {N_OUT} outputs")
    files = st3c["files"]
    disk = sorted(str(p.relative_to(DECOY)) for p in DECOY.rglob("*") if p.is_file() and not p.name.startswith("."))
    if disk != sorted(files):
        stop("decoy folder differs from stage3c.json file list")
    if a.smoke:
        check = [k for k in files if k.endswith(("/X.npy", "/index.json"))
                 or any(k.rsplit("/", 1)[-1].startswith(o + ".") for o in SMOKE_OUTS)]
    else:
        check = sorted(files)
    t = time.time()
    bad = [k for k in check if sha256(DECOY / k) != files[k]["sha256"]]
    if bad:
        stop(f"{len(bad)} decoy files differ from stage3c.json, e.g. {bad[:5]}")
    say(f"gate 2 OK: {len(check)} of {len(files)} decoy files = stage3c.json sha256 ({time.time() - t:.0f} s)")

    todo = jobs(outs, a.smoke)
    total = len(todo)
    if not a.smoke:
        ident = {"commit": commit, "script_sha256": sha256(Path(__file__))}
        if PART.exists():
            if json.loads((PART / "identity.json").read_text()) != ident:
                stop("classifiers.partial comes from a different commit or script; to restart: rm -rf results/phase3/classifiers.partial")
            for p in PART.rglob("*.tmp.npz"):
                p.unlink()
        else:
            (PART / "jobs").mkdir(parents=True)
            (PART / "identity.json").write_text(json.dumps(ident))
        todo = [j for j in todo if not jpath(j).exists()]
        say(f"resume: {total - len(todo)} of {total} jobs already done, {len(todo)} to run")
    results, obs, pending = [], {}, list(todo)
    t1 = time.time()
    with mp.get_context("spawn").Pool(PROCS) as pool:
        it = pool.imap_unordered(run_job, todo)
        for k in range(1, len(todo) + 1):
            try:
                j, r = next(it)
            except Exception as e:
                pool.terminate()
                stop(f"job failed: {e!r}")
            pending.remove(j)
            obs.setdefault((j[0], j[1]), []).append(r["seconds"])
            allm = float(np.mean([x for v in obs.values() for x in v]))
            left = sum(np.mean(obs.get((q[0], q[1]), [allm])) for q in pending) / PROCS
            if a.smoke:
                results.append((j, r))
            else:
                save(j, r)
            say(f"   [{k}/{len(todo)}] {j[0]:5s} {j[1]:4s} {j[2]:11s} c{j[3]} {describe(j, r)} | {r['seconds']:.0f} s | "
                f"{el()} | left ~{el(left)}")
    if a.smoke:
        rep = [r for j, r in results if j == ("hai", "rnn", "C1-Full_s0", 0)]
        if len(rep) != 2 or not all(np.array_equal(rep[0][k], rep[1][k], equal_nan=True) for k in ("test_real", "test_decoy", "p_syn")):
            stop("RNN job not reproducible across processes (hai C1-Full_s0 c0)")
        say("determinism OK: hai rnn C1-Full_s0 c0 identical in two processes")
        full = {"tstr": N_OUT, "trtr": len(SEEDS), "rnn": N_OUT * len(SEEDS), "mr": N_OUT * len(SEEDS)}
        tot = sum(np.mean(v) * full[k[1]] for k, v in obs.items())
        say("seconds per job: " + ", ".join(f"{k[0]} {k[1]} {np.mean(v):.0f}" for k, v in sorted(obs.items())))
        say(f"projected full run: about {tot / PROCS / 3600:.1f} h at {PROCS} processes (plus a few minutes hashing decoy files)")
        say("SMOKE 3d-3 PASSED (nothing written)")
        return
    S = aggregate(outs)
    res = {"stage": "3d-3", "provenance": {"commit": commit, "dirty": dirty, "smoke": False, "config_sha12": CFG_HASH,
           "metrics_toml_sha256": sha256(ROOT / "configs/metrics.toml"),
           "evaluation_toml_sha256": sha256(ROOT / "configs/evaluation.toml"), "selection_sha256": sha256(SELP),
           "script_sha256": sha256(Path(__file__)), "torch": torch.__version__, "numpy": np.__version__}, "plants": S,
           "seconds_this_session": time.time() - t1}
    (PART / "summary.json").write_text(json.dumps(res, indent=1))
    fl = {str(p.relative_to(PART)): sha256(p) for p in sorted(PART.rglob("*")) if p.is_file()}
    (PART / "files.json").write_text(json.dumps(fl, indent=1))
    os.replace(PART, OUT)
    say(f"STAGE 3d-3 DONE -> results/phase3/classifiers | {len(fl)} files | summary.json sha256 {sha256(OUT / 'summary.json')} | "
        f"files.json sha256 {sha256(OUT / 'files.json')} | {el()}")


if __name__ == "__main__":
    main()
