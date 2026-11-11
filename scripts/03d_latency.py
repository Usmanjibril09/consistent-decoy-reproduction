#!/usr/bin/env python
"""Stage 3d-4: per-step latency and model size (protocol E8; D51(h); configs [latency]).

Validation windows only (data.load_plant); no held-out file is opened. Models: selected configuration, training seed 0, loaded
exactly as stage 3c (scripts/03_evaluate.py frozen/load_model). One thread. Every step function timed here is first checked
against the method's own batch output on two validation windows (max |diff| <= 1e-4, z units): the timed code reproduces the
code that produced the results. GRU/LSTM models carry their state one step at a time; B8 recomputes its full window at every
step (no cache; last position = worst case); C2/C3 cost included for C1-Full and B8-L; B6 generates a whole window at once and
is timed per window (per-step value = window time / H, labelled amortised).
"""
import argparse, hashlib, importlib.util, json, platform, subprocess, sys, time, tomllib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoy import baselines, constraints, data, models  # noqa: E402
_spec = importlib.util.spec_from_file_location("evaluate03", ROOT / "scripts/03_evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
torch.set_num_threads(1)

OUT = ROOT / "results/phase3/latency"
CFG_SHA = "59a11b235108"
SEED = 0
GATE_TOL = 1e-4
PERIOD_MS = {"hai": 1000.0, "qappd": 10.0}      # sampling periods (protocol section 4 runtime budget)
PLANTS = ("hai", "qappd")


def say(*a):
    print(*a, flush=True)


def stop(m):
    say(f"STOP: {m}")
    sys.exit(1)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def T(X):
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))


def settings():
    e = tomllib.loads((ROOT / "configs/evaluation.toml").read_text())["latency"]
    m = tomllib.loads((ROOT / "configs/metrics.toml").read_text())["latency"]
    got = (e["steps"], e["batch"], e["position"], m["threads"], m["warmup_steps"])
    if got != (1000, 1, "last", 1, 50):
        stop(f"latency settings in configs differ from D51(h): {got}")
    return int(e["steps"]), int(m["warmup_steps"])


# ---------- step functions: prepare(X, i) runs untimed set-up (encoder warm-up), returns step() -> response (nr,) ----------
def prep_gru(m):
    def prepare(X, i):
        b = T(X)
        _, h = m.enc(b[:, :m.W])
        st = {"h": h, "t": m.W}

        def step():
            t = st["t"]
            parts = ([b[:, t:t + 1, :m.nc]] if m.use_cmd else []) + ([b[:, t:t + 1, m.nc:m.ncu]] if m.ne else [])
            x = torch.cat(parts, -1) if parts else b.new_ones(1, 1, 1)
            o, st["h"] = m.dec(x, st["h"])
            st["t"] = t + 1
            return m.head(o)[0, 0].numpy()
        return step
    return prepare


def prep_tf(m):
    def prepare(X, i):
        b = T(X)
        st = {"k": 0}

        def step():
            k = st["k"]
            st["k"] = k + 1
            return m.respond(b)[0, k].numpy()      # full window recomputed every step (no cache)
        return step
    return prepare


def prep_neupot(m):
    def prepare(X, i):
        b = T(X)
        _, hc = m.enc(b[:, :m.W])
        st = {"hc": hc, "prev": b[:, m.W - 1:m.W, m.ncu:], "t": m.W}

        def step():
            t = st["t"]
            o, st["hc"] = m.dec(torch.cat([b[:, t:t + 1, :m.ncu], st["prev"]], -1), st["hc"])
            st["prev"] = m.head(o)
            st["t"] = t + 1
            return st["prev"][0, 0].numpy()
        return step
    return prepare


def prep_lstm(m):
    def prepare(X, i):
        b = T(X)
        o, h = m.rnn(m.values(b)[:, :m.W])
        st = {"h": h, "prev": m.head(o[:, -1:])}

        def step():
            y = st["prev"]
            o, st["h"] = m.rnn(y, st["h"])
            st["prev"] = m.head(o)
            return y[0, 0, m.nc:].numpy()
        return step
    return prepare


def prep_rcgan(m):
    def prepare(X, i):
        b = T(X)
        _, h = m.enc(b[:, :m.W])
        z = torch.randn((1, b.shape[1] - m.W, m.nz), generator=torch.Generator().manual_seed(SEED))
        st = {"h": h, "k": 0}

        def step():
            k = st["k"]
            t = m.W + k
            o, st["h"] = m.gen(torch.cat([b[:, t:t + 1, :m.ncu], z[:, k:k + 1]], -1), st["h"])
            st["k"] = k + 1
            return m.head(o)[0, 0].numpy()
        return step
    return prepare


def _seq(net, x, h):
    o, h = net.rnn(x, h)
    y = net.head(o)
    return (y.sigmoid() if net.sigmoid else y), h


def prep_timegan(m):
    def prepare(X, i):
        b = T(X)
        z = m.noise(b, torch.Generator().manual_seed(SEED))
        st = {"k": 0, "h": [None, None, None]}

        def step():
            k, h = st["k"], st["h"]
            e, h[0] = _seq(m.gen, z[:, k:k + 1], h[0])
            s, h[1] = _seq(m.supervisor, e, h[1])
            r, h[2] = _seq(m.recover, s, h[2])
            st["k"] = k + 1
            return (r * m.span + m.lo)[0, 0, m.nc:].numpy()
        return step
    return prepare


def prep_window(m):
    def prepare(X, i):
        b = T(X)

        def step():
            return m.respond(b, torch.Generator().manual_seed(SEED))[0]
        return step
    return prepare


def layered(base_prepare, c, grey):
    """C2/C3 one step at a time; same operations as constraints.apply(variant='Full') (checked by the gate)."""
    n, nc, W = c["ncu"], c["nc"], c["W"]
    mu, sd = np.array(c["mu"]), np.array(c["sd"])
    lo, hi = np.array(c["lo"][:nc]), np.array(c["hi"][:nc])
    tol = 1e-6 * np.maximum(1, hi - lo)
    span = c["config"]["c3_frac"] * np.maximum(hi - lo, 1e-8)

    def prepare(X, i):
        base = base_prepare(X, i)
        st = {"t": W, "g": X[:, W - 1, n:].astype("float64").copy(), "p": X[:, W - 1, n:] * sd[n:] + mu[n:]}

        def step():
            t = st["t"]
            y = base().astype(np.float64)
            u = X[0, t, :nc] * sd[:nc] + mu[:nc]
            w = float(np.clip((np.maximum(np.maximum(lo - u, u - hi) - tol, 0) / span).max(), 0, 1))
            g = st["g"]
            for j, p in enumerate(grey):
                g[:, j] = p["a"] * g[:, j] + p["b"] * X[:, t - 1 - p["delay"], p["input"]] + p["c"]
            y = (1 - w) * y + w * g[0].astype("float32")
            st["p"] = constraints.local_limit(c, st["p"], (y * sd[n:] + mu[n:])[None], X[:, t, :n] * sd[:n] + mu[:n])
            st["t"] = t + 1
            return ((st["p"][0] - mu[n:]) / sd[n:]).astype("float32")
        return step
    return prepare


def gate(prep, ref, Xg, H, label):
    worst = 0.0
    for i, X in enumerate(Xg):
        step = prep(X, i)
        got = np.stack([np.asarray(step(), np.float64) for _ in range(H)])
        want = np.asarray(ref(i), np.float64)
        if got.shape != want.shape:
            stop(f"{label}: step output {got.shape} vs batch output {want.shape}")
        if not np.isfinite(got).all():
            stop(f"{label}: non-finite step output")
        worst = max(worst, float(np.abs(got - want).max()))
    if not worst <= GATE_TOL:
        stop(f"{label}: step function differs from the batch output by {worst:.3g} (> {GATE_TOL})")
    return worst


def bench(prep, per_win, total, va):
    t, k = [], 0
    while len(t) < total:
        step = prep(va[(k * 97) % len(va)][None], k)
        k += 1
        for _ in range(per_win):
            t0 = time.perf_counter_ns()
            step()
            t.append(time.perf_counter_ns() - t0)
            if len(t) == total:
                break
    return np.array(t, np.float64) / 1e6


def plant(cfg, pn, steps, warm, raw):
    t_p = time.time()
    P = data.load_plant(cfg, pn)
    ncu, W, H = P.nc + P.ne, P.W, P.H
    c = constraints.fit(P, cfg["train"]["c2"])
    where = ev.same(ev.jsonable(c), ev.rjson(ev.LAYERS / pn / "layers" / "parameters.json"), "c2")
    if where:
        stop(f"{pn}: refitted C2/C3 parameters differ from results/phase2_d32 at {where}")
    grey = ev.rjson(ev.PHASE2 / pn / "B3" / "parameters.json")
    k2, hp2 = ev.selection(pn, "B2")
    theta = np.load(ev.PHASE2 / pn / "B2" / f"cfg{k2:02d}" / "theta.npy", allow_pickle=False)
    M = ev.frozen(pn, P, SEED)
    for m in M.values():
        m.eval()
    va = P.va
    gi = [0, len(va) // 2]
    Xg = [va[i:i + 1] for i in gi]
    pick = np.random.default_rng(SEED).integers(0, len(P.tr), size=len(gi))
    say(f"== {pn}: W {W} H {H} | validation windows {len(va)} | gate windows {gi} | sampling period {PERIOD_MS[pn]:g} ms")

    def nref(m, gen=False):
        return lambda i: m.respond(T(Xg[i]), torch.Generator().manual_seed(SEED) if gen else None)[0].numpy()

    def lref(m):
        return lambda i: constraints.apply(c, grey, Xg[i], nref(m)(i)[None], "Full")[0]

    def prep_b1(X, i):
        r = P.tr[pick[i % len(pick)]]
        st = {"t": W}

        def step():
            t = st["t"]
            st["t"] = t + 1
            return r[t, ncu:].copy()
        return step

    def prep_b2(X, i):
        na, nb = hp2["na"], hp2["nb"]
        U = X[..., :ncu].astype(np.float64)
        Ys = X[..., ncu:].astype(np.float64).copy()
        one = np.ones((1, 1))
        st = {"t": W}

        def step():
            t = st["t"]
            st["t"] = t + 1
            phi = np.concatenate([Ys[:, t - k] for k in range(1, na + 1)] + [U[:, t - k] for k in range(nb)] + [one], 1)
            Ys[:, t] = phi @ theta
            return Ys[0, t].copy()
        return step

    def prep_b3(X, i):
        prev = X[:, W - 1, ncu:].astype("float64").copy()
        st = {"t": W}

        def step():
            t = st["t"]
            st["t"] = t + 1
            for j, p in enumerate(grey):
                prev[:, j] = p["a"] * prev[:, j] + p["b"] * X[:, t - 1 - p["delay"], p["input"]] + p["c"]
            return prev[0].astype("float32")
        return step

    table = [
        ("C1", "step", prep_gru(M["C1"]), nref(M["C1"]), M["C1"]),
        ("C1-Full", "step + C2/C3", layered(prep_gru(M["C1"]), c, grey), lref(M["C1"]), M["C1"]),
        ("A3", "step", prep_gru(M["A3"]), nref(M["A3"]), M["A3"]),
        ("B8", "full-context", prep_tf(M["B8"]), nref(M["B8"]), M["B8"]),
        ("B8-L", "full-context + C2/C3", layered(prep_tf(M["B8"]), c, grey), lref(M["B8"]), M["B8"]),
        ("B9", "step", prep_neupot(M["B9"]), nref(M["B9"]), M["B9"]),
        ("B4", "step", prep_lstm(M["B4"]), nref(M["B4"]), M["B4"]),
        ("B5", "step", prep_rcgan(M["B5"]), nref(M["B5"], True), M["B5"]),
        ("B7", "step", prep_timegan(M["B7"]), nref(M["B7"], True), M["B7"]),
        ("B6", "window (amortised)", prep_window(M["B6"]), None, M["B6"]),
        ("B1", "step", prep_b1, lambda i: baselines.b1_replay(P.tr, np.concatenate(Xg), P.nc, P.ne, W, SEED)[i], None),
        ("B2", "step", prep_b2, lambda i: baselines.arx_sim(theta, Xg[i], P.nc, P.ne, W, hp2["na"], hp2["nb"])[0], None),
        ("B3", "step", prep_b3, lambda i: baselines.grey_sim(grey, Xg[i], W, ncu)[0], None),
    ]
    out, t_bench = {}, 0.0
    for name, kind, prep, ref, m in table:
        g = None if ref is None else gate(prep, ref, Xg, H, f"{pn} {name}")
        t0 = time.time()
        ms = bench(prep, 1 if kind.startswith("window") else H, warm + steps, va)[warm:]
        t_bench += time.time() - t0
        per_step = ms / H if kind.startswith("window") else ms
        size = models.n_params(m) if m is not None else {"B1": 0, "B2": int(theta.size), "B3": 3 * len(grey)}[name]
        r = {"kind": kind, "gate_max_abs_diff": g, "timed_calls": int(len(ms)),
             "median_ms": float(np.median(per_step)), "p90_ms": float(np.percentile(per_step, 90)),
             "p99_ms": float(np.percentile(per_step, 99)), "mean_ms": float(per_step.mean()),
             "per_window_median_ms": float(np.median(ms)) if kind.startswith("window") else None,
             "within_period": bool(np.median(per_step) < PERIOD_MS[pn]), "parameters": int(size)}
        out[name] = r
        raw[f"{pn}/{name}"] = ms
        gs = "n/a" if g is None else f"{g:.1e}"
        say(f"   {name:8s} {kind:21s} gate {gs:>7s} | median {r['median_ms']:9.4f} ms  p99 {r['p99_ms']:9.4f} ms"
            f" | params {size:>8d} | {'within' if r['within_period'] else 'ABOVE'} {PERIOD_MS[pn]:g} ms")
    return {"W": W, "H": H, "period_ms": PERIOD_MS[pn], "gate_windows": gi, "seed": SEED, "outputs": out,
            "notes": {"B1": "parameters 0; replays stored training windows", "B2": "theta entries",
                      "B3": "3 continuous parameters per tag (input and delay discrete)",
                      "C1-Full": "C1 parameters; C2/C3 are fitted constants", "B8-L": "B8 parameters; C2/C3 fitted constants",
                      "B4": "timed without sampling noise (one add per step)"},
            "seconds": time.time() - t_p, "seconds_timing": t_bench}


def main():
    ap = argparse.ArgumentParser(description="Stage 3d-4 latency and model size")
    ap.add_argument("--smoke", action="store_true", help="10 warm-up + 50 timed calls per method; writes nothing")
    a = ap.parse_args()
    steps, warm = settings()
    if a.smoke:
        steps, warm = 50, 10
    if data.cfg_hash() != CFG_SHA:
        stop(f"training config hash {data.cfg_hash()}")
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    if not a.smoke:
        if dirty:
            stop("working tree not clean (commit the script first)")
        if OUT.exists():
            stop("results/phase3/latency exists; latency already measured")
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    say(f"Stage 3d-4 latency | commit {commit[:7]}{' (dirty)' if dirty else ''} | config {CFG_SHA} | torch {torch.__version__}"
        f" numpy {np.__version__} | threads {torch.get_num_threads()} | {cpu} | {'SMOKE' if a.smoke else 'FULL'}"
        f" ({warm} warm-up + {steps} timed calls per method)")
    cfg = data.load_cfg()
    t_all = time.time()
    raw = {}
    res = {"stage": "3d-4", "provenance": {"commit": commit, "dirty": dirty, "smoke": a.smoke, "config_sha12": CFG_SHA,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__,
           "numpy": np.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "cpu": cpu},
           "settings": {"warmup": warm, "timed": steps, "batch": 1, "gate_tol": GATE_TOL, "data": "validation windows"},
           "plants": {}}
    with torch.inference_mode():
        for pn in PLANTS:
            res["plants"][pn] = plant(cfg, pn, steps, warm, raw)
    res["seconds_total"] = time.time() - t_all
    if a.smoke:
        tb = sum(p["seconds_timing"] for p in res["plants"].values())
        proj = res["seconds_total"] - tb + tb * (1050 / 60)
        say(f"   projected full run: about {proj / 60:.0f} min")
        say("SMOKE 3d-4 PASSED (nothing written)")
        return
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "times_ms.npz", **{k.replace("/", "__"): v for k, v in raw.items()})
    (OUT / "latency.json").write_text(json.dumps(res, indent=1))
    h = {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in ("latency.json", "times_ms.npz")}
    say(f"STAGE 3d-4 DONE -> results/phase3/latency | latency.json sha256 {h['latency.json']} | "
        f"times_ms.npz sha256 {h['times_ms.npz']} | {res['seconds_total']:.0f} s")


if __name__ == "__main__":
    main()
