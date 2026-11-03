#!/usr/bin/env python
"""Phase 2 (D28): train on training splits, select on validation. Held-out files are never opened.

Stage 1 methods: B1 record-and-replay, B2 linear ARX, C1 command-conditioned GRU, A3 C1 without commands.
  python scripts/02_train.py --smoke                      fast end-to-end check (writes results/smoke/)
  python scripts/02_train.py --plant hai --methods C1,A3  real run (clean git tree required)
"""
from __future__ import annotations
import argparse, itertools, json, os, platform, random, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
from decoy import baselines, data, metrics, models  # noqa: E402

ORDER = ["B1", "B2", "C1", "A3"]
TRAINED = [0]


def say(*a):
    print(*a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def wjson(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def score(v):
    return -np.inf if (v is None or np.isnan(v)) else float(v)


def expand(g, n):
    keys = list(g)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]
    if len(combos) != n:
        raise SystemExit(f"STOP: grid has {len(combos)} configurations, protocol requires {n}")
    return combos


def provenance(args, dev):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        f = ROOT / "CODE_COMMIT"
        if not f.exists():
            raise SystemExit("STOP: not a git repo and no CODE_COMMIT file")
        commit, dirty = f.read_text().strip(), ""
    if dirty and not args.smoke:
        raise SystemExit("STOP: uncommitted changes; only --smoke may run on a dirty tree\n" + dirty)
    return {"commit": commit, "dirty": bool(dirty), "config_sha12": data.cfg_hash(), "argv": sys.argv, "smoke": args.smoke,
            "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "threads": torch.get_num_threads(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def train_one(method, P, hp, k, seed, root, T, dev):
    d = root / f"cfg{k:02d}_seed{seed}"
    done = d / "done.json"
    if done.exists():
        r = json.loads(done.read_text())
        say(f"  {method} cfg{k:02d} seed{seed}: already done (val FIT {r['val_fit']:.2f})")
        return r
    seed_all(seed)
    t0 = time.time()
    m = models.Seq2SeqGRU(P.nc, P.ne, P.nr, hp["hidden"], hp["layers"], use_cmd=(method != "A3")).to(dev)
    st = models.train(m, P.tr, P.va, P.nc, P.ne, P.W, hp["lr"], T["batch"], T["max_epochs"], T["patience"], seed, d / "ckpt.pt", dev)
    per, mean = metrics.fit_percent(P.va[:, P.W:, P.nc + P.ne:], models.predict(m, P.va, P.nc, P.ne, P.W, dev))
    torch.save(m.state_dict(), d / "model.pt")
    TRAINED[0] += 1
    r = {"method": method, "plant": P.name, "cfg": k, "hp": hp, "seed": seed, "epochs": st["epoch"],
         "resumed_from": st["resumed_from"], "best_val_loss": st["best"], "val_fit": mean,
         "val_fit_per_tag": dict(zip(P.resp_names, per.tolist())), "n_params": models.n_params(m),
         "seconds": round(time.time() - t0, 1)}
    wjson(done, r)
    say(f"  {method} cfg{k:02d} seed{seed}: epochs {st['epoch']} val loss {st['best']:.4f} val FIT {mean:.2f} ({r['seconds']} s)")
    return r


def run_neural(method, P, T, out, dev, smoke):
    root = out / P.name / method
    seeds = ([T["seed_select"]] + list(T["seeds_final"]))[: (T["smoke"]["seeds"] if smoke else None)]
    if method == "C1":
        grid = expand(T["grid"]["C1"], T["grid_size"])[: (T["smoke"]["grid"] if smoke else None)]
        res = [train_one(method, P, hp, k, T["seed_select"], root, T, dev) for k, hp in enumerate(grid)]
        best = max(range(len(res)), key=lambda k: (score(res[k]["val_fit"]), -k))
        hp = grid[best]
        wjson(root / "selection.json", {"criterion": "validation free-run FIT% (mean over response tags), seed 0",
                                        "val_fit": [r["val_fit"] for r in res], "best": best, "hp": hp})
    else:
        sp = out / P.name / "C1" / "selection.json"
        if not sp.exists():
            raise SystemExit("STOP: A3 needs C1's selection; run C1 first")
        s = json.loads(sp.read_text())
        best, hp = s["best"], s["hp"]
    finals = [train_one(method, P, hp, best, sd, root, T, dev) for sd in seeds]
    return {"best_cfg": best, "hp": hp, "seed_fit": [r["val_fit"] for r in finals]}


def run_arx(P, T, out, smoke):
    root = out / P.name / "B2"
    grid = expand(T["grid"]["B2"], T["grid_size"])[: (T["smoke"]["grid"] if smoke else None)]
    res = []
    for k, hp in enumerate(grid):
        d = root / f"cfg{k:02d}"
        done = d / "done.json"
        if done.exists():
            res.append(json.loads(done.read_text()))
            continue
        t0 = time.time()
        th = baselines.arx_fit(P.tr, P.nc, P.ne, hp["na"], hp["nb"])
        yh = baselines.arx_sim(th, P.va, P.nc, P.ne, P.W, hp["na"], hp["nb"])
        per, mean = metrics.fit_percent(P.va[:, P.W:, P.nc + P.ne:], yh)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "theta.npy", th)
        r = {"method": "B2", "plant": P.name, "cfg": k, "hp": hp, "val_fit": mean,
             "val_fit_per_tag": dict(zip(P.resp_names, per.tolist())), "seconds": round(time.time() - t0, 1)}
        wjson(done, r)
        res.append(r)
        say(f"  B2 cfg{k:02d} na={hp['na']} nb={hp['nb']}: val FIT {mean:.2f}")
    best = max(range(len(res)), key=lambda k: (score(res[k]["val_fit"]), -k))
    wjson(root / "selection.json", {"criterion": "validation free-run FIT%", "val_fit": [r["val_fit"] for r in res],
                                    "best": best, "hp": grid[best]})
    return {"best_cfg": best, "hp": grid[best], "seed_fit": [res[best]["val_fit"]], "note": "deterministic least squares; one fit"}


def run_replay(P, T, smoke):
    seeds = ([T["seed_select"]] + list(T["seeds_final"]))[: (T["smoke"]["seeds"] if smoke else None)]
    yv = P.va[:, P.W:, P.nc + P.ne:]
    fits = [metrics.fit_percent(yv, baselines.b1_replay(P.tr, P.va, P.nc, P.ne, P.W, s))[1] for s in seeds]
    return {"best_cfg": None, "hp": None, "seed_fit": fits, "note": "no parameters; training windows replayed, drawn per seed"}


def selftest(P, T, out, dev):
    y = np.random.default_rng(0).normal(size=(4, 10, 3))
    assert abs(metrics.fit_percent(y, y)[1] - 100.0) < 1e-9, "FIT(y, y) != 100"
    assert abs(metrics.fit_percent(y, np.broadcast_to(y.reshape(-1, 3).mean(0), y.shape))[1]) < 1e-9, "FIT(y, mean) != 0"

    def run(name, epochs):
        seed_all(0)
        m = models.Seq2SeqGRU(P.nc, P.ne, P.nr, 16, 1).to(dev)
        return models.train(m, P.tr, P.va, P.nc, P.ne, P.W, 1e-3, T["batch"], epochs, 99, 0, out / "_selftest" / name / "ckpt.pt", dev)
    a1, a2, b = run("resumed", 1), run("resumed", 2), run("straight", 2)
    assert a1["epoch"] == 1 and a2["epoch"] == 2 and a2["resumed_from"] == 1, "checkpoint resume failed"
    assert np.allclose(a2["history"], b["history"], rtol=1e-5, atol=1e-7), f"resumed {a2['history']} != uninterrupted {b['history']}"
    say("  selftest OK: FIT sanity; resumed training identical to uninterrupted training")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", choices=["hai", "qappd", "all"], default="all")
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = data.load_cfg()
    T = dict(cfg["train"])
    if args.smoke:
        T["max_epochs"] = int(T["smoke"]["max_epochs"])
    asked = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in asked if m not in ORDER]
    if unknown:
        raise SystemExit(f"STOP: unknown methods {unknown}; stage 1 supports {ORDER}")
    methods = [m for m in ORDER if m in asked]
    plants = ["hai", "qappd"] if args.plant == "all" else [args.plant]
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(os.cpu_count() or 1)
    smoke_dir = ROOT / "results" / "smoke"
    out = Path(args.out).resolve() if args.out else (smoke_dir if args.smoke else ROOT / "results" / "phase2")
    if args.smoke and out == smoke_dir and out.exists():
        shutil.rmtree(out)
    prov = provenance(args, dev)
    say(f"Phase 2 | commit {prov['commit'][:12]}{' (dirty: smoke only)' if prov['dirty'] else ''} | config {prov['config_sha12']} | {dev} | out {out}")
    runlog = {"provenance": prov, "plants": {}}
    for pn in plants:
        P = data.load_plant(cfg, pn, args.smoke)
        say(f"=== {pn}: command {P.nc} exogenous {P.ne} response {P.nr} | warm {P.W} horizon {P.H} | windows train {len(P.tr)} val {len(P.va)}")
        say("  splits: " + ", ".join(f"{f}={v['split']}({v['windows']})" for f, v in P.info["files"].items()))
        if args.smoke and pn == plants[0]:
            selftest(P, T, out, dev)
        res = {}
        for m in methods:
            say(f"-- {m}")
            res[m] = run_replay(P, T, args.smoke) if m == "B1" else run_arx(P, T, out, args.smoke) if m == "B2" else run_neural(m, P, T, out, dev, args.smoke)
        if args.smoke:
            n0 = TRAINED[0]
            for m in methods:
                if m in ("C1", "A3"):
                    run_neural(m, P, T, out, dev, True)
            assert TRAINED[0] == n0, "rerun retrained finished models (done markers ignored)"
            for m in ("C1", "A3"):
                if m in res:
                    assert all(np.isfinite(v) for v in res[m]["seed_fit"]), f"{pn} {m}: non-finite validation FIT"
        runlog["plants"][pn] = {"data": P.info, "results": res}
        for m, r in res.items():
            say(f"   {pn} {m:3s} best cfg {r['best_cfg']} | val FIT per seed {[round(score(v), 2) for v in r['seed_fit']]}")
    wjson(out / f"run_{time.strftime('%Y%m%d-%H%M%S')}.json", runlog)
    say("SMOKE PASSED" if args.smoke else "PHASE 2 STAGE DONE")


if __name__ == "__main__":
    main()
