#!/usr/bin/env python
"""Phase 2 (D28): train on training splits, select on validation. Held-out files are never opened.

Methods: B1-B9 baselines, C1/A3 neural models and Full/A1-A5 inference variants.
  python scripts/02_train.py --smoke                      fast end-to-end check (writes results/smoke/)
  python scripts/02_train.py --plant hai --methods C1,A3  real run (clean git tree required)
"""
from __future__ import annotations
import argparse, hashlib, tempfile, zipfile, itertools, json, os, platform, random, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
from decoy import baselines, data, metrics, models, attacker, constraints  # noqa: E402

ORDER = ["C1", "A3", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
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
    manifest = ROOT / 'EXPORT_MANIFEST.json'
    if manifest.exists():
        for rel, expected in json.loads(manifest.read_text())['sha256'].items():
            if data.file_sha(ROOT/rel) != expected:
                raise SystemExit(f'STOP: exported snapshot changed: {rel}')
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        f = ROOT / "CODE_COMMIT"
        if not f.exists():
            raise SystemExit("STOP: not a git repo and no CODE_COMMIT file")
        if not manifest.exists() and not args.smoke:
            raise SystemExit("STOP: full training outside git requires a verified export manifest")
        commit, dirty = f.read_text().strip(), ""
    if dirty and not args.smoke:
        raise SystemExit("STOP: uncommitted changes; only --smoke may run on a dirty tree\n" + dirty)
    return {"commit": commit, "dirty": bool(dirty), "config_sha12": data.cfg_hash(), "argv": sys.argv, "smoke": args.smoke,
            "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "threads": torch.get_num_threads(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def guard_run(d, method, P, hp, T, seed=None):
    """Bind cached results and resumable checkpoints to the exact experiment."""
    if not hasattr(P, "_cache_digest"):
        h = hashlib.sha256()
        for x in (P.tr, P.va):
            h.update(str((x.shape, str(x.dtype))).encode())
            for row in x:
                h.update(np.ascontiguousarray(row).tobytes())
        P._cache_digest = h.hexdigest()
    sources = [ROOT / "scripts/02_train.py", ROOT / "configs/experiment.toml"] + sorted((ROOT / "src/decoy").glob("*.py"))
    h = hashlib.sha256()
    for path in sources:
        h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    expected = json.loads(json.dumps({"method": method, "hp": hp, "seed": seed,
        "training": T, "data": P._cache_digest, "files": P.info.get("files", {}) if hasattr(P,"info") else {}, "code": h.hexdigest(),
        "dimensions": [P.nc, P.ne, P.nr, P.W],
        "numpy": np.__version__, "torch": str(torch.__version__)}))
    manifest = d / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise RuntimeError(f"Experiment mismatch at {d}; use a new output folder. Existing results preserved.")
    elif d.exists() and any(d.iterdir()):
        raise RuntimeError(f"Unverified existing results at {d}; use a new output folder. Existing results preserved.")
    else:
        wjson(manifest, expected)


def train_one(method, P, hp, k, seed, root, T, dev):
    d = root / f"cfg{k:02d}_seed{seed}"
    guard_run(d, method, P, hp, T, seed)
    done = d / "done.json"
    if done.exists():
        r = json.loads(done.read_text())
        say(f"  {method} cfg{k:02d} seed{seed}: already done (val FIT {r['val_fit']:.2f})")
        return r
    seed_all(seed)
    t0 = time.time()
    m = models.build(method, P, hp).to(dev)
    if m.adversarial:
        st = models.train_gan(m, P.tr, hp['lr'], T['batch'], T['gan']['iters'], seed, d/'ckpt.pt', dev, T['gan']['ckpt_every'])
        epochs, best = None, None
    else:
        st = models.train(m, P.tr, P.va, hp["lr"], T["batch"], T.get("micro", {}).get(method, T["batch"]), T["max_epochs"], T["patience"], seed, d / "ckpt.pt", dev)
        epochs, best = st['epoch'], st['best']
    if method == 'B4': m.fit_noise(P.tr, dev)
    generated = models.generate(m, P.va, dev, seed)
    per, mean = metrics.fit_percent(P.va[:, P.W:, P.nc + P.ne:], generated[..., P.nc+P.ne:])
    kd = attacker.discriminate(P, generated, T['kdev'], dev) if method in ('B4','B5','B6','B7') else None
    models.save_atomic(m.state_dict(), d / "model.pt")
    TRAINED[0] += 1
    r = {"method": method, "plant": P.name, "cfg": k, "hp": hp, "seed": seed, "epochs": epochs,
         "iterations": st.get('step'), "resumed_from": st["resumed_from"], "best_val_loss": best, "val_fit": mean,
         "kdev": kd, "val_fit_per_tag": dict(zip(P.resp_names, per.tolist())), "n_params": models.n_params(m),
         "seconds": round(time.time() - t0, 1)}
    wjson(done, r)
    say(f"  {method} cfg{k:02d} seed{seed}: val FIT {mean:.2f}, K-dev {kd}, {r['seconds']} s")
    return r


def run_neural(method, P, T, out, dev, smoke):
    root = out / P.name / method
    seeds = ([T["seed_select"]] + list(T["seeds_final"]))[: (T["smoke"]["seeds"] if smoke else None)]
    if method != "A3":
        grid = expand(T["grid"][method], T["grid_size"])[: (T["smoke"]["grid"] if smoke else None)]
        res = [train_one(method, P, hp, k, T["seed_select"], root, T, dev) for k, hp in enumerate(grid)]
        best = min(range(len(res)), key=lambda k: (res[k]["kdev"]["score"], k)) if method in ("B4","B5","B6","B7") else max(range(len(res)), key=lambda k: (score(res[k]["val_fit"]), -k))
        hp = grid[best]
        wjson(root / "selection.json", {"criterion": "K-dev score, seed 0" if method in ("B4","B5","B6","B7") else "validation free-run FIT%, seed 0",
                                        "kdev": [r["kdev"] for r in res],
                                        "val_fit": [r["val_fit"] for r in res], "best": best, "hp": hp})
    else:
        sp = out / P.name / "C1" / "selection.json"
        if not sp.exists():
            raise SystemExit("STOP: A3 needs C1's selection; run C1 first")
        s = json.loads(sp.read_text())
        best, hp = s["best"], s["hp"]
        guard_run(sp.parent / f"cfg{best:02d}_seed{T['seed_select']}", "C1", P, hp, T, T["seed_select"])
    finals = [train_one(method, P, hp, best, sd, root, T, dev) for sd in seeds]
    return {"best_cfg": best, "hp": hp, "seed_fit": [r["val_fit"] for r in finals], "seed_kdev": [r["kdev"] for r in finals]}


def run_arx(P, T, out, smoke):
    root = out / P.name / "B2"
    grid = expand(T["grid"]["B2"], T["grid_size"])[: (T["smoke"]["grid"] if smoke else None)]
    res = []
    for k, hp in enumerate(grid):
        d = root / f"cfg{k:02d}"
        guard_run(d, "B2", P, hp, T)
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


def run_grey(P, T, out):
    root = out / P.name / 'B3'
    hp = {'delays': T['b3'][P.name+'_delays'], 'max_eval': T['b3']['max_eval']}
    guard_run(root, 'B3', P, hp, T)
    done=root/'done.json'
    if done.exists(): return json.loads(done.read_text())
    params=baselines.grey_fit(P, hp['delays'], hp['max_eval'])
    yh=baselines.grey_sim(params,P.va,P.W,P.nc+P.ne)
    per,mean=metrics.fit_percent(P.va[:,P.W:,P.nc+P.ne:],yh)
    wjson(root/'parameters.json',params)
    r={'best_cfg':None,'hp':hp,'seed_fit':[mean], 'val_fit_per_tag':dict(zip(P.resp_names,per.tolist()))}
    wjson(done,r); return r


def build_layers(P,T,out,dev,smoke,include_a3=True):
    run_grey(P,T,out)
    grey=json.loads((out/P.name/'B3'/'parameters.json').read_text())
    c=constraints.fit(P,T['c2'])
    if np.any(constraints.coverage(c,P.tr)>0):raise AssertionError('C3 activated on training windows')
    wjson(out/P.name/'layers'/'parameters.json',c)
    seeds=([T['seed_select']]+list(T['seeds_final']))[:T['smoke']['seeds'] if smoke else None]
    report={}
    sel=json.loads((out/P.name/'C1'/'selection.json').read_text());hp=sel['hp'];k=sel['best']
    for base in (['C1','A3'] if include_a3 else ['C1']):
        report[base]={}
        for seed in seeds:
            root=out/P.name/base/f'cfg{k:02d}_seed{seed}'
            guard_run(root,base,P,hp,T,seed)
            m=models.build(base,P,hp).to(dev);m.load_state_dict(torch.load(root/'model.pt',map_location=dev,weights_only=True))
            pred=models.predict(m,P.va,dev)
            per,mean=metrics.fit_percent(P.va[:,P.W:,P.nc+P.ne:],pred)
            recorded=json.loads((root/'done.json').read_text())['val_fit']
            if not np.isclose(mean,recorded,rtol=1e-5,atol=1e-4):raise AssertionError('Reloaded C1/A3 score mismatch')
            variants=['Full','A1','A2','A4','A5'] if base=='C1' else ['A3']
            results={}
            for v in variants:
                yh=constraints.apply(c,grey,P.va,pred,v)
                per,mean=metrics.fit_percent(P.va[:,P.W:,P.nc+P.ne:],yh)
                if not np.isfinite(yh).all():raise AssertionError(f'Non-finite {v} output')
                results[v]={'val_fit':mean,'per_tag':dict(zip(P.resp_names,per.tolist()))}
            report[base][str(seed)]=results
    wjson(out/P.name/'layers'/'validation.json',report)
    say(f'{P.name}: Full/A1/A2/A3/A4/A5 layers saved; training coverage and limits checks passed')


def selftest(P, T, out, dev):
    with tempfile.TemporaryDirectory(prefix="decoy-selftest-") as tmp:
        _selftest(P, T, Path(tmp), dev)


def _selftest(P, T, out, dev):
    y = np.random.default_rng(0).normal(size=(4, 10, 3))
    assert abs(metrics.fit_percent(y, y)[1] - 100.0) < 1e-9, "FIT(y, y) != 100"
    assert abs(metrics.fit_percent(y, np.broadcast_to(y.reshape(-1, 3).mean(0), y.shape))[1]) < 1e-9, "FIT(y, mean) != 0"

    def run(name, epochs):
        seed_all(0)
        m = models.build("C1", P, {"hidden": 16, "layers": 1}).to(dev)
        return models.train(m, P.tr, P.va, 1e-3, T["batch"], T["batch"], epochs, 99, 0, out / "_selftest" / name / "ckpt.pt", dev)
    a1, a2, b = run("resumed", 1), run("resumed", 2), run("straight", 2)
    assert a1["epoch"] == 1 and a2["epoch"] == 2 and a2["resumed_from"] == 1, "checkpoint resume failed"
    assert np.allclose(a2["history"], b["history"], rtol=1e-5, atol=1e-7), f"resumed {a2['history']} != uninterrupted {b['history']}"
    say("  selftest OK: FIT sanity; resumed training identical to uninterrupted training")


def gpu_preflight(cfg, plants, dev):
    import gc
    if dev.type != 'cuda': raise SystemExit('GPU preflight requires a CUDA GPU')
    for pn in plants:
        P=data.load_plant(cfg,pn,True)
        batch=cfg['train']['batch']
        tr=P.tr[np.arange(batch)%len(P.tr)]
        for method in ('B4','B5','B6','B7','B8','B9','C1'):
            hp={k:max(v) for k,v in cfg['train']['grid'][method].items()}
            say(f'GPU capacity preflight: {pn}/{method}, largest grid model, batch {batch}')
            try:
                m=models.build(method,P,hp).to(dev)
                if hasattr(m,'fit_scale'):m.fit_scale(P.tr)
                if m.adversarial:
                    opts=m.optimizers(hp['lr'])
                    m.step(torch.from_numpy(tr).to(dev),m.n_phases-1,opts,torch.Generator(device=dev).manual_seed(0))
                    del opts
                else:
                    opt=torch.optim.Adam(m.parameters(),lr=hp['lr']);opt.zero_grad()
                    micro=cfg['train'].get('micro',{}).get(method,batch)
                    for start in range(0,batch,micro):
                        loss=m.loss(torch.from_numpy(tr[start:start+micro]).to(dev))*len(tr[start:start+micro])/batch
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight loss')
                        loss.backward()
                    opt.step();del opt,loss
                pred=models.generate(m,P.va[:32],dev)
                assert np.isfinite(pred).all()
                del m,pred
                gc.collect();torch.cuda.empty_cache()
            except Exception as error:
                raise RuntimeError(f'GPU preflight failed at {pn}/{method}: {error}') from error
    say('GPU CAPACITY PREFLIGHT PASSED')


def export_kaggle(target, cfg):
    """Do not bundle raw data, held-out data, credentials, environments or previous outputs."""
    files = [ROOT/'scripts/02_train.py', ROOT/'configs/experiment.toml', ROOT/'spec/PROTOCOL.md',
             ROOT/'spec/decisions.md', ROOT/'README.md', ROOT/'spec/requirements-kaggle.txt']
    files += sorted((ROOT/'src/decoy').glob('*.py')) + sorted((ROOT/'tests').glob('*.py'))
    for plant in ('hai','qappd'):
        for part in ('train','val'):
            for name in cfg['train'][plant]['files_'+part]:
                path=ROOT/cfg['paths']['processed']/plant/(name+'.npz')
                with np.load(path,allow_pickle=False) as z: data.check_split(str(z['split']),part,name)
                files.append(path)
    contents={str(p.relative_to(ROOT)):p.read_bytes() for p in files}
    try: base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: base=(ROOT/'CODE_COMMIT').read_text().strip()
    digest=hashlib.sha256(b''.join(k.encode()+v for k,v in sorted(contents.items()))).hexdigest()
    contents['CODE_COMMIT']=(base+'+snapshot-'+digest[:12]).encode()
    manifest={'snapshot':digest,'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in contents.items()},
              'scope':'training and validation only; no raw or held-out data'}
    contents['EXPORT_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for k,v in contents.items():z.writestr('consistent-decoy/'+k,v)
    os.replace(tmp,target)
    say(f'Kaggle package saved: {target} ({target.stat().st_size/1024**2:.1f} MiB), snapshot {digest[:12]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-preflight", action="store_true", help="Check largest configured models at full batch size, without retaining weights")
    ap.add_argument("--export-kaggle", metavar="ZIP", help="Export a hash-verified train/validation-only snapshot and exit")
    ap.add_argument("--hours", type=float, default=None, help="Stop gracefully at a checkpoint after this wall-time budget")
    ap.add_argument("--plant", choices=["hai", "qappd", "all"], default="all")
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = data.load_cfg()
    if args.export_kaggle:
        export_kaggle(Path(args.export_kaggle), cfg); return
    if args.hours is not None:
        if args.hours <= 0: raise SystemExit('--hours must be positive')
        models.DEADLINE = time.monotonic() + args.hours * 3600
    T = dict(cfg["train"])
    if args.smoke:
        T["max_epochs"] = int(T["smoke"]["max_epochs"])
        T["gan"] = dict(T["gan"], iters=T["smoke"]["gan_iters"], ckpt_every=T["smoke"]["ckpt_every"])
        T["kdev"] = dict(T["kdev"], epochs=T["smoke"]["kdev_epochs"])
    asked = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in asked if m not in ORDER]
    if unknown:
        raise SystemExit(f"STOP: unknown methods {unknown}; Phase 2 supports {ORDER}")
    methods = [m for m in ORDER if m in asked]
    plants = ["hai", "qappd"] if args.plant == "all" else [args.plant]
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    smoke_dir = ROOT / "results" / "smoke"
    out = Path(args.out).resolve() if args.out else (smoke_dir if args.smoke else ROOT / "results" / "phase2")
    if args.smoke and out == smoke_dir and out.exists():
        shutil.rmtree(out)
    if args.gpu_preflight:
        gpu_preflight(cfg, plants, dev); return
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
            if models.expired(): raise TimeoutError("Time budget reached; completed runs preserved")
            say(f"-- {m}")
            res[m] = run_replay(P, T, args.smoke) if m == "B1" else run_arx(P, T, out, args.smoke) if m == "B2" else run_grey(P, T, out) if m == "B3" else run_neural(m, P, T, out, dev, args.smoke)
            if ('C1' in methods) and (m == 'A3' or (m == 'C1' and 'A3' not in methods)):
                build_layers(P, T, out, dev, args.smoke, include_a3='A3' in methods)
        if args.smoke:
            n0 = TRAINED[0]
            for m in methods:
                if m not in ("B1", "B2", "B3"):
                    run_neural(m, P, T, out, dev, True)
            assert TRAINED[0] == n0, "rerun retrained finished models (done markers ignored)"
            for m in [k for k in res if k not in ("B1", "B2", "B3")]:
                if m in res:
                    assert all(np.isfinite(v) for v in res[m]["seed_fit"]), f"{pn} {m}: non-finite validation FIT"
        runlog["plants"][pn] = {"data": P.info, "results": res}
        for m, r in res.items():
            say(f"   {pn} {m:3s} best cfg {r['best_cfg']} | val FIT per seed {[round(score(v), 2) for v in r['seed_fit']]}")
    wjson(out / f"run_{args.plant}_{time.strftime('%Y%m%d-%H%M%S')}.json", runlog)
    wjson(out / f"COMPLETE_{args.plant}.json", {"smoke": args.smoke, "provenance": prov, "methods": methods})
    say("SMOKE PASSED" if args.smoke else "PHASE 2 STAGE DONE")


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as error:
        say(f"PAUSED: {error}. Rerun the same command and output directory to resume.")
