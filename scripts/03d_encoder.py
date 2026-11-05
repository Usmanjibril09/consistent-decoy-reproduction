#!/usr/bin/env python
"""Stage 3d-0 (D44): Context-FID encoder. Official TS2Vec (commit b0088e1, vendored) trained per plant on real
training windows only (protocol E8): free-run horizon, response tags, Phase 2 z-scores. No held-out file and no
decoy output is opened. --smoke: tiny runs exercising every function; writes nothing."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time, tomllib
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoy import data, cfid  # noqa: E402


def stop(m):
    print("STOP:", m, flush=True); sys.exit(1)


def sh(*c):
    return subprocess.run(c, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = data.load_cfg()
    if data.cfg_hash() != "59a11b235108":
        stop(f"training config hash {data.cfg_hash()}")
    evp = ROOT / "configs" / "evaluation.toml"
    ev = tomllib.loads(evp.read_text())["context_fid"]
    if (ev["commit"] != cfid.COMMIT or int(ev["output_dims"]) != cfid.SETTINGS["output_dims"]
            or ev["train_on"] != "training_windows_real_only"):
        stop("configs/evaluation.toml [context_fid] does not match src/decoy/cfid.py")
    seed = int(ev["seed"])
    commit, dirty = sh("git", "rev-parse", "HEAD"), bool(sh("git", "status", "--porcelain"))
    out = ROOT / "results" / "phase3" / "encoder"
    work = out.with_name("encoder.partial")
    if not a.smoke:
        if dirty:
            stop("full run needs a clean, committed tree")
        if out.exists():
            stop(f"{out} exists; not overwriting")
        if work.exists():
            shutil.rmtree(work)
    print(f"Stage 3d-0 encoder | commit {commit[:7]}{' (dirty)' if dirty else ''} | TS2Vec {cfid.COMMIT[:7]} | "
          f"torch {torch.__version__} numpy {np.__version__} | {'SMOKE' if a.smoke else 'FULL'}", flush=True)
    if a.smoke:
        rs = np.random.RandomState(0); x = rs.randn(1000, 320)
        f0, f1 = cfid.fid(x, x), cfid.fid(x, x + 1.0)
        if abs(f0) > 1e-4 or abs(f1 - 320.0) > 1e-4:
            stop(f"FID self-test: fid(x,x)={f0}, fid(x,x+1)={f1} (expected 0 and 320)")
        print(f"FID self-test OK: fid(x,x) {f0:.2e}, fid(x,x+1) {f1:.6f} (expected 320)", flush=True)
    summary = {}
    for pn in ("hai", "qappd"):
        P = data.load_plant(cfg, pn, smoke=a.smoke)
        s = P.nc + P.ne
        if P.tr.shape[2] != s + P.nr or len(P.resp_names) != P.nr:
            stop(f"{pn}: column layout differs from [command, exogenous, response]")
        tr = np.ascontiguousarray(P.tr[:, P.W:, s:]); va = np.ascontiguousarray(P.va[:, P.W:, s:])
        print(f"== {pn}: train {tr.shape} val {va.shape} (horizon, response tags, Phase 2 z-scores)", flush=True)

        def train(n_iters):
            cfid.seed_all(seed)
            m = cfid.new_encoder(P.nr)
            log = m.fit(tr, n_iters=n_iters, verbose=False)
            return m, log

        if a.smoke:
            (m1, _), (m2, _) = train(3), train(3)
            v = va[:64]
            e1, e2 = cfid.embed(m1, v), cfid.embed(m2, v)
            if e1.shape != (len(v), 320) or not np.isfinite(e1).all():
                stop(f"{pn}: embedding shape {e1.shape} or non-finite")
            if not np.array_equal(e1, e2):
                stop(f"{pn}: two seeded runs differ")
            with tempfile.TemporaryDirectory() as td:
                wp = Path(td) / "encoder.pt"; m1.save(str(wp))
                if not np.array_equal(cfid.embed(cfid.load_encoder(wp, P.nr), v), e1):
                    stop(f"{pn}: reloaded encoder differs")
            noisy = (v + np.random.RandomState(seed).normal(0, 0.25, v.shape)).astype(np.float32)
            if not np.isfinite(cfid.embed(m1, noisy)).all():
                stop(f"{pn}: noisy embedding non-finite")
            print(f"   smoke OK: embedding {e1.shape}, two seeded runs identical, save/reload identical", flush=True)
            continue
        t0 = time.time(); m, log = train(None); secs = time.time() - t0
        d = work / pn; d.mkdir(parents=True, exist_ok=True)
        wp = d / "encoder.pt"; m.save(str(wp))
        etr, eva = cfid.embed(m, tr), cfid.embed(m, va)
        if not np.array_equal(cfid.embed(cfid.load_encoder(wp, P.nr), va), eva):
            stop(f"{pn}: reloaded encoder differs")
        noisy = (va + np.random.RandomState(seed).normal(0, 0.25, va.shape)).astype(np.float32)
        san = {"train_vs_val": cfid.fid(etr, eva),
               "val_even_vs_odd": cfid.fid(eva[0::2], eva[1::2]),
               "val_vs_val_plus_0.25sd_noise": cfid.fid(eva, cfid.embed(m, noisy))}
        if not all(np.isfinite(list(san.values()))):
            stop(f"{pn}: non-finite sanity FID {san}")
        rec = {"plant": pn, "weights_sha256": sha(wp), "input_dims": int(P.nr),
               "resp_names": [str(r) for r in P.resp_names], "settings": cfid.SETTINGS, "seed": seed,
               "n_iters": int(m.n_iters), "n_epochs": int(m.n_epochs), "train_windows": int(len(tr)),
               "val_windows": int(len(va)), "loss_per_epoch": [float(x) for x in log], "sanity_fid": san,
               "seconds": round(secs, 1)}
        (d / "encoder.json").write_text(json.dumps(rec, indent=1))
        summary[pn] = {k: rec[k] for k in ("weights_sha256", "n_iters", "n_epochs", "sanity_fid", "seconds")}
        print(f"   trained {m.n_iters} iterations ({m.n_epochs} epochs, {secs:.0f} s), final loss {log[-1]:.4f}", flush=True)
        print(f"   weights sha256 {rec['weights_sha256']}", flush=True)
        print("   sanity FID: " + " | ".join(f"{k} {v:.4f}" for k, v in san.items()), flush=True)
    if a.smoke:
        print("SMOKE ENCODER PASSED (nothing written)"); return
    prov = {"commit": commit, "ts2vec_commit": cfid.COMMIT, "evaluation_toml_sha256": sha(evp),
            "experiment_cfg_hash": data.cfg_hash(), "torch": torch.__version__, "numpy": np.__version__,
            "threads": torch.get_num_threads()}
    (work / "summary.json").write_text(json.dumps({"provenance": prov, "plants": summary}, indent=1))
    os.replace(work, out)
    print(f"ENCODER DONE -> results/phase3/encoder (summary.json sha256 {sha(out / 'summary.json')})")


if __name__ == "__main__":
    main()
