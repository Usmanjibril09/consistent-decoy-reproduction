#!/usr/bin/env python3
"""EPIC arm (D48/D49): full generation on pinned upstream SimProcess 7533f59; outputs hashed and frozen
before any classifier sees them.
(1) VRNN: extract_noise.py unmodified (must equal smoke 2), train.py unmodified, 41 epochs; it must write
    vrnn_state_dict_41.pth (the file generate.py loads); first epochs' losses must equal smoke 2.
(2) 10 generations, generator.sh unmodified for Pandapower and Mosaik, EPIC_SEED 0-4 'static' ('dynamic_'
    prefix stripped, D49) and 100-104 'dynamic', 4 in parallel, each in its own git-archive extract.
    Static seed 0 must equal smoke 1 seed 0 (values; wall-clock columns exempt); noisy outputs must change
    with the seed.
(3) generate.py unmodified on each generation's noise-free Mosaik / Panda_clean files (power_reactive ->
    power_effective, D49), EPIC_SEED = EPIC_TORCH_SEED = generation seed; outputs renamed '+autoencoder'.
--smoke: 300 samples (generator.sh copies), 2 epochs (train.py copy; last checkpoint copied to the
epoch-41 name, as smoke 2), seeds 0 and 100; writes nothing to the repo.
Full mode: clean tree required; finished generations are reused after hash verification."""
import argparse, json, math, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import epic_smoke as es
import epic_smoke2 as es2
from epic_smoke import COMMIT, ROOT, UP, GENS, CLOCK, say, stop, hms, sha, compare

VR = es2.VR
SCRIPT = Path(__file__).resolve()
PREFIX = "dynamic_"
FULL_JOBS = ([("static_s0", 0, "static"), ("dynamic_s100", 100, "dynamic")]
             + [(f"static_s{s}", s, "static") for s in range(1, 5)]
             + [(f"dynamic_s{s}", s, "dynamic") for s in range(101, 105)])
SMOKE_JOBS = FULL_JOBS[:2]
PAR = 4
NOISE_FREE = {"Pandapower": ["dynamic_Panda_clean.csv", "dynamic_Panda_denoised.csv"], "Mosaik": ["dynamic_Mosaik.csv"]}

def workdir(smoke): return Path.home() / "epic_work" / ("generation_smoke" if smoke else "generation")
def ident(seed, kind, smoke): return {"script": sha(SCRIPT), "upstream": COMMIT, "seed": seed, "kind": kind, "smoke": smoke}

def out_name(e, kind):
    if not e.startswith(PREFIX): stop(f"{e}: expected prefix {PREFIX}")
    return e[len(PREFIX):] if kind == "static" else e

def expected():
    r = {}
    for g in GENS:
        exp, quiet, pre = es.parse_gen(UP / "Generators" / g / "generator.sh")
        if pre != [PREFIX]: stop(f"{g}: generator.sh prefixes {pre}, expected ['{PREFIX}']")
        if not exp: stop(f"{g}: no outputs parsed")
        r[g] = (exp, quiet)
    return r

def done_ok(W, want):
    f = W / "DONE.json"
    if not f.is_file(): return None
    d = json.loads(f.read_text())
    if d.get("ident") != want: return None
    if any(not (W / k).is_file() or sha(W / k) != v for k, v in d["files"].items()): return None
    return d

def shorten(gsh):
    s = gsh.read_text()
    rx = re.compile(r'^([ \t]*generate_\w+[ \t]+"dynamic_"[ \t].*[ \t])3000([ \t]*)$', re.M)
    n = len(rx.findall(s))
    if n != 1:
        stop(f"{gsh.parent.name}/generator.sh: {n} generator calls ending in 3000; lines: "
             f"{[l for l in s.splitlines() if '3000' in l]}")
    gsh.write_text(rx.sub(r"\g<1>300\g<2>", s))

# ---------------- one generation (runs as a separate process)
def worker(label, seed, kind, smoke):
    W = workdir(smoke) / label
    if W.exists(): shutil.rmtree(W)
    dest = W / "src"; dest.mkdir(parents=True)
    subprocess.run(f"git -C '{UP}' archive --format=tar {COMMIT} | tar -x -C '{dest}'", shell=True, check=True)
    ds = dest / "Discriminator" / "dataset"; ds.mkdir(parents=True, exist_ok=True)
    for i, p in es.epic_sources().items(): shutil.copyfile(p, ds / f"EPIC{i}.csv")
    if smoke:
        for g in GENS: shorten(dest / "Generators" / g / "generator.sh")
    es.WORK = W
    exp = expected()
    files, secs = {}, {}
    for g in GENS:
        new, secs[g] = es.run_gen(dest, g, seed, label, exp[g][0])
        for e in exp[g][0]:
            hit = [k for k in new if Path(k).name == e]
            if len(hit) != 1: stop(f"{label} {g}: {e} found {len(hit)} times among new files")
            d = W / "out" / g / out_name(e, kind); d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest / hit[0], d); d.chmod(0o444)
            files[str(d.relative_to(W))] = sha(d)
    (W / "DONE.json").write_text(json.dumps({"ident": ident(seed, kind, smoke), "files": files,
                                             "seconds": {g: round(v, 1) for g, v in secs.items()}}, indent=1))
    say(f"   [{label}] DONE: {len(files)} files frozen in {hms(sum(secs.values()))}")

# ---------------- checks against the smoke references
def smoke1_refs(exp):
    s1 = json.loads((ROOT / "spec" / "epic_smoke.json").read_text())["seed0_outputs"]
    S = Path.home() / "epic_work" / "smoke" / "seed0_a"
    refs = {}
    for g in GENS:
        for e in exp[g][0]:
            hit = [k for k in s1[g] if Path(k).name == e]
            if len(hit) != 1: stop(f"smoke 1 record: {e} found {len(hit)} times")
            ref = S / hit[0]
            if not ref.is_file() or sha(ref) != s1[g][hit[0]]: stop(f"smoke 1 reference {ref} missing or changed")
            refs[(g, e)] = ref
    say(f"smoke 1 seed-0 references: {len(refs)} files present, sha256 as recorded")
    return refs

def recheck_seed0(W, refs):
    clock = {}
    for (g, e), ref in refs.items():
        d = compare(ref, W / "out" / g / out_name(e, "static"))
        if d and any(not CLOCK.search(x) for x in d):
            stop(f"static seed 0 differs from smoke 1 seed 0: {e} columns {d}")
        if d: clock[e] = d
    say(f"[check] static seed 0 = smoke 1 seed 0 on all {len(refs)} files (only wall-clock columns differ in {len(clock)})")
    return clock

def check_seeds(W0, jobs, exp):
    same_quiet = 0
    for lab, s, k in jobs[1:]:
        for g in GENS:
            for e in exp[g][0]:
                d = compare(W0 / "static_s0" / "out" / g / out_name(e, "static"), W0 / lab / "out" / g / out_name(e, k))
                changed = bool(d) and any(not CLOCK.search(x) for x in d)
                if e in exp[g][1]: same_quiet += not changed
                elif not changed: stop(f"{lab} {g} {e}: same values as seed 0 (noisy output ignores the seed)")
    say(f"[check] every noisy output differs from seed 0 in all {len(jobs) - 1} other generations | "
        f"noise-free outputs unchanged across seeds: {same_quiet}")

# ---------------- VRNN
def vrnn_train(smoke, src):
    W = workdir(smoke) / "vrnn"; es2.WORK = W
    want = {"script": sha(SCRIPT), "upstream": COMMIT, "smoke": smoke}
    d = done_ok(W, want)
    if d:
        say(f"[vrnn] already trained (reused), weights hash-verified | epochs saved {d['info']['epochs_saved']}")
        return W / "src" / VR, d["info"]
    if W.exists(): shutil.rmtree(W)
    dest = W / "src"; dest.mkdir(parents=True)
    subprocess.run(f"git -C '{UP}' archive --format=tar {COMMIT} '{VR}' Discriminator/tools | tar -x -C '{dest}'",
                   shell=True, check=True)
    vr = dest / VR
    pre = list(vr.rglob("*.pth"))
    if pre: stop(f"upstream extract already contains weights: {pre}")
    es2.check_imports(vr)
    ne = es2.literal(vr / "train.py", "n_epochs")
    if ne != 41: stop(f"train.py n_epochs {ne}, expected 41")
    ref = json.loads((ROOT / "spec" / "epic_smoke2.json").read_text())["noise_files_sha256"]
    raw = vr / "raw_data"; raw.mkdir(); up = dest / "noise_upstream"; up.mkdir()
    for i, p in src.items():
        shutil.copyfile(p, raw / f"EPIC{i}.csv")
        es2.run([sys.executable, str(dest / "Discriminator/tools/extract_noise.py"), "-i", str(raw / f"EPIC{i}.csv"),
                 "-o", str(up / f"EPIC{i}_noise.csv")], raw, es2.env(0), f"extract_EPIC{i}")
        df = pd.read_csv(up / f"EPIC{i}_noise.csv").rename(columns={"power_reactive_noise": "power_effective_noise"})
        df.to_csv(raw / f"EPIC{i}_noise.csv", index=False)
        if sha(raw / f"EPIC{i}_noise.csv") != ref[f"EPIC{i}_noise.csv"]: stop(f"EPIC{i} noise file differs from smoke 2")
    say("[vrnn] noise extraction: 8 files identical to smoke 2 (sha256)")
    script = "train.py"
    if smoke:
        s = (vr / "train.py").read_text(); old = "    n_epochs = 41\n"
        if s.count(old) != 1: stop(f"train.py: '{old.strip()}' found {s.count(old)} times")
        (vr / "train_smoke.py").write_text(s.replace(old, "    n_epochs = 2\n")); script = "train_smoke.py"
    say(f"[vrnn] training {2 if smoke else 41} epochs (smoke 2: ~26 s/epoch); progress every 30 s")
    txt, secs = es2.run([sys.executable, script], vr, es2.env(0), "train")
    saves = vr / "saves"
    ep = sorted(int(m.group(1)) for f in saves.glob("*.pth") if (m := re.fullmatch(r"vrnn_state_dict_(\d+)\.pth", f.name)))
    say(f"[vrnn] trained in {hms(secs)} | checkpoints written for epochs {ep}")
    w41 = saves / "vrnn_state_dict_41.pth"
    if smoke:
        if not ep: stop("no checkpoint written")
        shutil.copyfile(saves / f"vrnn_state_dict_{ep[-1]}.pth", w41)
        say(f"[vrnn] smoke only: epoch-{ep[-1]} weights copied to vrnn_state_dict_41.pth (full run requires the real file)")
    elif 41 not in ep:
        stop(f"train.py did not write vrnn_state_dict_41.pth (the file generate.py loads); epochs saved {ep}")
    import torch
    torch.load(w41, map_location="cpu")
    la = pd.read_csv(saves / "losses_all.csv")
    if not np.isfinite(la[["train_loss", "kld_loss", "nll_loss"]].to_numpy(float)).all(): stop("non-finite loss")
    lb = pd.read_csv(Path.home() / "epic_work" / "smoke2" / "A" / VR / "saves" / "losses_all.csv")
    if len(la) < len(lb) or not la.iloc[:len(lb)].reset_index(drop=True).equals(lb):
        stop("first epochs' losses differ from smoke 2 (same data, same seed): training not reproducible")
    say(f"[vrnn] losses finite; first {len(lb)} loss rows identical to smoke 2 | last row {la.iloc[-1].to_dict()}")
    info = {"epochs_saved": ep, "seconds": round(secs, 1), "last_losses": la.iloc[-1].to_dict(),
            "weights_sha256": sha(w41), "smoke_copied_checkpoint": bool(smoke)}
    files = {str(f.relative_to(W)): sha(f) for f in (w41, saves / "losses_all.csv")}
    (W / "DONE.json").write_text(json.dumps({"ident": want, "files": files, "info": info}, indent=1))
    return vr, info

def vrnn_generate(smoke, jobs, vr):
    W0 = workdir(smoke); es2.WORK = W0 / "vrnn"
    gen_cols = es2.literal(vr / "generate.py", "measurements_to_keep")
    raw, outdir = vr / "raw_data", vr / "saves" / "output"
    res = {}
    for n, (lab, s, k) in enumerate(jobs):
        o = W0 / lab / "out"
        clean = o / "Pandapower" / out_name("dynamic_Panda_clean.csv", k)
        den = o / "Pandapower" / out_name("dynamic_Panda_denoised.csv", k)
        a = pd.read_csv(clean).drop(columns=["timestamp"], errors="ignore")
        b = pd.read_csv(den).drop(columns=["timestamp"], errors="ignore")
        if not a.equals(b): stop(f"{lab}: Panda_clean and Panda_denoised differ; upstream's Panda.csv ambiguous")
        inputs = {"Mosaik.csv": o / "Mosaik" / out_name("dynamic_Mosaik.csv", k), "Panda.csv": clean}
        for name, p in inputs.items():
            df = pd.read_csv(p)
            if "power_reactive" not in df.columns or "power_effective" in df.columns:
                stop(f"{p.name}: expected power_reactive and no power_effective, has {df.columns.tolist()}")
            df.rename(columns={"power_reactive": "power_effective"}).to_csv(raw / name, index=False)
        got = []
        for r in range(2 if n == 0 else 1):
            if outdir.exists():
                for f in outdir.glob("*_autoencoderEPICnoise.csv"): f.unlink()
            es2.run([sys.executable, "generate.py"], vr, es2.env(s, s), f"generate_{lab}_{r}")
            cur = {}
            for name in inputs:
                f = outdir / f"{name[:-4]}_autoencoderEPICnoise.csv"
                if not f.is_file(): stop(f"generate {lab}: {f.name} not written")
                df = pd.read_csv(f); n_in = len(pd.read_csv(raw / name))
                if list(df.columns) != gen_cols or len(df) != n_in - 1 or not np.isfinite(df.to_numpy(float)).all():
                    stop(f"generate {lab} {f.name}: columns {df.columns.tolist()} rows {len(df)} (expected {n_in - 1}) or non-finite")
                cur[name] = (df, f.read_bytes())
            got.append(cur)
        if len(got) == 2 and any(not got[0][x][0].equals(got[1][x][0]) for x in inputs):
            stop(f"generate.py not repeatable under the same seed ({lab})")
        dst = o / "VRNN"; dst.mkdir(parents=True, exist_ok=True)
        pre = "" if k == "static" else PREFIX
        for name in inputs:
            d = dst / f"{pre}{name[:-4]}+autoencoder.csv"
            if d.exists(): d.unlink()
            d.write_bytes(got[0][name][1]); d.chmod(0o444)
            res[str(d.relative_to(W0))] = sha(d)
        say(f"[vrnn] {lab}: Mosaik/Panda +autoencoder written (seed {s})" + (" | same seed twice identical" if n == 0 else ""))
    return res

# ---------------- parallel generations
def generations(smoke, jobs, refs):
    W0 = workdir(smoke)
    todo, done, run, clock = [], {}, {}, {}
    for lab, s, k in jobs:
        d = done_ok(W0 / lab, ident(s, k, smoke))
        if d:
            done[lab] = d; say(f"[gen] {lab}: already done, {len(d['files'])} files hash-verified (reused)")
            if lab == "static_s0" and refs: clock = recheck_seed0(W0 / lab, refs)
        else:
            todo.append((lab, s, k))
    def kill_all():
        for p in run.values():
            try: os.killpg(p.pid, signal.SIGTERM)
            except OSError: pass
        for p in run.values():
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid, signal.SIGKILL)
                except OSError: pass
    t0 = last = time.time()
    try:
        while todo or run:
            while todo and len(run) < PAR:
                lab, s, k = todo.pop(0)
                cmd = [sys.executable, str(SCRIPT), "--worker", lab, str(s), k] + (["--smoke"] if smoke else [])
                run[lab] = subprocess.Popen(cmd, start_new_session=True)
                say(f"[gen] started {lab} (EPIC_SEED={s})")
            time.sleep(5)
            for lab, p in list(run.items()):
                rc = p.poll()
                if rc is None: continue
                del run[lab]
                if rc != 0: stop(f"generation {lab} failed (exit {rc}); all other jobs stopped. Logs in {W0 / lab}")
                s, k = next((s, k) for l, s, k in jobs if l == lab)
                d = done_ok(W0 / lab, ident(s, k, smoke))
                if d is None: stop(f"{lab} exited 0 but DONE.json is missing or its hashes differ")
                done[lab] = d
                if lab == "static_s0" and refs: clock = recheck_seed0(W0 / lab, refs)
                say(f"[gen] {lab} finished | {len(done)}/{len(jobs)} done | elapsed {hms(time.time() - t0)}")
            if time.time() - last >= 300:
                secs = [sum(d["seconds"].values()) for d in done.values()]
                eta = f" | ~{hms(np.mean(secs) * math.ceil((len(todo) + len(run)) / PAR))} left (rough)" if secs else ""
                say(f"[gen] {len(done)}/{len(jobs)} done | running {sorted(run)} | elapsed {hms(time.time() - t0)}{eta}")
                last = time.time()
    except KeyboardInterrupt:
        stop("interrupted; finished generations are kept and reused on rerun")
    finally:
        kill_all()
    return done, clock

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", nargs=3)
    a = ap.parse_args()
    if a.worker:
        lab, s, k = a.worker; worker(lab, int(s), k, a.smoke); return
    smoke = a.smoke
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout.strip()
    if not smoke:
        if git("status", "--porcelain"): stop("full run needs a clean tree")
        if "EPIC arm generation script" not in git("log", "-1", "--format=%s"): stop("HEAD is not the committed generation script")
    import torch
    say(f"EPIC generation | {'SMOKE' if smoke else 'FULL'} | repo {git('rev-parse', '--short', 'HEAD')} | upstream {COMMIT[:7]} | "
        f"script {sha(SCRIPT)[:12]} | torch {torch.__version__} | numpy {np.__version__} | {PAR} parallel")
    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < 5: stop(f"only {free:.1f} GB free")
    es.check_shim(); es2.check_torch_shim()
    src = es.epic_sources()
    s1 = json.loads((ROOT / "spec" / "epic_smoke.json").read_text())
    for i, p in src.items():
        if s1["epic_copies"][f"EPIC{i}.csv"]["sha256"] != sha(p): stop(f"EPIC{i} source changed since smoke 1")
    say("EPIC sources unchanged since smoke 1 (8 sha256 match)")
    exp = expected()
    for g, names in NOISE_FREE.items():
        miss = [x for x in names if x not in exp[g][1]]
        if miss: stop(f"{g}: {miss} not among generator.sh noise-free outputs {exp[g][1]}")
    jobs = SMOKE_JOBS if smoke else FULL_JOBS
    W0 = workdir(smoke)
    if smoke and W0.exists(): shutil.rmtree(W0)
    W0.mkdir(parents=True, exist_ok=True)
    refs = None if smoke else smoke1_refs(exp)
    t0 = time.time()
    vr, vinfo = vrnn_train(smoke, src)
    gdone, clock = generations(smoke, jobs, refs)
    check_seeds(W0, jobs, exp)
    vfiles = vrnn_generate(smoke, jobs, vr)
    files = {}
    for lab, _, _ in jobs:
        for f in sorted((W0 / lab / "out").rglob("*")):
            if f.is_file(): files[str(f.relative_to(W0))] = sha(f)
    for lab, _, _ in jobs:
        for kk, v in gdone[lab]["files"].items():
            if files.get(f"{lab}/{kk}") != v: stop(f"{lab}/{kk} changed after freezing")
    for kk, v in vfiles.items():
        if files.get(kk) != v: stop(f"{kk} changed after freezing")
    size = sum((W0 / k).stat().st_size for k in files) / 1e6
    if smoke:
        say(f"EPIC GENERATION SMOKE PASSED | {len(files)} files ({size:.1f} MB) | {hms(time.time() - t0)} | nothing written to the repo")
        return
    man = "".join(f"{v}  {k}\n" for k, v in sorted(files.items()))
    (ROOT / "spec" / "epic_generation_manifest.sha256").write_text(man)
    out = {"upstream_commit": COMMIT, "repo_head": git("rev-parse", "HEAD"), "script_sha256": sha(SCRIPT),
           "requirements_epic_sha256": sha(ROOT / "spec" / "requirements-epic.txt"), "location": str(W0),
           "jobs": [{"label": l, "seed": s, "kind": k, "seconds": gdone[l]["seconds"]} for l, s, k in jobs],
           "static_rule": "generator.sh unmodified; 'dynamic_' prefix stripped from file names (D49)",
           "seed0_equals_smoke1": True, "seed0_clock_only_differences": clock,
           "vrnn": vinfo, "vrnn_generate_rule": "generate.py unmodified on noise-free Mosaik / Panda_clean; "
           "power_reactive->power_effective; EPIC_SEED=EPIC_TORCH_SEED=generation seed; renamed +autoencoder",
           "n_files": len(files), "megabytes": round(size, 1),
           "manifest_sha256": sha(ROOT / "spec" / "epic_generation_manifest.sha256"),
           "wall_seconds": round(time.time() - t0, 1)}
    (ROOT / "spec" / "epic_generation.json").write_text(json.dumps(out, indent=1) + "\n")
    say(f"EPIC GENERATION DONE | {len(files)} files ({size:.1f} MB) frozen in {W0} | {hms(time.time() - t0)}")

if __name__ == "__main__":
    main()
