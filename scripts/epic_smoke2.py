#!/usr/bin/env python3
"""EPIC arm, D48(h) smoke part 2: VRNN autoencoder pipeline (D49) on pinned upstream SimProcess 7533f59.
1. Discriminator/tools/extract_noise.py (unmodified) on EPIC1-8; adapter renames power_reactive_noise ->
   power_effective_noise (train.py expects power_effective; D49 'adapted by us').
2. train.py with n_epochs 41 -> 2 (smoke-only copy, one asserted line), run twice in separate extracts:
   weights and losses must be identical.
3. generate.py (unmodified) with the epoch-1 weights copied to the file name it loads (smoke only), on the
   noise-free Mosaik/Panda outputs of smoke 1 seed 0 (power_reactive -> power_effective): outputs exist,
   10 columns, rows = input - 1, finite; same seeds identical; effect of a different seed reported.
Writes spec/epic_smoke2.json only on PASS. No classifier is trained here."""
import ast, importlib.util, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from epic_smoke import COMMIT, ROOT, UP, env_for, epic_sources, check_shim, hms, say, sha, stop

VR = "Generators/VariationalRecurrentNeuralNetwork"
WORK = Path.home() / "epic_work" / "smoke2"
S1 = Path.home() / "epic_work" / "smoke" / "seed0_a" / "Generators"

def env(seed, tseed=None):
    e = env_for(seed); e.pop("EPIC_TORCH_SEED", None)
    if tseed is not None: e["EPIC_TORCH_SEED"] = str(tseed)
    return e

def literal(path, name):
    vals = [ast.literal_eval(n.value) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(vals) != 1: stop(f"{path.name}: {len(vals)} assignments to {name}")
    return vals[0]

def check_imports(d):
    local = {p.stem for p in d.glob("*.py")}
    miss = {}
    for f in d.glob("*.py"):
        for node in ast.parse(f.read_text()).body:
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                   [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            for m in mods:
                top = m.split(".")[0]
                if top in sys.stdlib_module_names or top in local: continue
                if importlib.util.find_spec(top) is None: miss.setdefault(top, []).append(f.name)
    if miss: stop(f"VRNN imports not installed: {miss}")
    say("VRNN imports OK")

def check_torch_shim():
    code = "import torch; print(repr(torch.rand(1).item()))"
    outs = [subprocess.run([sys.executable, "-c", code], env=env(0, 0), capture_output=True, text=True) for _ in range(2)]
    if any(o.returncode for o in outs): stop("torch shim check failed:\n" + outs[0].stderr)
    import torch
    torch.manual_seed(0); want = repr(torch.rand(1).item())
    got = [o.stdout.strip() for o in outs]
    if got != [want, want]: stop(f"torch shim does not seed: {got} expected {want}")
    say(f"torch shim OK: {want} (= torch seed 0, twice)")

def run(cmd, cwd, e, label):
    log = WORK / f"{label}.log"
    t0 = time.time(); last = t0
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=lf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(2)
            if time.time() - last >= 30:
                lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
                ep = [l for l in lines if l.startswith("====>")]
                say(f"   [{label}] {hms(time.time() - t0)} | {ep[-1] if ep else (lines[-1][:90] if lines else '')}")
                last = time.time()
    txt = log.read_text(errors="replace")
    if p.returncode or "Traceback" in txt:
        stop(f"{label}: exit {p.returncode} (log {log}):\n" + "\n".join(txt.splitlines()[-25:]))
    return txt, time.time() - t0

def main():
    import torch
    say(f"EPIC smoke 2 | upstream {COMMIT[:7]} | torch {torch.__version__} threads {torch.get_num_threads()} | numpy {np.__version__}")
    check_shim()
    check_torch_shim()
    src = epic_sources()
    s1 = json.loads((ROOT / "spec" / "epic_smoke.json").read_text())
    for i, p in src.items():
        if s1["epic_copies"][f"EPIC{i}.csv"]["sha256"] != sha(p): stop(f"EPIC{i} source changed since smoke 1")
    say("EPIC sources unchanged since smoke 1 (8 sha256 match)")
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    dests = {}
    for lab in ("A", "B"):
        d = WORK / lab; d.mkdir()
        subprocess.run(f"git -C '{UP}' archive --format=tar {COMMIT} '{VR}' Discriminator/tools | tar -x -C '{d}'",
                       shell=True, check=True)
        dests[lab] = d
    vr = dests["A"] / VR
    check_imports(vr)
    sig = literal(dests["A"] / "Discriminator/tools/extract_noise.py", "signal_types")
    noise_src = [c for cols in sig.values() for c in cols]
    train_cols = literal(vr / "train.py", "measurements_to_keep")
    gen_cols = literal(vr / "generate.py", "measurements_to_keep")
    win, bs, ne = (literal(vr / "train.py", k) for k in ("window_size", "batch_size", "n_epochs"))
    say(f"extract_noise columns {noise_src}")
    say(f"train.py: window {win}, batch {bs}, epochs {ne}, columns {train_cols}")
    say(f"generate.py columns {gen_cols}")
    say(f"EPIC1 header: {pd.read_csv(src[1], nrows=1).columns.tolist()}")
    for i, p in src.items():
        cols = pd.read_csv(p, nrows=1).columns
        miss = [c for c in noise_src if c not in cols]
        if miss: stop(f"EPIC{i}: columns extract_noise.py would silently skip: {miss}")
        if "power_effective" in cols: stop(f"EPIC{i} already has power_effective; mapping ambiguous")

    # 1. noise extraction (deterministic; both extracts must agree)
    noise_sha, rows = {}, {}
    for lab, d in dests.items():
        raw = d / VR / "raw_data"; raw.mkdir()
        up = d / "noise_upstream"; up.mkdir()
        for i, p in src.items():
            shutil.copyfile(p, raw / f"EPIC{i}.csv")
            txt, _ = run([sys.executable, str(d / "Discriminator/tools/extract_noise.py"), "-i", str(raw / f"EPIC{i}.csv"),
                          "-o", str(up / f"EPIC{i}_noise.csv")], raw, env(0), f"{lab}_extract_EPIC{i}")
            if "Warning" in txt: say(f"   [{lab} EPIC{i}] " + "; ".join(l for l in txt.splitlines() if "Warning" in l))
            df = pd.read_csv(up / f"EPIC{i}_noise.csv").rename(columns={"power_reactive_noise": "power_effective_noise"})
            miss = [c for c in train_cols if c not in df.columns]
            if miss: stop(f"EPIC{i} noise file lacks {miss} (has {df.columns.tolist()})")
            v = df[train_cols].to_numpy(float)
            if not np.isfinite(v).all(): stop(f"EPIC{i}: {int((~np.isfinite(v)).sum())} non-finite noise values")
            df.to_csv(raw / f"EPIC{i}_noise.csv", index=False)
            noise_sha.setdefault(lab, {})[f"EPIC{i}_noise.csv"] = sha(raw / f"EPIC{i}_noise.csv")
            rows[i] = len(df)
    if noise_sha["A"] != noise_sha["B"]: stop("noise extraction differs between extracts")
    n = sum(r - win for r in rows.values()); n_test = math.ceil(0.1 * n); n_train = n - n_test
    say(f"noise extraction OK (8 files, identical in both extracts) | sequences {n}: train {n_train}, test {n_test} | "
        f"last batch sizes {n_train % bs or bs} / {n_test % bs or bs}")
    if n_train % bs == 1 or n_test % bs == 1:
        stop("a final batch of 1 sequence: data.squeeze() in train.py would silently drop the batch dimension")

    # 2. training, 2 epochs, twice
    old = f"    n_epochs = {ne}\n"
    times = {}
    for lab, d in dests.items():
        s = (d / VR / "train.py").read_text()
        if s.count(old) != 1: stop(f"train.py: '{old.strip()}' found {s.count(old)} times")
        (d / VR / "train_smoke2ep.py").write_text(s.replace(old, "    n_epochs = 2\n"))
        txt, times[lab] = run([sys.executable, "train_smoke2ep.py"], d / VR, env(0), f"{lab}_train")
        for f in ("saves/vrnn_state_dict_1.pth", "saves/losses_all.csv"):
            if not (d / VR / f).is_file(): stop(f"{lab}: {f} not written")
        say(f"   [{lab}_train] done in {hms(times[lab])} | " + " | ".join(l for l in txt.splitlines() if l.startswith("====>")))
    wa = torch.load(dests["A"] / VR / "saves/vrnn_state_dict_1.pth", map_location="cpu")
    wb = torch.load(dests["B"] / VR / "saves/vrnn_state_dict_1.pth", map_location="cpu")
    if wa.keys() != wb.keys() or any(not torch.equal(wa[k], wb[k]) for k in wa): stop("training not repeatable: weights differ")
    la = pd.read_csv(dests["A"] / VR / "saves/losses_all.csv"); lb = pd.read_csv(dests["B"] / VR / "saves/losses_all.csv")
    if not la.equals(lb): stop("training not repeatable: losses differ")
    if not np.isfinite(la[["train_loss", "kld_loss", "nll_loss"]].to_numpy(float)).all(): stop("non-finite loss")
    sec_per_epoch = min(times.values()) / 2
    say(f"training repeatable: {len(wa)} weight tensors identical, losses identical | ~{sec_per_epoch:.0f} s/epoch "
        f"-> full 41-epoch training ~{hms(41 * sec_per_epoch)}")

    # 3. generate.py on the noise-free outputs of smoke 1 (epoch-1 weights, smoke only)
    pc, pdn = S1 / "Pandapower/dynamic_Panda_clean.csv", S1 / "Pandapower/dynamic_Panda_denoised.csv"
    a = pd.read_csv(pc).drop(columns=["timestamp"], errors="ignore"); b = pd.read_csv(pdn).drop(columns=["timestamp"], errors="ignore")
    if not a.equals(b):
        same_cols = list(a.columns) == list(b.columns)
        diff = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))) if same_cols and a.shape == b.shape else "shape/columns"
        stop(f"Panda_clean and Panda_denoised differ (max |d| {diff}); which file is upstream's Panda.csv must be decided")
    say("Panda_clean == Panda_denoised (all values except timestamp): either serves as upstream's noise-free Panda.csv")
    raw = vr / "raw_data"
    inputs = {"Mosaik.csv": S1 / "Mosaik/dynamic_Mosaik.csv", "Panda.csv": pc}
    for name, p in inputs.items():
        df = pd.read_csv(p)
        if "power_reactive" not in df.columns or "power_effective" in df.columns:
            stop(f"{p.name}: expected power_reactive and no power_effective, has {df.columns.tolist()}")
        df = df.rename(columns={"power_reactive": "power_effective"})
        miss = [c for c in gen_cols if c not in df.columns]
        if miss: stop(f"{p.name} lacks {miss} (has {df.columns.tolist()})")
        df.to_csv(raw / name, index=False)
    shutil.copyfile(vr / "saves/vrnn_state_dict_1.pth", vr / "saves/vrnn_state_dict_41.pth")
    outs, gen_rel = {}, {}
    for lab, s in (("s0a", 0), ("s0b", 0), ("s1", 1)):
        for f in (vr / "saves/output").glob("*_autoencoderEPICnoise.csv"): f.unlink()
        run([sys.executable, "generate.py"], vr, env(s, s), f"generate_{lab}")
        got = {}
        for name in inputs:
            o = vr / "saves/output" / f"{name[:-4]}_autoencoderEPICnoise.csv"
            if not o.is_file(): stop(f"generate {lab}: {o.name} not written")
            df = pd.read_csv(o)
            n_in = len(pd.read_csv(raw / name))
            if list(df.columns) != gen_cols or len(df) != n_in - 1:
                stop(f"generate {lab} {o.name}: columns {df.columns.tolist()} rows {len(df)} (expected {gen_cols}, {n_in - 1})")
            if not np.isfinite(df.to_numpy(float)).all(): stop(f"generate {lab} {o.name}: non-finite values")
            got[name] = df
        outs[lab] = got
    for name in inputs:
        if not outs["s0a"][name].equals(outs["s0b"][name]): stop(f"generate not repeatable for {name} under the same seed")
        same1 = outs["s0a"][name].equals(outs["s1"][name])
        inp = pd.read_csv(raw / name)[gen_cols].to_numpy(float)[1:]
        out = outs["s0a"][name].to_numpy(float)
        sd = inp.std(0); mad = np.abs(out - inp).mean(0)
        rel = np.where(sd > 0, mad / np.where(sd > 0, sd, 1), np.nan)
        const = [c for c, v in zip(gen_cols, sd) if v == 0]
        gen_rel[name] = {"same_seed_identical": True, "seed1_identical": bool(same1),
                         "median_mad_over_sd": float(np.nanmedian(rel)), "constant_input_columns": const}
        say(f"generate {name}: {out.shape}, seed 0 twice identical, seed 1 "
            f"{'IDENTICAL (output does not depend on the seed)' if same1 else 'differs'} | "
            f"mean|out-in|/sd(in) median {np.nanmedian(rel):.3f} (epoch-1 weights, not representative) | constant inputs {const}")
    say("note: generate.py output names contain 'EPIC'; features.py would label them real -> renamed '+autoencoder' when "
        "the classifier dataset is built (D49)")
    out = {"upstream_commit": COMMIT, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
           "smoke_change": "train.py n_epochs 41 -> 2 in a copy; epoch-1 weights copied to vrnn_state_dict_41.pth for generate.py",
           "adapter": "power_reactive(_noise) -> power_effective(_noise)",
           "train_py": {"window": win, "batch": bs, "epochs": ne, "columns": train_cols, "sequences": n,
                        "train": n_train, "test": n_test},
           "noise_files_sha256": noise_sha["A"], "seconds_per_epoch": round(sec_per_epoch, 1),
           "generate": gen_rel, "panda_clean_equals_denoised": True}
    (ROOT / "spec" / "epic_smoke2.json").write_text(json.dumps(out, indent=1) + "\n")
    say("EPIC SMOKE 2 PASSED")

if __name__ == "__main__":
    main()
