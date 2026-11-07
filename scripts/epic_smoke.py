#!/usr/bin/env python3
"""EPIC arm, D48(h) smoke part 1: import check, seed shim, generator determinism.
Runs the pinned upstream generators (SimProcess 7533f59) unmodified, in fresh `git archive`
extracts under ~/epic_work/smoke. Pass: every output named in each generator.sh is produced,
no traceback, no GMM fitting error; seed 0 twice identical (bytes, or values with only a
wall-clock time column differing); every noisy output changes with seed 1 in a non-time column.
Noise-free outputs (--output-noiseless, or --noise "none" only) may stay the same.
Writes spec/epic_smoke.json only on PASS. No classifier is trained here."""
import ast, hashlib, importlib.util, json, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

COMMIT = "7533f59e94e9f54d8e3f151cc53a02a5c6146b6f"
UP = Path.home() / "epic_work" / "upstream_7533f59"
EPIC = Path("/Users/ujw/epic2018_local")
ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "tools" / "epic_seed"
WORK = Path.home() / "epic_work" / "smoke"
GENS = ("Pandapower", "Mosaik")
RUNS = (("seed0_a", 0), ("seed0_b", 0), ("seed1", 1))
CLOCK = re.compile(r"(?i)time|date")

def say(*a): print(*a, flush=True)
def stop(m): say("STOP:", m); sys.exit(1)
def hms(s): s = int(s); return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def env_for(seed):
    e = dict(os.environ)
    e["PATH"] = str(Path(sys.executable).parent) + os.pathsep + e["PATH"]
    e.update(PYTHONPATH=str(SHIM), EPIC_SEED=str(seed), PYTHONHASHSEED="0",
             MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    return e

def check_imports():
    local = {p.stem for p in UP.rglob("*.py")} | {p.name for p in UP.rglob("*") if p.is_dir()}
    files = [UP / "Generators" / g / "main.py" for g in GENS] + list((UP / "Discriminator" / "simprocess").rglob("*.py"))
    missing = {}
    for f in files:
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except SyntaxError as e:
            stop(f"cannot parse {f.relative_to(UP)}: {e}")
        for node in tree.body:
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                   [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            for m in mods:
                top = m.split(".")[0]
                if top in sys.stdlib_module_names or top in local: continue
                if importlib.util.find_spec(top) is None:
                    missing.setdefault(top, set()).add(str(f.relative_to(UP)))
    if missing:
        stop("modules imported by upstream but not installed: " +
             "; ".join(f"{k} (in {sorted(v)})" for k, v in sorted(missing.items())))
    say("imports OK (generator main.py files and simprocess: all module-level imports resolve)")

def check_shim():
    code = ("import sitecustomize, numpy, random; print(sitecustomize.__file__); "
            "print(repr(numpy.random.rand())); print(repr(random.random()))")
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code], env=env_for(0), capture_output=True, text=True)
        if r.returncode: stop("shim check failed:\n" + r.stderr)
        outs.append(r.stdout.split())
    f, a, b = outs[0]
    if Path(f).resolve() != (SHIM / "sitecustomize.py").resolve():
        stop(f"sitecustomize loaded from {f}, not {SHIM}")
    import random
    want_np = repr(np.random.RandomState(0).rand()); random.seed(0); want_py = repr(random.random())
    if outs[0] != outs[1] or a != want_np or b != want_py:
        stop(f"shim does not seed: {outs} expected numpy {want_np} random {want_py}")
    say(f"shim OK: numpy {a} | random {b} (= seed 0, twice)")

def epic_sources():
    src = {}
    allcsv = sorted(q for q in EPIC.rglob("*.csv") if q.is_file())
    proc = [q for q in allcsv if q.name.lower().startswith("processed")]
    for i in range(1, 9):
        rx = re.compile(rf"(?i)scenario[ _]?{i}(?!\d)")
        m = [q for q in proc if rx.search(q.name)]
        if len(m) != 1:
            say(f"csv files under {EPIC}: {len(allcsv)} (showing up to 80)")
            for q in allcsv[:80]: say("   " + str(q.relative_to(EPIC)))
            stop(f"scenario {i}: {len(m)} processed files found: {[str(q.relative_to(EPIC)) for q in m]}")
        src[i] = m[0]
    return src

def parse_gen(gsh):
    live = "\n".join(l for l in gsh.read_text().splitlines() if not l.lstrip().startswith("#"))
    live = re.sub(r"\\[ \t]*\n", " ", live)
    prefixes = re.findall(r'^\s*generate_\w+\s+"([^"]*)"', live, re.M)
    outs, quiet = set(), set()
    for cmd in re.findall(r"python3\s+main\.py[^\n]*", live):
        noises = re.findall(r'--noise\s+"([^"]*)"', cmd)
        noisy = re.findall(r'--(?:output-noisy|processed-output)\s+"\$\{name\}([^"]+)"', cmd)
        clean = re.findall(r'--output-noiseless\s+"\$\{name\}([^"]+)"', cmd)
        outs |= set(noisy + clean); quiet |= set(clean)
        if noises and all(n.strip() == "none" for n in noises): quiet |= set(noisy)
    return (sorted(p + n for p in prefixes for n in outs),
            sorted(p + n for p in prefixes for n in quiet), prefixes)

def snapshot(d):
    return {str(p.relative_to(d)): sha(p) for p in d.rglob("*.csv") if p.is_file()}

def tail(p, n=25):
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])

def compare(p1, p2):
    """None if identical bytes; else list of differing columns (or ['<shape>'])."""
    if sha(p1) == sha(p2): return None
    a, b = pd.read_csv(p1), pd.read_csv(p2)
    if list(a.columns) != list(b.columns) or len(a) != len(b): return ["<shape>"]
    return [c for c in a.columns if not a[c].equals(b[c])]

def run_gen(dest, gen, seed, label, exp):
    gdir = dest / "Generators" / gen
    before = snapshot(dest)
    log = WORK / f"{label}_{gen}.log"
    t0 = time.time(); last = t0
    with open(log, "w") as lf:
        p = subprocess.Popen(["bash", "generator.sh"], cwd=gdir, env=env_for(seed),
                             stdout=lf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(5)
            if time.time() - last >= 30:
                have = {q.name for q in dest.rglob("*.csv")}
                lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
                say(f"   [{label} {gen}] {sum(e in have for e in exp)}/{len(exp)} outputs | "
                    f"{hms(time.time() - t0)} | {(lines[-1][:70] if lines else '')}")
                last = time.time()
    dt = time.time() - t0
    txt = log.read_text(errors="replace")
    if "Traceback" in txt: stop(f"{label} {gen}: Python traceback (log {log}):\n{tail(log)}")
    errs = [l for l in txt.splitlines() if l.startswith("Error:")]
    if errs: stop(f"{label} {gen}: upstream main.py caught an exception and continued: {errs[:5]} (log {log})")
    if "Error initializing GMM" in txt:
        stop(f"{label} {gen}: GMM fit on EPIC6.csv failed, upstream would silently use manual GMM defaults:\n" +
             "\n".join(l for l in txt.splitlines() if "GMM" in l)[:2000])
    if p.returncode: stop(f"{label} {gen}: generator.sh exit {p.returncode} (log {log}):\n{tail(log)}")
    after = snapshot(dest)
    new = {k: v for k, v in after.items() if before.get(k) != v}
    names = {Path(k).name for k in new}
    miss = [e for e in exp if e not in names]
    if miss: stop(f"{label} {gen}: expected outputs not produced: {miss} (log {log})")
    gm = [l.strip() for l in txt.splitlines() if "Successfully fitted GMM" in l]
    warn = sum(1 for l in txt.splitlines() if re.search(r"(?i)warning|error", l))
    say(f"   [{label} {gen}] done: {len(new)} csv in {hms(dt)} | GMM: {gm[:1] or 'no fit message'} | warning/error lines {warn}")
    return new, dt

def main():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    say(f"EPIC smoke 1 | repo {head} | upstream {COMMIT[:7]} | python {platform.python_version()} | numpy {np.__version__}")
    check_imports()
    check_shim()
    src = epic_sources()
    epic = {f"EPIC{i}.csv": {"source": p.name, "sha256": sha(p)} for i, p in src.items()}
    say("EPIC copies: " + ", ".join(f"EPIC{i} <- {p.name}" for i, p in src.items()))
    gens = {}
    for g in GENS:
        gsh = UP / "Generators" / g / "generator.sh"
        if not gsh.is_file(): stop(f"{gsh} not found")
        exp, quiet, pre = parse_gen(gsh)
        if not exp: stop(f"{g}: no output names parsed from generator.sh")
        gens[g] = (exp, quiet)
        say(f"{g}: prefixes {pre}, {len(exp)} expected outputs, noise-free {quiet}")
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    res = {}
    for label, seed in RUNS:
        dest = WORK / label; dest.mkdir()
        subprocess.run(f"git -C '{UP}' archive --format=tar {COMMIT} | tar -x -C '{dest}'", shell=True, check=True)
        ds = dest / "Discriminator" / "dataset"; ds.mkdir(parents=True, exist_ok=True)
        for i, p in src.items(): shutil.copyfile(p, ds / f"EPIC{i}.csv")
        say(f"== run {label} (EPIC_SEED={seed})")
        res[label] = {g: run_gen(dest, g, seed, label, gens[g][0]) for g in GENS}
    a, b, c = (res[l] for l, _ in RUNS)
    clock_exempt = {}
    for g in GENS:
        fa, fb, fc = a[g][0], b[g][0], c[g][0]
        if fa.keys() != fb.keys() or fa.keys() != fc.keys(): stop(f"{g}: output file sets differ between runs")
        quiet = set(gens[g][1])
        for k in sorted(fa):
            d = compare(WORK / "seed0_a" / k, WORK / "seed0_b" / k)
            if d is not None:
                if not d or any(not CLOCK.search(x) for x in d):
                    stop(f"{g}: seed 0 twice NOT identical: {Path(k).name} differs in {d}")
                clock_exempt[k] = d
        changed, same = [], []
        for k in sorted(fa):
            d = compare(WORK / "seed0_a" / k, WORK / "seed1" / k)
            (changed if d is not None and [x for x in d if not CLOCK.search(x)] else same).append(Path(k).name)
        bad = [n for n in same if n not in quiet]
        say(f"{g}: seed 0 twice identical ({len(fa)} files); seed 1 changes {len(changed)}; unchanged {same}")
        if bad: stop(f"{g}: noisy outputs ignore the seed: {bad}")
        for k in sorted(fa):
            df = pd.read_csv(WORK / "seed0_a" / k)
            say(f"   {g} {Path(k).name}: {len(df)} rows x {df.shape[1]} cols")
    if clock_exempt: say("identical except wall-clock column(s): " + json.dumps(clock_exempt))
    out = {"upstream_commit": COMMIT, "repo_head": head, "python": platform.python_version(),
           "numpy": np.__version__, "requirements_epic_sha256": sha(ROOT / "spec" / "requirements-epic.txt"),
           "epic_copies": epic,
           "rule": "seed0 twice identical (bytes, or only time/date columns differ); noisy outputs change with seed 1; noise-free outputs exempt",
           "clock_column_exemptions": clock_exempt,
           "noise_free": {g: gens[g][1] for g in GENS},
           "seed0_outputs": {g: dict(sorted(a[g][0].items())) for g in GENS},
           "seconds": {l: {g: round(res[l][g][1], 1) for g in GENS} for l, _ in RUNS}}
    (ROOT / "spec" / "epic_smoke.json").write_text(json.dumps(out, indent=1) + "\n")
    say("EPIC SMOKE 1 PASSED")

if __name__ == "__main__":
    main()
