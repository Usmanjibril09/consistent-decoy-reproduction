#!/usr/bin/env python3
"""Stage 06 figures, Batch B (R6-R10). Same layout, fonts, sizes, input gates and display names as Batch A:
everything shared is imported from scripts/06_figures_batchA.py, nothing is re-implemented. Every quantity is checked
for every method and seed before anything is drawn. R10 compares the EPIC detector results.
Output: results/phase3/figures/batchB/R{6..10}.pdf/.png/_values.csv + manifest.json."""
import csv, os, re, json, subprocess, importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("BATCH_A_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = Path(os.environ.get("BATCH_B_OUT", ROOT / "results/phase3/figures/batchB")).resolve()
BATCH_A = Path(__file__).with_name("06_figures_batchA.py")
spec_a = importlib.util.spec_from_file_location("figA", BATCH_A)
if spec_a is None or spec_a.loader is None:
    raise ImportError(f"Cannot load Batch A from {BATCH_A}")
FA: Any = importlib.util.module_from_spec(spec_a)
spec_a.loader.exec_module(FA)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

need, rec = FA.need, FA.rec
need(hasattr(FA, "LAB"), "scripts/06_figures_batchA.py does not apply the shared display names (fig_labels)")
M, C, SP, RULES, METHODS, FULL, W_IN = FA.M, FA.C, FA.SP, FA.RULES, FA.METHODS, FA.FULL, FA.W_IN

PLANTS = ("hai", "qappd")
FIT_FLOOR, FIT_CEIL = -100.0, 100.0   # FIT colour range of R1(b); FIT cannot exceed 100 %
EXC = RULES.get("exclude", {})
need("simprocess/m2b" in EXC.get("qappd", []), "reporting_rules.toml does not exclude simprocess/m2b on qappd (D43/D58)")
M2B_PLANTS = tuple(p for p in PLANTS if "simprocess/m2b" not in EXC.get(p, []))

SRC = {"det": lambda p: M[p]["outputs"]["detector"], "fit": lambda p: M[p]["outputs"]["fit"],
       "cfid": lambda p: M[p]["outputs"]["cfid"], "dtw": lambda p: M[p]["outputs"]["dtw"],
       "clf": lambda p: C[p]["outputs"], "seam": lambda p: FA.J["seam"]["plants"][p]["outputs"]}
SPECS = {}
def spec(name, plants, src, get): SPECS[name] = (tuple(plants), src, get)
spec("det_in", PLANTS, "det", lambda o: o["p_syn"]["in"]["rate"])
spec("det_oor", PLANTS, "det", lambda o: o["p_syn"]["oor"]["rate"])
spec("seam_int", PLANTS, "seam", lambda o: o["p_syn_oor"]["interior"])
spec("sp_m2b", M2B_PLANTS, "sp", lambda o: o["m2b"])
spec("sp_flag", PLANTS, "sp", lambda o: o["psyn_flag_rate"])
spec("sp_pass_decoy", ("hai",), "sp", lambda o: o["decoy_pass_rate"])   # filter is HAI-only (D43)
spec("sp_pass_real", ("hai",), "sp", lambda o: o["real_pass_rate"])
spec("sp_acc", PLANTS, "sp", lambda o: o["accuracy"])
spec("cfid", PLANTS, "cfid", lambda o: o)
spec("dtw_normal", PLANTS, "dtw", lambda o: o["normal"])
spec("dtw_anom", ("qappd",), "dtw", lambda o: o["p_anom"])
spec("tstr", PLANTS, "clf", lambda o: o["tstr_mae"])
spec("fit_normal", PLANTS, "fit", lambda o: o["normal"])
spec("fit_cross", ("hai",), "fit", lambda o: o["crosstime"]["pooled"])
spec("fit_anom", ("qappd",), "fit", lambda o: o["p_anom"])
DIR = {"fit_normal": "metrics/fit/normal", "cfid": "metrics/cfid", "dtw_normal": "metrics/dtw/normal", "tstr": "clf/tstr_mae"}
RANK_NAME = {"fit_normal": "FIT", "cfid": "Context-FID", "dtw_normal": "DTW", "tstr": "Train-on-decoy"}

def okeys(d, m):
    ks = [k for k in d if re.fullmatch(re.escape(m) + r"(_s\d+)?", k)]
    return sorted(ks, key=lambda k: int(k.rsplit("_s", 1)[1]) if re.search(r"_s\d+$", k) else 0)

def sp_groups(p, m):
    pat = re.compile(re.escape(f"{p}/{m}") + r"(?:_s(\d+))?(?:_c\d+)?"); g = {}
    for k in SP:
        mm = pat.fullmatch(k) if isinstance(k, str) else None
        if mm: g.setdefault(int(mm.group(1) or 0), []).append(k)
    return [sorted(g[s]) for s in sorted(g)]

def values(name, p, m):
    plants, src, get = SPECS[name]
    if p not in plants: raise KeyError(f"{name} is not defined for {p}")
    if src == "sp":
        gs = sp_groups(p, m)
        if not gs: raise KeyError(f"no SimProcess entries for {p}/{m}")
        return [float(np.mean([float(get(SP[k])) for k in g])) for g in gs]
    d = SRC[src](p); ks = okeys(d, m)
    if not ks: raise KeyError(f"no {src} outputs for {m}")
    return [float(get(d[k])) for k in ks]

def table(name, p): return {m: values(name, p, m) for m in METHODS}

def shape(x, depth=2):
    if isinstance(x, dict): return {k: shape(v, depth - 1) for k, v in list(x.items())[:8]} if depth > 0 else f"dict[{len(x)}]"
    if isinstance(x, list): return f"list[{len(x)}]"
    return type(x).__name__

def folds():
    d = SRC["fit"]("hai"); f0 = None
    for m in METHODS:
        for k in okeys(d, m):
            ct = d[k]["crosstime"]
            f = sorted(x for x, v in ct.items() if x != "pooled" and isinstance(v, (int, float)))
            if f0 is None: f0 = f
            if not f or f != f0: raise KeyError(f"cross-time sets of {k}: {f} vs {f0}")
            bad = [x for x in f if not np.isfinite(ct[x])]
            if bad: raise ValueError(f"non-finite cross-time FIT {k}: {bad}")
    return f0

def preflight():
    miss = []
    for name, (plants, src, get) in SPECS.items():
        for p in plants:
            for m in METHODS:
                try:
                    v = values(name, p, m); want = len(okeys(SRC["fit"](p), m))
                    if len(v) != want: miss.append(f"{name} {p} {m}: {len(v)} seeds, fit has {want}")
                    elif not np.isfinite(v).all(): miss.append(f"{name} {p} {m}: non-finite {v}")
                except (KeyError, TypeError, IndexError, ValueError) as e:
                    miss.append(f"{name} {p} {m}: {type(e).__name__} {e}")
    try: FA.SEL["R9_folds"] = folds()
    except (KeyError, TypeError, ValueError) as e: miss.append(f"cross-time sets: {type(e).__name__} {e}")
    if miss:
        print(f"   {len(miss)} problems; first 40:")
        for x in miss[:40]: print(f"      {x}")
        print("   input structure (to fix all paths in one round):")
        for lab, f in (("metrics hai outputs", lambda: list(M["hai"]["outputs"])),
                       ("detector entry", lambda: shape(next(iter(SRC["det"]("hai").values())))),
                       ("fit entry", lambda: shape(next(iter(SRC["fit"]("hai").values())))),
                       ("cfid entry", lambda: shape(next(iter(SRC["cfid"]("hai").values())))),
                       ("dtw entry", lambda: shape(next(iter(SRC["dtw"]("hai").values())))),
                       ("classifier entry", lambda: shape(next(iter(SRC["clf"]("hai").values())))),
                       ("seam.json", lambda: shape(FA.J["seam"], 4)),
                       ("simprocess keys", lambda: list(SP)[:12]),
                       ("simprocess entry", lambda: shape(SP[next(k for k in SP if str(k).startswith("hai/"))]))):
            try: print(f"      {lab}: {f()}")
            except Exception as e: print(f"      {lab}: unavailable ({type(e).__name__} {e})")
        FA.stop(f"{len(miss)} quantities missing or invalid; nothing drawn")
    for k, o in SP.items():
        if isinstance(o, dict) and "m2" in o and "accuracy" in o:
            need(abs(o["m2"] - abs(o["accuracy"] - 0.5)) < 1e-9, f"SimProcess {k}: m2 != |accuracy - 0.5|")
    print(f"   gate B: {len(SPECS)} quantities present and finite for {len(METHODS)} methods and all seeds; "
          f"SimProcess P(real) only on {M2B_PLANTS} (D43/D58); SimProcess m2 = |accuracy - 0.5| for every entry")

def keep(n, lt, p, name, vals):
    for m, v in vals.items():
        for i, x in enumerate(v): rec(n, lt, f"{p} {m}", f"{name} seed {i}", x, "")

def direction(name):
    return "(higher is better)" if FA.BETTER[DIR[name]] > 0 else "(lower is better)"

def rows(ax, series, xlab, scale="linear", lo=None, ref=None, ref_text=None):
    ys = np.arange(len(METHODS))[::-1]
    offs = np.linspace(-0.2, 0.2, len(series)) if len(series) > 1 else np.zeros(1)
    allv = np.array([x for s in series for v in s[3].values() for x in v], float)
    if lo is not None: allv = np.maximum(allv, lo)
    if scale == "auto-log": scale = "log" if (allv > 0).all() else "symlog"
    if scale == "log": ax.set_xscale("log")
    elif scale == "symlog":
        pos = np.abs(allv[allv != 0])
        ax.set_xscale("symlog", linthresh=float(10 ** np.floor(np.log10(pos.min()))) if pos.size else 1.0)
    for (lab, c, mk, vals), o in zip(series, offs):
        for y, m in zip(ys, METHODS):
            v = np.asarray(vals[m], float); sv = np.maximum(v, lo) if lo is not None else v
            if v.size > 1:
                ax.plot([sv.min(), sv.max()], [y + o] * 2, color=c, lw=2.4, alpha=0.25, solid_capstyle="butt", zorder=2)
            ax.scatter(sv, np.full(v.size, y + o), s=8, color=c, marker=mk, lw=0, alpha=0.85, zorder=3)
            mu = float(v.mean()); smu = max(mu, lo) if lo is not None else mu
            ax.plot([smu], [y + o], marker=mk, ms=5, mfc="white", mec=c, mew=1.0, ls="", zorder=4)
            if smu != mu:
                ax.annotate(f"{mu:.0f}", (smu, y + o), xytext=(5, 0), textcoords="offset points", fontsize=8, va="center", color=c)
        ax.plot([], [], marker=mk, ls="", mfc="white", mec=c, label=lab)
    if ref is not None:
        ax.axvline(ref, color="0.5", lw=0.6, ls=":")
        if ref_text: ax.text(ref, 0.01, f" {ref_text}", transform=ax.get_xaxis_transform(), fontsize=8, color="0.35", ha="left", va="bottom")
    ax.set_yticks(ys, METHODS, fontsize=8); ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_xlabel(xlab); FA.clean_grid(ax, "x")
    if len(series) > 1:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=len(series), fontsize=8,
                  handlelength=1.0, columnspacing=1.2)

def annotate(ax, V, fmt, dark):
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, fmt(V[i, j]), ha="center", va="center", fontsize=8, color="white" if dark(V[i, j]) else "black")

def both(name, cols, marks="os"):
    return [(p.upper(), c, mk, table(name, p)) for p, c, mk in zip(PLANTS, cols, marks)]

# ---------------- R6: where the decoys are caught
def fig_R6():
    FA.LAB.begin("R6")
    n = "R6"; ca, cb = FA.TOL_VIBRANT[0], FA.TOL_VIBRANT[3]
    fig = plt.figure(figsize=(W_IN, 6.5))
    for j, p in enumerate(PLANTS):
        ax = FA.panel_axes(fig, j); lt = "ab"[j]
        a, b = table("det_oor", p), table("seam_int", p)
        rows(ax, [("All free-run steps", ca, "o", a), ("Interior steps only", cb, "s", b)], "Out-of-range alarm rate")
        keep(n, lt, p, "det_oor", a); keep(n, lt, p, "seam_int", b)
        FA.panel_title(ax, lt, f"{p.upper()}, interior steps")
    for j, p in enumerate(PLANTS):
        ax = FA.panel_axes(fig, 2 + j); lt = "cd"[j]
        a, b = table("det_in", p), table("det_oor", p)
        rows(ax, [("In-range probes", ca, "o", a), ("Out-of-range probes", cb, "s", b)], "Physics-rule alarm rate")
        keep(n, lt, p, "det_in", a); keep(n, lt, p, "det_oor", b)
        FA.panel_title(ax, lt, f"{p.upper()}, probe range")
    FA.save(fig, n, out=OUT)

# ---------------- R7: SimProcess
def fig_R7():
    FA.LAB.begin("R7")
    n = "R7"; ca, cb = FA.TOL_MUTED[0], FA.TOL_MUTED[6]
    need(len(M2B_PLANTS) == 1, f"SimProcess P(real) plants {M2B_PLANTS}"); p0 = M2B_PLANTS[0]
    fig = plt.figure(figsize=(W_IN, 6.5))
    ax = FA.panel_axes(fig, 0); v = table("sp_m2b", p0)
    rows(ax, [(p0.upper(), ca, "o", v)], "P(real) of decoy windows", scale="symlog")
    keep(n, "a", p0, "sp_m2b", v); FA.panel_title(ax, "a", f"{p0.upper()}, judged real")
    ax = FA.panel_axes(fig, 1); s = both("sp_flag", (ca, cb))
    rows(ax, s, "Probe windows flagged as simulated")
    for p, t in zip(PLANTS, s): keep(n, "b", p, "sp_flag", t[3])
    FA.panel_title(ax, "b", "Probe windows flagged")
    ax = FA.panel_axes(fig, 2); a, b = table("sp_pass_decoy", "hai"), table("sp_pass_real", "hai")
    rows(ax, [("Decoy windows", ca, "o", a), ("Real windows", cb, "s", b)], "Windows passing the noise filter")
    keep(n, "c", "hai", "sp_pass_decoy", a); keep(n, "c", "hai", "sp_pass_real", b)
    FA.panel_title(ax, "c", "HAI, noise filter")
    ax = FA.panel_axes(fig, 3); s = both("sp_acc", (ca, cb))
    rows(ax, s, "SimProcess accuracy", ref=0.5, ref_text="chance")
    for p, t in zip(PLANTS, s): keep(n, "d", p, "sp_acc", t[3])
    FA.panel_title(ax, "d", "Detection accuracy")
    FA.save(fig, n, out=OUT)

# ---------------- R8: distributional fidelity
def fig_R8():
    FA.LAB.begin("R8")
    n = "R8"; ca, cb = FA.TOL_BRIGHT[0], FA.TOL_BRIGHT[1]
    fig = plt.figure(figsize=(W_IN, 6.5))
    for j, (name, xlab, scale, title) in enumerate((("cfid", "Context-FID", "auto-log", "Context-FID"),
                                                    ("dtw_normal", "DTW distance", "auto-log", "Shape distance"),
                                                    ("tstr", "Train-on-decoy error (MAE)", "linear", "Usefulness as training data"))):
        ax = FA.panel_axes(fig, j); lt = "abc"[j]; s = both(name, (ca, cb))
        rows(ax, s, f"{xlab} {direction(name)}", scale=scale)
        for p, t in zip(PLANTS, s): keep(n, lt, p, name, t[3])
        FA.panel_title(ax, lt, title)
    ax = FA.panel_axes(fig, 3); cols = [(p, k) for p in PLANTS for k in DIR]
    R = np.zeros((len(METHODS), len(cols)))
    for j, (p, k) in enumerate(cols):
        mu = np.array([np.mean(values(k, p, m)) for m in METHODS]); s = mu * FA.BETTER[DIR[k]]
        R[:, j] = [1 + np.count_nonzero(s > x) + 0.5 * (np.count_nonzero(s == x) - 1) for x in s]
        need(abs(R[:, j].sum() - len(METHODS) * (len(METHODS) + 1) / 2) < 1e-9, f"ranks {p} {k}")
        for i, m in enumerate(METHODS): rec(n, "d", f"{p} {m}", k, float(R[i, j]), f"seed mean {mu[i]}")
    im = ax.imshow(R, cmap="viridis", vmin=1, vmax=len(METHODS), aspect="auto")
    annotate(ax, R, lambda x: f"{x:g}", lambda x: x <= (len(METHODS) + 1) / 2)
    ax.axvline(len(DIR) - 0.5, color="white", lw=1.6)
    ax.set_xticks(range(len(cols)), [f"{p.upper()}\n{RANK_NAME[k]}" for p, k in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8); FA.heat_axes(ax)
    cb = FA.colorbar(fig, im, ax); cb.set_label("Rank (1 = best)", fontsize=8); cb.ax.tick_params(labelsize=7); cb.ax.invert_yaxis()
    FA.panel_title(ax, "d", "Ranks across fidelity metrics")
    print("   gate R8: ranks complete in every column (sum n(n+1)/2); better-direction from the ablation definitions")
    FA.save(fig, n, out=OUT)

# ---------------- R9: generalisation
def fig_R9():
    FA.LAB.begin("R9")
    n = "R9"; ca, cb = FA.TOL_HC[0], FA.TOL_HC[2]
    fig = plt.figure(figsize=(W_IN, 6.5))
    ax = FA.panel_axes(fig, 0); a, b = table("fit_normal", "hai"), table("fit_cross", "hai")
    rows(ax, [("Held-out normal", ca, "o", a), ("Other periods", cb, "s", b)], "FIT (%)", lo=FIT_FLOOR)
    keep(n, "a", "hai", "fit_normal", a); keep(n, "a", "hai", "fit_cross", b)
    FA.panel_title(ax, "a", "HAI, across time")
    ax = FA.panel_axes(fig, 1); fl = FA.SEL["R9_folds"]; FO = SRC["fit"]("hai")
    V = np.array([[np.mean([FO[k]["crosstime"][f] for k in okeys(FO, m)]) for f in fl] for m in METHODS])
    im = ax.imshow(V, cmap="PuOr", norm=mcolors.TwoSlopeNorm(vmin=FIT_FLOOR, vcenter=0, vmax=FIT_CEIL), aspect="auto")
    annotate(ax, V, lambda x: f"{x:.0f}", lambda x: abs(x) > 60)
    pre = os.path.commonprefix(fl)
    ax.set_xticks(range(len(fl)), [FA.display(f[len(pre):] or f) for f in fl], fontsize=8); ax.set_xlabel("Cross-time set")
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8); FA.heat_axes(ax)
    cbar = FA.colorbar(fig, im, ax, extend="min"); cbar.set_label("FIT (%)", fontsize=8); cbar.ax.tick_params(labelsize=7)
    for i, m in enumerate(METHODS):
        for j, f in enumerate(fl): rec(n, "b", f"hai {m}", f, float(V[i, j]), "seed mean cross-time FIT")
    FA.panel_title(ax, "b", "HAI, FIT by set")
    ax = FA.panel_axes(fig, 2); a, b = table("fit_normal", "qappd"), table("fit_anom", "qappd")
    rows(ax, [("Normal data", ca, "o", a), ("Anomalous data", cb, "s", b)], "FIT (%)", lo=FIT_FLOOR)
    keep(n, "c", "qappd", "fit_normal", a); keep(n, "c", "qappd", "fit_anom", b)
    FA.panel_title(ax, "c", "QAPPD, anomalous data")
    ax = FA.panel_axes(fig, 3); a, b = table("dtw_normal", "qappd"), table("dtw_anom", "qappd")
    rows(ax, [("Normal data", ca, "o", a), ("Anomalous data", cb, "s", b)], f"DTW distance {direction('dtw_normal')}", scale="auto-log")
    keep(n, "d", "qappd", "dtw_normal", a); keep(n, "d", "qappd", "dtw_anom", b)
    FA.panel_title(ax, "d", "QAPPD, shape distance")
    FA.save(fig, n, out=OUT)

# ---------------- R10: EPIC, the authors' detector and ours

R10_COL = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]   # Okabe-Ito, not used in R6-R9

def r10_tracked(rel):
    p = ROOT / rel
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    need(p.exists() and tracked and clean, f"R10: {rel} is not committed unchanged")
    return p

def r10_csv(p):
    rows = [r for r in csv.reader(open(p)) if r and not r[0].lstrip().startswith("#")]
    return [dict(zip(rows[0], r)) for r in rows[1:]]

def r10_meas(v):   # 'power_real' -> 'Real power', 'current I1' -> 'Current I1' (never the raw EPIC column name)
    v = v.strip()
    return v[6:].capitalize() + " power" if v.startswith("power_") else v[:1].upper() + v[1:]

def r10_noise(k):  # 'gaussian2' -> 'Gaussian 2', 'gmm' -> 'GMM', 'laplace+uniform' -> 'Laplace+Uniform'
    return "+".join(t.upper() if len(t) <= 3 else re.sub(r"(\D)(\d+)$", r"\1 \2", t.capitalize()) for t in k.split("+"))

def r10_legend(ax, ncol, *, y=-0.17, handles=None):
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, fontsize=8, handlelength=1.0, columnspacing=1.0)

def fig_R10():
    FA.LAB.begin("R10")
    n = "R10"
    p_t2, p_cs, p_se = (r10_tracked(r) for r in ("spec/simprocess_table2.csv", "spec/epic_classify_summary.json",
                                                   "spec/epic_sensitivity_summary.json"))
    p_ao = ROOT / "results/phase3/epic_armO/summary.json"
    need(p_ao.exists() and FA.sha(p_ao) in FA.DEC, "R10: arm O summary sha256 not recorded in spec/decisions.md (D69)")
    FA.SEL["R10_inputs"] = {str(p.relative_to(ROOT)): FA.sha(p) for p in (p_t2, p_cs, p_se, p_ao)}
    T2 = r10_csv(p_t2); CS = json.load(open(p_cs)); SE = json.load(open(p_se))["rows"]; AO = json.load(open(p_ao))["table"]
    order = [r["value"] for r in T2]; ys = np.arange(len(order))[::-1]; names = ["Appar. power" if v == "power_apparent" else r10_meas(v) for v in order]
    arms = [("U", "SimProcess code"), ("U-rank", "Ranking fixed"), ("U-split", "Leak-free split")]
    E = {}
    for a, _ in arms:
        need(f"{a}__all" in CS["evaluation"], f"R10: arm {a}__all missing in epic_classify_summary.json")
        E[a] = {e["value"]: e for e in CS["evaluation"][f"{a}__all"]}
        need(set(E[a]) == set(order), f"R10: {a} measurements {sorted(E[a])} differ from Table 2 {order}")
    for r in T2:
        for a, _ in arms:
            e = E[a][r["value"]]
            need(abs(e["table2"] - float(r["prob"])) < 1e-12, f"R10: Table 2 value differs for {r['value']} ({a})")
            need((e["column"], e["noise"]) == (E["U"][r["value"]]["column"], E["U"][r["value"]]["noise"]),
                 f"R10: column/noise differ between arms for {r['value']}")
    passed = sum(1 for v in order if E["U"][v]["primary"] and E["U"][v]["within"])
    need(passed == CS["rule"]["passed"], f"R10: {passed} rows within tolerance, recorded rule says {CS['rule']['passed']}")
    print(f"   gate R10(a): Table 2 values match; {passed} of {CS['rule']['primary_rows']} primary rows within tolerance, as recorded (D65)")
    N = max(d["n_generations"] for d in AO.values())
    for k, d in AO.items():
        g = np.asarray(d["per_generation"], float)
        need(len(g) == d["n_generations"] and np.isfinite(g).all() and abs(g.mean() - d["p_real_mean"]) < 1e-12,
             f"R10: arm O entry {k} inconsistent")
    print(f"   gate R10(b,d): {len(AO)} arm O entries, per-generation means reproduce stored means (up to {N} generations)")
    fig = plt.figure(figsize=(W_IN, 6.5))

    # (a) their code against their own Table 2
    ax = FA.panel_axes(fig, 0); offs = (0.22, 0.0, -0.22)
    for y, v in zip(ys, order):
        e = E["U"][v]
        ax.add_patch(Rectangle((e["table2"] - e["tol"], y - 0.42), 2 * e["tol"], 0.84, color="0.88", lw=0, zorder=1))
        ax.plot([e["table2"]], [y], marker="D", ms=5, mfc="white", mec="black", mew=1.0, ls="", zorder=4)
        rec(n, "a", "Published", r10_meas(v), e["table2"], f"tolerance {e['tol']}")
        for (a, lab), o, c, mk in zip(arms, offs, R10_COL, "os^"):
            ea = E[a][v]
            if ea["mean"] is None: continue
            ax.errorbar(ea["mean"], y + o, xerr=ea["sd"] if ea["sd"] is not None else None, fmt=mk, ms=4, color=c,
                        mfc=c, mec=c, elinewidth=0.9, capsize=1.5, zorder=3)
            rec(n, "a", lab, r10_meas(v), ea["mean"], f"sd {ea['sd']} over generations; noise {ea['noise']}")
    ax.plot([], [], marker="D", ls="", mfc="white", mec="black", label="SimProcess, published")
    for (a, lab), c, mk in zip(arms, R10_COL, "os^"): ax.plot([], [], marker=mk, ls="", color=c, label=lab)
    ax.set_yticks(ys, names, fontsize=8); ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlim(0, 1)
    ax.set_xlabel("P(judged real), best SimProcess noise"); FA.clean_grid(ax, "x"); r10_legend(ax, 2)
    FA.panel_title(ax, "a", "Rerunning SimProcess")

    # Arm O records the aggregate measurement under all_columns.
    arm_o_columns = {v: ("all_columns" if E["U"][v]["column"] == "all" else E["U"][v]["column"]) for v in order}

    # (b) published value against our detector, same generations
    ax = FA.panel_axes(fig, 1); subs = (("all", "Our detector", True, 0.15), ("no6", "Our detector, w/o scenario 6", False, -0.15))
    few = False
    for y, v in zip(ys, order):
        e = E["U"][v]
        ax.plot([e["table2"]], [y], marker="D", ms=5, mfc="white", mec="black", mew=1.0, ls="", zorder=4)
        for s, lab, filled, o in subs:
            k = f"{s}/{arm_o_columns[v]}/{e['noise']}"; need(k in AO, f"R10: arm O entry {k} missing")
            d = AO[k]; g = np.asarray(d["per_generation"], float); mk = "o" if d["n_generations"] == N else "^"
            few |= d["n_generations"] < N; c = R10_COL[5]
            if g.size > 1: ax.plot([g.min(), g.max()], [y + o] * 2, color=c, lw=2.4, alpha=0.25, solid_capstyle="butt", zorder=2)
            ax.plot([d["p_real_mean"]], [y + o], marker=mk, ms=5, ls="", mec=c, mfc=c if filled else "white", mew=1.0, zorder=3)
            rec(n, "b", lab, r10_meas(v), d["p_real_mean"], f"{d['n_generations']} of {N} generations; {k}")
    ax.set_xscale("log")
    ax.plot([], [], marker="D", ls="", mfc="white", mec="black", label="SimProcess, published")
    for s, lab, filled, o in subs: ax.plot([], [], marker="o", ls="", mec=R10_COL[5], mfc=R10_COL[5] if filled else "white", label=lab)
    if few: ax.plot([], [], marker="^", ls="", mec="0.3", mfc="0.3", label=f"<{N} generations")
    ax.set_yticks(ys, names, fontsize=8); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("P(judged real), best SimProcess noise"); FA.clean_grid(ax, "x"); r10_legend(ax, 2)
    FA.panel_title(ax, "b", "SimProcess vs. our detector")

    # (c) detector settings on HAI and QAPPD (accuracy only; QAPPD P(real) not plotted, D43)
    ax = FA.panel_axes(fig, 2)
    ser = [("Our detector", "ours_accuracy", None), ("SimProcess settings", "U_accuracy", "U_accuracy_sd"),
           ("Ordered split", "U-split_accuracy", "U-split_accuracy_sd")]
    plants = [p for p in PLANTS if any(k.startswith(p + "/") for k in SE)]
    yt, yl, y = [], [], 0.0; heads = []
    for p in plants[::-1]:
        keys = [k for k in SE if k.startswith(p + "/")][::-1]
        for k in keys:
            for (lab, f, fsd), o, c in zip(ser, (0.27, 0.0, -0.27), R10_COL):
                v = SE[k][f]; sd = SE[k][fsd] if fsd else None
                need(0.0 <= v <= 1.0 and (sd is None or sd >= 0), f"R10: sensitivity {k} {f} out of range")
                ax.barh(y + o, v, height=0.25, color=c, xerr=sd, error_kw={"elinewidth": 0.8, "capsize": 1.2}, zorder=3)
                rec(n, "c", lab, k, v, f"sd {sd}")
            yt.append(y); yl.append(k.split("/", 1)[1]); y += 1
        heads.append((y, p.upper())); y += 1
    for yy, t in heads:
        ax.text(0.5, yy, t, ha="center", va="center", fontsize=8, zorder=5, bbox={"fc": "white", "ec": "none", "pad": 0.5})
    ax.axvline(0.5, color="0.5", lw=0.6, ls=":")
    ax.set_yticks(yt, yl, fontsize=8); ax.set_ylim(-0.6, y - 0.4); ax.set_xlim(0, 1.02)
    ax.set_xlabel("Detection accuracy"); FA.clean_grid(ax, "x")
    handles: list[Patch | Line2D] = [Patch(facecolor=c, label=lab) for (lab, _, _), c in zip(ser, R10_COL)]
    handles.append(Line2D([], [], color="0.5", lw=0.6, ls=":", label="Chance"))
    r10_legend(ax, 2, y=-0.24, handles=handles)
    FA.panel_title(ax, "c", "Our decoys, three detectors")

    # (d) every noise model under our detector (all recordings); outlined = their Table 2 choice
    ax = FA.panel_axes(fig, 3)
    cols = set(arm_o_columns.values())
    noises = sorted({k.split("/", 2)[2] for k in AO if k.startswith("all/") and k.split("/")[1] in cols})
    V = np.full((len(order), len(noises)), np.nan)
    for i, v in enumerate(order):
        for j, z in enumerate(noises):
            k = f"all/{arm_o_columns[v]}/{z}"
            if k in AO:
                V[i, j] = AO[k]["p_real_mean"]
                rec(n, "d", r10_meas(v), r10_noise(z), V[i, j], f"{AO[k]['n_generations']} of {N} generations")
    fin = V[np.isfinite(V)]; need(fin.size and (fin > 0).all(), "R10(d): non-positive P(real) on a log colour scale")
    norm = mcolors.LogNorm(vmin=float(fin.min()), vmax=float(fin.max()))
    im = ax.imshow(np.ma.masked_invalid(V), cmap="cividis", norm=norm, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if not np.isfinite(V[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="0.6", lw=0))
            elif V[i, j] >= 0.1:
                ax.text(j, i, f"{V[i, j]:.2f}"[1:], ha="center", va="center", fontsize=8,
                        color="white" if norm(V[i, j]) < 0.5 else "black")
        z = E["U"][order[i]]["noise"]
        if z in noises:
            ax.add_patch(Rectangle((noises.index(z) - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.2, zorder=4))
    ax.set_xticks(range(len(noises)), [r10_noise(z) for z in noises], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), names, fontsize=8); FA.heat_axes(ax)
    cb = FA.colorbar(fig, im, ax); cb.set_label("P(judged real)", fontsize=8); cb.ax.tick_params(labelsize=7)
    FA.panel_title(ax, "d", "SimProcess noises, our detector")
    FA.save(fig, n, out=OUT)


if __name__ == "__main__":
    preflight()
    for f in (fig_R6, fig_R7, fig_R8, fig_R9, fig_R10): f()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {"stage": "06_figures batch B", "repo_head": head, "script_sha256": FA.sha(Path(__file__)),
           "batch_a_script_sha256": FA.sha(BATCH_A),
           "labels": {"file": "spec/figure_labels.toml", "sha256": FA.sha(FA.LAB.FILE)},
           "matplotlib": FA.matplotlib.__version__, "numpy": np.__version__,
           "inputs": {k: {"path": str(q.relative_to(FA.ROOT)), "sha256": FA.H[k]} for k, q in FA.F.items()},
           "recorded_in": FA.REC, "selections": FA.SEL, "figures": FA.FIGS}
    for p_ in FA.LAB.PROBLEMS: print("   label note (for your review, figure written): " + p_)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"BATCH B DONE -> {OUT} ({len(FA.FIGS)} figures; manifest sha256 {FA.sha(OUT / 'manifest.json')[:12]})")
