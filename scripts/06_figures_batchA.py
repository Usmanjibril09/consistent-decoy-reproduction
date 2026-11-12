#!/usr/bin/env python3
"""Stage 06 figures, Batch A (R1-R5), IEEE two-column span, Times New Roman.
Every plotted number is read from Phase 3 result files whose sha256 is checked against the records
(statistics/ablation inputs, spec/decisions.md, stage3c.json); cross-checks stop the run on any mismatch.
Output: results/phase3/figures/batchA/R{1..5}.pdf + .png (600 dpi) + R{n}_values.csv + manifest.json."""
import os
import csv, hashlib, json, re, subprocess, sys, tomllib
from pathlib import Path
from typing import NoReturn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, font_manager, transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = Path(os.environ.get("BATCH_A_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import fig_labels as LAB  # display names from spec/figure_labels.toml, applied when text is set
# Short display names for Batch A. Batch B keeps the shared names.
LAB.BASE.update({
    "C1-Full": "Proposed", "Full": "Proposed", "C1": "No layers",
    "C1-A1": "No physics", "C1-A2": "No fallback", "C1-A4": "Limits only",
    "A1": "No physics", "A2": "No fallback", "A4": "Limits only",
    "A5": "No layers", "A3": "No cmd/layers", "A3-bare": "No cmd/layers",
    "A3-L": "No command", "B2": "ARX", "B3": "Grey-box",
    "B6": "GAN + AE", "B8-L": "Layered TF",
})
LAB._T["override"]["R4"]["A3"] = "No command"
LAB.EXACT.update({
    "Det. in-range": "In-range", "Det. OOR": "Out-of-range",
    "Interior OOR": "Interior", "M2 RNN": "RNN", "M2b SP": "SimProc P(real)",
    "full-context + C2/C3": "Context + layers", "window (amortised)": "Window per step",
})
LAB.install()

P3 = ROOT / "results/phase3"
OUT = Path(os.environ.get("BATCH_A_OUT", P3 / "figures/batchA")).resolve()
W_IN = 7.16  # Use figure* and include at text width.

def stop(msg) -> NoReturn:
    print(f"STOP: {msg}", flush=True); sys.exit(1)

def need(ok, msg):
    if not ok: stop(msg)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def fmtp(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"

# ---------------- inputs and gates
F = {"metrics": P3 / "metrics/metrics.json", "clf": P3 / "classifiers/summary.json", "sp": P3 / "simprocess/summary.json",
     "stats": P3 / "statistics/statistics.json", "abl_tsv": P3 / "ablation/ablation.tsv", "abl_json": P3 / "ablation/ablation.json",
     "seam": P3 / "seam/seam.json", "lat": P3 / "latency/latency.json", "times": P3 / "latency/times_ms.npz",
     "stage3c": P3 / "stage3c.json", "rules": ROOT / "spec/reporting_rules.toml",
     "sp_manifest": ROOT / "spec/phase3_simprocess_manifest.sha256", "attacks_csv": ROOT / "spec/hai2305_command_attacks.csv"}
for k, p in F.items(): need(p.exists(), f"missing {p}")
H = {k: sha(p) for k, p in F.items()}
DEC = (ROOT / "spec/decisions.md").read_text()
J = {k: json.load(open(F[k])) for k in ("metrics", "clf", "sp", "stats", "abl_json", "seam", "lat")}
S, A = J["stats"], J["abl_json"]
for src, key, k in ((S, "results/phase3/metrics/metrics.json", "metrics"), (S, "results/phase3/classifiers/summary.json", "clf"),
                    (S, "spec/reporting_rules.toml", "rules"), (S, "results/phase3/stage3c.json", "stage3c"),
                    (S, "spec/phase3_simprocess_manifest.sha256", "sp_manifest"),
                    (A, "results/phase3/metrics/metrics.json", "metrics"), (A, "results/phase3/seam/seam.json", "seam"),
                    (A, "results/phase3/classifiers/summary.json", "clf")):
    need(src["inputs"].get(key) == H[k], f"{key}: sha256 differs from the one recorded by stage {src.get('stage')}")
need(H["sp"] in F["sp_manifest"].read_text(), "simprocess/summary.json not in the D47 manifest")

def recorded(k):
    if H[k] in DEC: return "spec/decisions.md"
    fj = F[k].parent / "files.json"
    if fj.exists() and H[k] in fj.read_text() and sha(fj) in DEC: return str(fj.relative_to(ROOT))
    return None
REC = {}
for k in ("stats", "abl_tsv", "abl_json", "lat", "times"):
    REC[k] = recorded(k)
    need(REC[k], f"{F[k].relative_to(ROOT)} sha256 {H[k][:12]} not recorded in spec/decisions.md or a recorded files.json")
cfg = set()
for k in ("metrics", "clf", "stats", "abl_json", "seam", "lat"):
    pv = J[k]["provenance"]
    need(pv.get("dirty") is False and pv.get("smoke") is False, f"{k}: provenance dirty={pv.get('dirty')} smoke={pv.get('smoke')}")
    if "config_sha12" in pv: cfg.add(pv["config_sha12"])
need(len(cfg) == 1, f"config hashes differ between result files: {cfg}")
print(f"   gates: inputs match statistics/ablation records, D47 manifest, decisions.md; provenance clean; config {cfg.pop()}")

RULES = tomllib.load(open(F["rules"], "rb"))
M = J["metrics"]["plants"]; C = J["clf"]["plants"]; SP = J["sp"]; L = J["lat"]["plants"]
T = np.load(F["times"])
S3C = F["stage3c"].read_text()
CMP = {c["id"]: c for c in S["comparisons"]}
H1 = [c for c in S["comparisons"] if c["hypothesis"] == "H1"]
FULL = H1[0]["compare"].split(" - ")[0]
METHODS = [FULL] + [c["compare"].split(" - ")[1] for c in H1]
for p in ("hai", "qappd"):
    need(set(S["extra"][f"m2_{p}"]) == set(METHODS), f"m2_{p} methods differ from H1 methods")
FIGS, SEL, VALS = {}, {}, {}

def keys_of(d, m):
    ks = [k for k in d if re.fullmatch(re.escape(m) + r"(_s\d+)?", k)]
    ks.sort(key=lambda k: int(k.rsplit("_s", 1)[1]) if re.search(r"_s\d+$", k) else 0)
    need(ks, f"no outputs for {m}")
    return ks

def rec(fig, panel, series, x, y, note=""):
    VALS.setdefault(fig, []).append((panel, series, x, repr(y) if isinstance(y, float) else y, note))

# ---------------- style
try:
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
except Exception as e:
    stop(f"Times New Roman not found by matplotlib: {e}")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                     "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
                     "axes.spines.top": True, "axes.spines.right": True,
                     "savefig.bbox": None, "savefig.facecolor": "white"})
TOL_MUTED = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]
TOL_BRIGHT = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
TOL_VIBRANT = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]
TOL_HC = ["#004488", "#DDAA33", "#BB5566", "#555555", "#88CCEE"]

def colours(names, pal, first=None):
    out, j = {}, 0
    for i, n in enumerate(names):
        if first is not None and i == 0: out[n] = first; continue
        out[n] = pal[j % len(pal)]; j += 1
    return out

def display(text):
    return str(text).replace("_", " ")

def panel_axes(fig, index):
    x = (0.115, 0.582)[index % 2]
    y = (0.615, 0.172)[index // 2]
    return fig.add_axes((x, y, 0.324, 0.30))

def colorbar(fig, image, ax, **kwargs):
    box = ax.get_position()
    cax = fig.add_axes((box.x1 + 0.008, box.y0, 0.011, box.height))
    return fig.colorbar(image, cax=cax, **kwargs)

def panel_title(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="center", pad=8, fontweight="bold")

def heat_axes(ax):
    ax.tick_params(length=0)
    ax.set_axisbelow(False)

def clean_grid(ax, axis="both"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#DFE3E8", linewidth=0.45)

def save(fig, name, *, out: Path | None = None):
    out = OUT if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    pdf, png, cv = out / f"{name}.pdf", out / f"{name}.png", out / f"{name}_values.csv"
    LAB.check(fig, name)
    fig.savefig(pdf); fig.savefig(png, dpi=600); plt.close(fig)
    b = pdf.read_bytes()
    need(b"TimesNewRoman" in b, f"{pdf.name}: Times New Roman not embedded")
    need(b"/Type3" not in b, f"{pdf.name}: Type 3 font present")
    with open(cv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["panel", "series", "x", "y", "note"]); w.writerows(VALS.get(name, []))
    FIGS[name] = {"pdf": sha(pdf), "png": sha(png), "values_csv": sha(cv), "n_values": len(VALS.get(name, []))}
    print(f"   wrote {name}.pdf / .png / _values.csv ({len(VALS.get(name, []))} values)")

def seed_mean(d, m, f):
    return float(np.mean([f(d[k]) for k in keys_of(d, m)]))

# ---------------- R1: H1, real attacks (HAI)
def fig_R1():
    n = "R1"; hai = M["hai"]; FO = hai["outputs"]["fit"]
    col = colours(METHODS, TOL_MUTED, first="#000000")
    att_list = list(RULES["ablation"]["m1_attacks"])
    need(sorted(att_list) == sorted(hai["attacks"]), "m1_attacks in reporting_rules differ from metrics.json attacks")
    Mv = np.array([[seed_mean(FO, m, lambda o, a=a: o["p_real_m1"][a]) for a in att_list] for m in METHODS])
    for c in H1:
        b = c["compare"].split(" - ")[1]
        for j, a in enumerate(att_list):
            u = c["units"][a]
            need(abs(u[0] - Mv[0, j]) < 1e-7 and abs(u[1] - Mv[METHODS.index(b), j]) < 1e-7,
                 f"M1 {a}: seed mean in metrics.json differs from statistics.json units of {c['id']}")
    print("   gate R1: per-attack M1 (seed means) equal statistics.json units for all H1 comparisons")
    fig = plt.figure(figsize=(W_IN, 6.5))
    # (a) response trace, example chosen by rule
    ax = panel_axes(fig, 0)
    best = min(H1, key=lambda c: c["median_diff"]); bm = best["compare"].split(" - ")[1]
    diffs = {a: best["units"][a][2] for a in att_list}
    med = float(np.median(list(diffs.values())))
    att = min(att_list, key=lambda a: (abs(diffs[a] - med), att_list.index(a)))
    D = P3 / "decoy/hai/p_real"
    for f in ("X.npy", "index.json"): need(sha(D / f) in S3C, f"{f}: sha256 not listed in stage3c.json")
    def npy(key):
        p = D / f"{key}.npy"; need(p.exists(), f"missing {p}")
        need(sha(p) in S3C, f"{p.name}: sha256 not listed in stage3c.json"); return np.load(p)
    idx = json.load(open(D / "index.json")); X = np.load(D / "X.npy")
    fk, bk = keys_of(FO, FULL), keys_of(FO, bm)
    DF = np.stack([npy(k) for k in fk]).astype(np.float64); DB = np.stack([npy(k) for k in bk]).astype(np.float64)
    nH, nr = DF.shape[2], DF.shape[3]; Wn = X.shape[1] - nH
    need(Wn > 0 and X.shape[0] == DF.shape[1] == DB.shape[1] == len(idx), "decoy / real / index shapes do not match")
    Y = X[:, Wn:, X.shape[2] - nr:].astype(np.float64)
    ii = [i for i, r in enumerate(idx) if r["attack"] == att]; need(ii, f"{att} not in index.json")
    def fits(y, yh):
        e = np.linalg.norm(y - yh)
        with np.errstate(all="ignore"):
            return {"global": 100 * (1 - e / np.linalg.norm(y - y.mean())),
                    "per_window_mean": 100 * (1 - e / np.linalg.norm(y - y.mean(axis=1, keepdims=True))),
                    "mean_of_windows": float(np.mean([100 * (1 - np.linalg.norm(a - b) / np.linalg.norm(a - a.mean())) for a, b in zip(y, yh)]))}
    tgt = {fk[0]: FO[fk[0]]["p_real_m1"][att], bk[0]: FO[bk[0]]["p_real_m1"][att]}
    hits = []
    for r in range(nr):
        ff, fb = fits(Y[ii, :, r], DF[0][ii, :, r]), fits(Y[ii, :, r], DB[0][ii, :, r])
        for var in ff:
            if abs(ff[var] - tgt[fk[0]]) < 1e-2 and abs(fb[var] - tgt[bk[0]]) < 1e-2: hits.append((r, var))
    need(len({h[0] for h in hits}) == 1, f"attacked response column of {att} not identified uniquely from stored M1: {hits}")
    r, var = hits[0]
    tag = {row["attack"]: row["response"] for row in csv.DictReader(open(F["attacks_csv"]))}.get(att, f"column {r}")
    print(f"   gate R1(a): {att} response column {r} ({tag}) reproduces stored M1 of {fk[0]} and {bk[0]} (FIT form '{var}')")
    i0 = min(ii, key=lambda i: idx[i]["tile"])
    dt = L["hai"]["period_ms"] / 1000.0
    t = (np.arange(X.shape[1]) - Wn) * dt; th = t[Wn:]
    real = X[i0, :, X.shape[2] - nr + r]
    ax.axvspan(t[0], 0, color="0.93", lw=0)
    ax.text(t[0] * 0.40, 0.97, "Warm-up\n(real)", transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="top", fontsize=8, color="0.35")
    ax.plot(t, real, color="0.2", lw=1.1, label="Real")
    for key, arr, ls, lab in ((FULL, DF, "--", FULL), (bm, DB, ":", bm)):
        trace_color = "#D55E00" if key == FULL else "#0072B2"
        tr = arr[:, i0, :, r]
        if tr.shape[0] > 1:
            band_color = "#009E73" if key == FULL else trace_color
            band_alpha = 0.55 if key == FULL else 0.22
            ax.fill_between(th, tr.min(0), tr.max(0), color=band_color, alpha=band_alpha, lw=0,
                            label="Seed range" if key == FULL else None)

        ax.plot(th, tr.mean(0), color=trace_color, lw=1.1, ls=ls, label=lab)
    ax.set_xlabel("Time from first free-run step (s)"); ax.set_ylabel(f"{display(tag)} (z)")
    ax.set_xlim(t[0], t[-1]); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=4, fontsize=8, columnspacing=0.60, handlelength=1.4)
    panel_title(ax, "a", "Response over time"); clean_grid(ax, "y")
    SEL["R1a"] = {"attack": att, "baseline": bm, "rule": "baseline with smallest H1 median difference; attack whose difference is closest to that median",
                  "window_index": i0, "tile": idx[i0]["tile"], "response_column": r, "tag": tag, "fit_form": var}
    for k in fk + bk: rec(n, "a", k, att, float(FO[k]["p_real_m1"][att]), "stored M1 of the plotted attack")
    # (b) heatmap M1 per attack x method
    ax = panel_axes(fig, 1)
    im = ax.imshow(Mv, cmap="PuOr", norm=mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100), aspect="auto")
    ooc = set(hai["out_of_coverage"])
    ax.set_xticks(range(len(att_list)), [a + ("†" if a in ooc else "") for a in att_list], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8); heat_axes(ax)
    cb = colorbar(fig, im, ax, extend="min"); cb.set_label("M1, FIT (%)", fontsize=8); cb.ax.tick_params(labelsize=7)
    for i, m in enumerate(METHODS):
        for j, a in enumerate(att_list): rec(n, "b", m, a, float(Mv[i, j]), "seed mean")
    panel_title(ax, "b", "FIT by attack and method")
    # (c) forest plot of H1 paired differences
    ax = panel_axes(fig, 2); ys = np.arange(len(H1))[::-1]
    trf = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, c in zip(ys, H1):
        b = c["compare"].split(" - ")[1]; lo, hi = c["ci95"]
        ax.plot([lo, hi], [y, y], color=col[b], lw=1.3)
        ax.plot(c["median_diff"], y, "o", ms=4.5, mec=col[b], mfc=col[b] if c["reject_h0"] else "white", mew=1.0)
        ax.text(1.02, y, fmtp(c["p_holm"]), transform=trf, fontsize=8, va="center")
        rec(n, "c", c["id"], "median_diff", float(c["median_diff"]), f"ci95 {lo} {hi}; p_holm {c['p_holm']}; reject {c['reject_h0']}")
    ax.text(1.02, ys[0] + 0.5, "Holm p", transform=trf, fontsize=8, va="center")
    ax.axvline(0, color="0.5", lw=0.6, ls=":")
    lt = 10 ** np.floor(np.log10(min(abs(c["median_diff"]) for c in H1)))
    ax.set_xscale("symlog", linthresh=lt)
    ax.set_yticks(ys, [c["compare"].split(" - ")[1] for c in H1], fontsize=8)
    ax.set_xlabel(f"ΔM1, {FULL} − baseline")
    panel_title(ax, "c", "Paired FIT differences"); clean_grid(ax, "y")
    # (d) per-seed M1
    ax = panel_axes(fig, 3)
    for x, m in enumerate(METHODS):
        v = [float(np.mean([FO[k]["p_real_m1"][a] for a in att_list])) for k in keys_of(FO, m)]
        off = np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(x + off, v, s=9, color=col[m], lw=0, zorder=3)
        ax.hlines(np.mean(v), x - 0.32, x + 0.32, color=col[m], lw=1.6, zorder=4)
        for k, vv in zip(keys_of(FO, m), v): rec(n, "d", m, k, vv, "M1 = mean over attacks")
    ax.axhline(0, color="0.5", lw=0.6, ls=":")
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=35, ha="right", fontsize=8); ax.set_ylabel("M1 per seed (FIT %)")
    panel_title(ax, "d", "FIT across seeds"); clean_grid(ax, "y")
    save(fig, n)

# ---------------- R2: H2, attacker classifiers
def fig_R2():
    n = "R2"; CL = [("rnn", "RNN"), ("mr", "MiniRocket"), ("sp", "SimProcess"), ("mean", "Mean")]
    H2 = [c for c in S["comparisons"] if c["hypothesis"] == "H2"]
    b1 = H2[0]["compare"].split(" - ")[1]
    col = {FULL: "#D55E00", b1: "#0072B2"}
    for p in ("hai", "qappd"):
        E = S["extra"][f"m2_{p}"]
        for m in METHODS:
            ks = keys_of(C[p]["outputs"], m)
            for c in ("rnn", "mr"):
                need(abs(E[m][c] - np.mean([C[p]["outputs"][k][c]["m2"] for k in ks])) < 1e-9,
                     f"M2 {p} {m} {c}: statistics.json differs from classifiers/summary.json")
            need(abs(E[m]["mean"] - np.mean([E[m][c] for c in ("rnn", "mr", "sp")])) < 1e-9, f"M2 mean {p} {m}")
            spk = [f"{p}/{k}" for k in ks]
            if all(k in SP for k in spk):
                d = abs(E[m]["sp"] - np.mean([SP[k]["m2"] for k in spk]))
                if d > 1e-9: print(f"   info R2: SimProcess M2 {p} {m} statistics vs summary differ by {d:.2e} (statistics value plotted)")
    print("   gate R2: RNN and MiniRocket M2 equal classifiers/summary.json; mean column consistent")
    fig = plt.figure(figsize=(W_IN, 6.5))
    axs = [panel_axes(fig, 0), panel_axes(fig, 1)]
    for ax, p, lt in zip(axs, ("hai", "qappd"), "ab"):
        E = S["extra"][f"m2_{p}"]
        V = np.array([[E[m][c] for c, _ in CL] for m in METHODS])
        im = ax.imshow(V, cmap="cividis", vmin=0, vmax=0.5, aspect="auto")
        for i in range(V.shape[0]):
            for k in range(V.shape[1]):
                ax.text(k, i, f"{V[i, k]:.2f}", ha="center", va="center", fontsize=8, color="white" if V[i, k] < 0.25 else "black")
                rec(n, lt, f"{p} {METHODS[i]}", CL[k][0], float(V[i, k]), "M2, mean over outputs and fits")
        ax.axvline(len(CL) - 1.5, color="white", lw=1.6)
        ax.set_xticks(range(len(CL)), [l for _, l in CL], rotation=32, ha="right", fontsize=8)
        ax.set_yticks(range(len(METHODS)), METHODS if p == "hai" else [], fontsize=8)
        panel_title(ax, lt, p.upper()); heat_axes(ax)
    cb = colorbar(fig, im, axs[1]); cb.set_label("M2 = |accuracy − 0.5|", fontsize=8); cb.ax.tick_params(labelsize=7)
    # (c) H2 difference
    ax = panel_axes(fig, 2); ys = np.arange(len(H2))[::-1]
    hi_all = max(c["ci95"][1] for c in H2); xmin = -0.45 * hi_all
    ax.axvspan(xmin, 0, color="0.92", lw=0)
    ax.text(xmin / 2, len(H2) - 0.55, "predicted", ha="center", va="top", fontsize=8, color="0.35")
    for y, c in zip(ys, H2):
        lo, hi = c["ci95"]
        ax.plot([lo, hi], [y, y], color=col[FULL], lw=1.3)
        ax.plot(c["median_diff"], y, "o", ms=4.5, mec=col[FULL], mfc=col[FULL] if c["reject_h0"] else "white", mew=1.0)
        ax.text(hi, y + 0.22, f"Holm p = {fmtp(c['p_holm'])}", fontsize=8, ha="right")
        rec(n, "c", c["id"], "median_diff", float(c["median_diff"]), f"ci95 {lo} {hi}; p_holm {c['p_holm']}")
    ax.axvline(0, color="0.5", lw=0.6, ls=":")
    ax.set_xlim(xmin, 1.1 * hi_all); ax.set_ylim(-0.6, len(H2) - 0.4)
    ax.set_yticks(ys, [c["plant"].upper() for c in H2]); ax.set_xlabel("Δ Detectability (Proposed − Replay)")
    panel_title(ax, "c", "Paired M2 differences"); clean_grid(ax, "y")
    # (d) beeswarm of RNN per-fit accuracy
    ax = panel_axes(fig, 3); groups = [(p, m) for p in ("hai", "qappd") for m in (FULL, b1)]
    for x, (p, m) in enumerate(groups):
        v = np.array([a for k in keys_of(C[p]["outputs"], m) for a in C[p]["outputs"][k]["rnn"]["acc_per_seed"]], float)
        span = (v.max() - v.min()) or 1.0; bw = span / 12
        bins = np.floor((v - v.min()) / bw).astype(int); off = np.zeros(len(v))
        for bi in np.unique(bins):
            ix = np.flatnonzero(bins == bi); ix = ix[np.argsort(v[ix])]
            off[ix] = (np.arange(len(ix)) - (len(ix) - 1) / 2) * min(0.36 / len(ix), 0.07)
        ax.scatter(x + off, v, s=8, color=col[m], lw=0, alpha=0.9, zorder=3)
        ax.hlines(np.median(v), x - 0.3, x + 0.3, color="0.15", lw=1.0, zorder=4)
        for k_, a_ in enumerate(v): rec(n, "d", f"{p} {m}", k_, float(a_), "RNN accuracy per fit")
        print(f"   R2(d): {p} {m}: {len(v)} RNN fits")
    ax.axhline(0.5, color="0.5", lw=0.6, ls=":"); ax.text(len(groups) - 0.5, 0.5, "chance", fontsize=8, va="bottom", ha="right", color="0.35")
    ax.set_xticks(range(len(groups)), [f"{p.upper()}\n{m}" for p, m in groups], fontsize=8)
    ax.set_ylabel("RNN accuracy per fit"); ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(True)
    panel_title(ax, "d", "RNN accuracy across fits"); clean_grid(ax, "y")
    save(fig, n)

# ---------------- R3: H3/H4, out-of-range probes
def fig_R3():
    n = "R3"
    HC = [c for c in S["comparisons"] if c["hypothesis"] in ("H3", "H4")]
    hai_c = [c for c in HC if c["plant"] == "hai"]; need(len(hai_c) == 3, f"expected 3 HAI H3/H4 tests, found {[c['id'] for c in hai_c]}")
    meths = []
    for c in HC:
        for m in c["compare"].split(" - "):
            if m not in meths: meths.append(m)
    if "B3" in M["hai"]["outputs"]["detector"] and "B3" not in meths: meths.append("B3")
    col = colours(meths, TOL_BRIGHT)
    DO = M["hai"]["outputs"]["detector"]
    for c in hai_c:
        ma, mb = c["compare"].split(" - ")
        for b, u in c["units"].items():
            for m, val in ((ma, u[0]), (mb, u[1])):
                need(abs(val - seed_mean(DO, m, lambda o: o["p_syn"]["oor"]["per_base"][b])) < 1e-9,
                     f"{c['id']} base {b} {m}: statistics.json differs from metrics.json per_base rate")
    print("   gate R3: H3/H4 HAI units equal seed-mean out-of-range rates per base window")
    fig = plt.figure(figsize=(W_IN, 6.5))
    lo_y = min(min(u[:2]) for c in hai_c for u in c["units"].values()); hi_y = max(max(u[:2]) for c in hai_c for u in c["units"].values())
    pad = 0.08 * ((hi_y - lo_y) or 1.0)
    for j, (c, lt) in enumerate(zip(hai_c, "abc")):
        ax = panel_axes(fig, j); ma, mb = c["compare"].split(" - ")
        U = np.array([u[:2] for u in c["units"].values()])
        for a, b in U: ax.plot([0, 1], [a, b], color="0.6", lw=0.6, alpha=0.6, zorder=1)
        ax.scatter(np.zeros(len(U)), U[:, 0], s=7, color=col[ma], lw=0, zorder=2)
        ax.scatter(np.ones(len(U)), U[:, 1], s=7, color=col[mb], lw=0, zorder=2)
        ax.plot([0, 1], np.median(U, axis=0), color="0.1", lw=1.8, zorder=3)
        ax.set_xticks([0, 1], [ma, mb], rotation=0, ha="center", fontsize=8); ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(lo_y - pad, hi_y + pad)
        summary = f"{c['n_nonzero']}/{c['n_units']} differ | Holm p {fmtp(c['p_holm'])}"
        panel_title(ax, lt, f"Paired detection rates\n{summary}")
        clean_grid(ax, "y")
        if j != 1: ax.set_ylabel("HAI out-of-range\ndetection rate")
        else: ax.set_yticklabels([])
        for b, u in c["units"].items(): rec(n, lt, c["id"], b, float(u[2]), f"{ma} {u[0]}; {mb} {u[1]}")
    ax = panel_axes(fig, 3)
    cols = [(p, i, rname) for p in ("hai", "qappd") for i, rname in enumerate(M[p]["rules"])]
    V = np.full((len(meths), len(cols)), np.nan)
    for i, m in enumerate(meths):
        for j, (p, ri, rname) in enumerate(cols):
            d = M[p]["outputs"]["detector"]
            for k in keys_of(d, m): need(len(d[k]["p_syn"]["oor"]["per_rule"]) == len(M[p]["rules"]), f"per_rule length {p} {k}")
            V[i, j] = seed_mean(d, m, lambda o, ri=ri: o["p_syn"]["oor"]["per_rule"][ri])
            rec(n, "d", m, f"{p} {rname}", float(V[i, j]), "seed mean out-of-range rate per rule")
    im = ax.imshow(V, cmap="YlOrBr", vmin=0, vmax=1, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if V[i, j] > 0.6 else "black")
    nh = len(M["hai"]["rules"]); ax.axvline(nh - 0.5, color="black", lw=1.2)
    ax.set_xticks(range(len(cols)), [f"{p.upper()}\n{display(r_)}" for p, _, r_ in cols], ha="right", rotation=45, fontsize=8)
    ax.set_yticks(range(len(meths)), meths, fontsize=8); heat_axes(ax)
    cb = colorbar(fig, im, ax); cb.set_label("Out-of-range\ndetection rate", fontsize=8); cb.ax.tick_params(labelsize=7)
    panel_title(ax, "d", "Detection rates by rule")
    save(fig, n)

# ---------------- R4: ablation
BETTER = {"metrics/fit/normal": 1, "metrics/fit/crosstime/pooled": 1, "metrics/fit/p_anom": 1, "metrics/cfid": -1,
          "metrics/dtw/normal": -1, "metrics/dtw/p_anom": -1, "metrics/detector/p_syn/in/rate": -1,
          "metrics/detector/p_syn/oor/rate": -1, "seam/p_syn_oor/interior": -1, "clf/mr/m2": -1, "clf/rnn/m2": -1,
          "simprocess/m2": -1, "simprocess/m2b": 1, "clf/tstr_mae": -1}
SHORT = {"metrics/fit/normal": "FIT normal", "metrics/fit/crosstime/pooled": "FIT cross", "metrics/fit/p_anom": "FIT P-anom",
         "metrics/cfid": "Context-FID", "metrics/dtw/normal": "DTW normal", "metrics/dtw/p_anom": "DTW P-anom",
         "metrics/detector/p_syn/in/rate": "Det. in-range", "metrics/detector/p_syn/oor/rate": "Det. OOR",
         "seam/p_syn_oor/interior": "Interior OOR", "clf/mr/m2": "M2 MR", "clf/rnn/m2": "M2 RNN",
         "simprocess/m2": "M2 SP", "simprocess/m2b": "M2b SP", "clf/tstr_mae": "TSTR MAE"}

def metric_label(metric):
    return {
        "metrics/fit/crosstime/pooled": "Cross FIT",
        "metrics/fit/p_anom": "FIT anom.",
        "metrics/dtw/p_anom": "DTW anom.",
        "metrics/detector/p_syn/oor/rate": "Out of range",
        "simprocess/m2b": "P(real)",
    }.get(metric, SHORT[metric])


def fig_R4():
    n = "R4"; labs = list(RULES["ablation"]["variants"]); VM = {v["label"]: v["method"] for v in A["variants"]}
    need(set(labs) == set(VM), f"variants differ: reporting_rules {labs} vs ablation.json {sorted(VM)}")
    rows = {}
    with open(F["abl_tsv"], newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"): rows[(r["plant"], r["variant"], r["metric"])] = r
    var = [v for v in labs if v != "Full"]; col = colours(labs, TOL_VIBRANT, first="#000000")
    fig = plt.figure(figsize=(W_IN, 6.5))
    axa, axb = panel_axes(fig, 0), panel_axes(fig, 1)
    for p, ax, lt in (("hai", axa, "a"), ("qappd", axb, "b")):
        cols = list(RULES["ablation"]["columns"][p])
        bad = [c for c in cols if c in RULES.get("exclude", {}).get(p, [])]; need(not bad, f"excluded metric in {p} columns: {bad}")
        miss = [c for c in cols if c not in BETTER]; need(not miss, f"no better-direction defined for {miss}")
        Dm = np.zeros((len(var), len(cols))); Ds = np.zeros_like(Dm)
        for i, v in enumerate(var):
            for j, c in enumerate(cols):
                r = rows.get((p, v, c))
                if r is None:
                    stop(f"ablation.tsv has no row {p} {v} {c}")
                need(r["method"] == VM[v], f"ablation.tsv {p} {v}: method {r['method']} != {VM[v]}")
                Dm[i, j], Ds[i, j] = float(r["delta_mean"]), float(r["delta_sd"])
        need(np.isfinite(Dm).all() and np.isfinite(Ds).all(), f"non-finite ablation values on {p}")
        mx = np.abs(Dm).max(axis=0); mx[mx == 0] = 1.0
        Z = Dm * np.array([BETTER[c] for c in cols]) / mx
        im = ax.imshow(Z, cmap="BrBG", vmin=-1, vmax=1, aspect="auto")
        for i in range(len(var)):
            for j in range(len(cols)):
                if abs(Dm[i, j]) <= Ds[i, j]: ax.plot(j, i, "o", ms=2.4, mfc="none", mec="0.2", mew=0.6)
                rec(n, lt, f"{p} {var[i]}", cols[j], float(Dm[i, j]), f"delta_sd {Ds[i, j]}; scaled {Z[i, j]:.4f}")
        ax.set_xticks(range(len(cols)), [metric_label(c) for c in cols], rotation=58, ha="right", fontsize=8)
        ax.set_yticks(range(len(var)), var, fontsize=8); heat_axes(ax)
        panel_title(ax, lt, p.upper())
    cb = colorbar(fig, im, axb)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.51, 0.975, "Δ vs Full, scaled per column (− worse, + better)",
             ha="center", va="center", fontsize=8)
    FO = M["hai"]["outputs"]["fit"]; DO = M["hai"]["outputs"]["detector"]; att = list(RULES["ablation"]["m1_attacks"])
    m1s = {v: [float(np.mean([FO[k]["p_real_m1"][a] for a in att])) for k in keys_of(FO, VM[v])] for v in labs}
    oor = {v: [float(DO[k]["p_syn"]["oor"]["rate"]) for k in keys_of(FO, VM[v])] for v in labs}
    ax = panel_axes(fig, 2)
    for x, v in enumerate(labs):
        y = m1s[v]; off = np.linspace(-0.18, 0.18, len(y)) if len(y) > 1 else np.zeros(1)
        ax.scatter(x + off, y, s=9, color=col[v], lw=0, zorder=3)
        ax.errorbar(x, np.mean(y), yerr=np.std(y, ddof=1) if len(y) > 1 else 0, fmt="_", ms=10, color=col[v], lw=1.0, capsize=2, zorder=4)
        for k, yy in zip(keys_of(FO, VM[v]), y): rec(n, "c", v, k, yy, "HAI M1 per seed")
    ax.axhline(0, color="0.5", lw=0.6, ls=":")
    variant_ticks = {
        "Full": "Proposed", "A1": "No\nphys.", "A2": "No\nfallback",
        "A3": "No\ncmd.", "A3-bare": "No cmd./\nlayers",
        "A4": "Limits\nonly", "A5": "No\nlayers",
    }
    tick_labels: list[str] = [variant_ticks.get(v, str(v)) for v in labs]
    ax.set_xticks(range(len(labs)), tick_labels, fontsize=8)
    ax.set_ylabel("HAI M1 (FIT %)")
    panel_title(ax, "c", "HAI FIT by variant"); clean_grid(ax, "y")
    ax = panel_axes(fig, 3)
    for v in labs:
        ax.scatter(m1s[v], oor[v], s=12, color=col[v], lw=0, alpha=0.9, label=v)
        for k, xx, yy in zip(keys_of(FO, VM[v]), m1s[v], oor[v]): rec(n, "d", v, xx, yy, f"{k}: x M1, y out-of-range rate")
    ax.set_xlabel("HAI M1 (FIT %)"); ax.set_ylabel("HAI out-of-range\ndetection rate")
    panel_title(ax, "d", "HAI FIT and detection rate"); clean_grid(ax, "y")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=col[v], ms=4) for v in labs], labels=[str(v) for v in labs],
               loc="upper center", bbox_to_anchor=(0.5, 0.09), ncol=4, frameon=False, fontsize=8)
    save(fig, n)

# ---------------- R5: latency and fidelity
def fig_R5():
    n = "R5"; meth = list(L["hai"]["outputs"]); need(set(meth) == set(L["qappd"]["outputs"]), "latency methods differ between plants")
    kinds = sorted({L[p]["outputs"][m]["kind"] for p in L for m in meth}); kc = {k: TOL_HC[i % len(TOL_HC)] for i, k in enumerate(kinds)}
    print(f"   R5 kinds: {kinds}")
    for p in ("hai", "qappd"):
        for m in meth:
            o = L[p]["outputs"][m]; v = T[f"{p}__{m}"]
            need(len(v) == o["timed_calls"], f"{p} {m}: {len(v)} timings vs timed_calls {o['timed_calls']}")
            s_ = v / json.load(open(P3 / "latency/latency.json"))["plants"][p]["H"] if o["kind"].startswith("window") else v  # 03d_latency.py line 316: window kinds store per-call times
            need(float(np.median(s_)) == o["median_ms"] and float(np.percentile(s_, 99)) == o["p99_ms"], f"{p} {m}: per-step median/p99 of times_ms.npz differ from latency.json")
            if o["kind"].startswith("window"): need(float(np.median(v)) == o["per_window_median_ms"], f"{p} {m}: per-window median differs from latency.json")
    print("   gate R5: per-step median and p99 of times_ms.npz equal latency.json exactly for all methods on both plants (window kinds / H, as 03d_latency.py)")
    fig = plt.figure(figsize=(W_IN, 6.5))
    for j, (p, lt) in enumerate((("hai", "a"), ("qappd", "b"))):
        ax = panel_axes(fig, j); O = L[p]["outputs"]
        step = sorted([m for m in meth if O[m]["kind"].startswith("step")], key=lambda m: O[m]["median_ms"])
        other = sorted([m for m in meth if not O[m]["kind"].startswith("step")], key=lambda m: O[m]["median_ms"])
        order = (step + other)[::-1]
        ps_ = [T[f"{p}__{m}"] / (L[p]["H"] if O[m]["kind"].startswith("window") else 1) for m in order]  # per step (window kinds / H), as the R5 gate
        for m, x_ in zip(order, ps_): need(float(np.median(x_)) == O[m]["median_ms"], f"{p} {m}: plotted median differs from latency.json")
        bp = ax.boxplot(ps_, vert=False, whis=(1, 99), showfliers=False, widths=0.6,
                        patch_artist=True, medianprops={"color": "black", "lw": 0.9}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
        for box, m in zip(bp["boxes"], order): box.set_facecolor(kc[O[m]["kind"]]); box.set_alpha(0.8); box.set_linewidth(0.5)
        ax.set_xscale("log")
        ax.set_yticks(range(1, len(order) + 1), [m + ("†" if m in other else "") for m in order], fontsize=8)
        if other: ax.axhline(len(other) + 0.5, color="0.5", lw=0.6, ls=":")
        ax.axvline(L[p]["period_ms"], color="#BB5566", lw=1.0, ls="--")
        ax.annotate(f"budget {L[p]['period_ms']:g} ms", (L[p]["period_ms"], 1),
                    xycoords=ax.get_xaxis_transform(), xytext=(-4, -5),
                    textcoords="offset points", fontsize=8, ha="right", va="top",
                    color="#BB5566")
        ax.set_xlabel("Latency per step (ms)")
        for m in order: rec(n, lt, f"{p} {m}", "median_ms", float(O[m]["median_ms"]), f"p99 {O[m]['p99_ms']}; kind {O[m]['kind']}")
        panel_title(ax, lt, p.upper()); clean_grid(ax, "x")
    markers = dict(zip(meth, ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*", "d", "H", "8", "1", "2", "3", "4"), strict=False))
    def pareto_panel(ax, p, yfun, ylab, lt):
        O = L[p]["outputs"]; FO = M[p]["outputs"]["fit"]
        xs, ys, es, ms = [], [], [], []
        for m in meth:
            v = [yfun(FO[k]) for k in keys_of(FO, m)]
            xs.append(O[m]["median_ms"]); ys.append(float(np.mean(v))); es.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0); ms.append(m)
            rec(n, lt, f"{p} {m}", O[m]["median_ms"], ys[-1], f"seed sd {es[-1]}; p99 {O[m]['p99_ms']}")
        xs, ys = np.array(xs), np.array(ys)
        for x, y, e, m in zip(xs, ys, es, ms):
            O_ = O[m]
            ax.errorbar(x, y, yerr=e, xerr=[[0], [O_["p99_ms"] - x]], fmt=markers[m], ms=4.5, color=kc[O_["kind"]], lw=0.7, capsize=1.5,
                        mfc=kc[O_["kind"]] if O_["kind"].startswith("step") else "white")
        st = [i for i, m in enumerate(ms) if O[m]["kind"].startswith("step")]
        o = sorted(st, key=lambda i: xs[i]); front, best = [], -np.inf
        for i in o:
            if ys[i] > best: front.append(i); best = ys[i]
        ax.step(xs[front], ys[front], where="post", color="0.3", lw=0.7, ls=":")
        ax.axvline(L[p]["period_ms"], color="#BB5566", lw=1.0, ls="--")
        ax.set_xscale("log"); ax.set_xlabel(f"{p.upper()} median latency (ms)"); ax.set_ylabel(ylab)
        SEL[f"R5{lt}_pareto"] = [ms[i] for i in front]
        panel_title(ax, lt, p.upper()); clean_grid(ax, "x")
    att = list(RULES["ablation"]["m1_attacks"])
    pareto_panel(panel_axes(fig, 2), "hai", lambda o: float(np.mean([o["p_real_m1"][a] for a in att])), "M1 (FIT %)", "c")
    pareto_panel(panel_axes(fig, 3), "qappd", lambda o: float(o["normal"]), "FIT normal (%)", "d")
    fig.legend(handles=[Patch(color=kc[k]) for k in kinds], labels=[display(k) for k in kinds],
               loc="upper center", bbox_to_anchor=(0.5, 0.105), ncol=len(kinds), frameon=False)
    fig.legend(handles=[Line2D([], [], marker=markers[m], ls="", color="0.25",
                               label=display(m) + ("†" if not L["hai"]["outputs"][m]["kind"].startswith("step") else ""))
                        for m in meth],
               loc="upper center", bbox_to_anchor=(0.5, 0.074), ncol=5, frameon=False, labelspacing=0.20)
    save(fig, n)

if __name__ == "__main__":
    for f in (fig_R1, fig_R2, fig_R3, fig_R4, fig_R5):
        LAB.begin(f.__name__.split("_", 1)[1]); f()
    LAB.report()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {"stage": "06_figures batch A", "repo_head": head, "script_sha256": sha(Path(__file__)),
           "script_committed": False, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
           "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256": H[k]} for k, p in F.items()},
           "recorded_in": REC, "selections": SEL, "figures": FIGS,
           "labels": {"file": "spec/figure_labels.toml", "sha256": sha(LAB.FILE), "module_sha256": sha(Path(LAB.__file__))}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"   R1(a) selection: {SEL['R1a']}")
    print(f"BATCH A DONE -> {OUT} ({len(FIGS)} figures; manifest sha256 {sha(OUT / 'manifest.json')[:12]})")
