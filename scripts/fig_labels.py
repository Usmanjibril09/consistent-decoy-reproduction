"""Display names for every figure (Batch A and B), from spec/figure_labels.toml.
install(): every matplotlib text takes its display name when it is set. begin(name): selects the figure.
check(fig, name): records codes left in text, text outside the figure, and labels running into another panel;
report(): lists every problem of the run and stops."""
import re, sys, tomllib
from pathlib import Path
import matplotlib.text as mtext

FILE = Path(__file__).resolve().parents[1] / "spec/figure_labels.toml"
_T = tomllib.load(open(FILE, "rb"))
EXACT = dict(_T.get("exact", {}))
_tabs = [set(_T["methods"]), set(_T["variants"]), set(_T["terms"])]
_dup = (_tabs[0] & _tabs[1]) | ((_tabs[0] | _tabs[1]) & _tabs[2])
if _dup: print(f"STOP: figure_labels.toml: keys in more than one table: {sorted(_dup)}"); sys.exit(1)
BASE = {**_T["methods"], **_T["variants"], **_T["terms"]}
_unset = sorted(k for k, v in BASE.items() if "CHECK" in v)
if _unset: print(f"STOP: figure_labels.toml: names still to confirm: {_unset}"); sys.exit(1)
CODE = re.compile(_T["gate"]["code_regex"])
_all = list(BASE.values()) + list(EXACT.values()) + [v for o in _T.get("override", {}).values() for v in o.values()]
_bad = sorted(v for v in _all if CODE.search(v))
if _bad: print(f"STOP: figure_labels.toml: display names contain codes: {_bad}"); sys.exit(1)
_CONV, _CUR, PROBLEMS = {}, [None], []
_ORIG = mtext.Text.set_text
_B = r"A-Za-z0-9_-"   # ASCII word boundary, so a code right after a symbol (e.g. "ΔM1") is still found

def names():
    """Method and variant names (these appear as tick labels)."""
    return set(_T["methods"].values()) | set(_T["variants"].values()) | {v for o in _T.get("override", {}).values() for v in o.values()}

def converter(name):
    if name not in _CONV:
        tok = {**BASE, **_T.get("override", {}).get(name, {})}
        pat = re.compile(rf"(?<![{_B}])(" + "|".join(re.escape(k) for k in sorted(tok, key=len, reverse=True)) + rf")(?![{_B}])")
        def conv(s):
            s = str(s)
            return EXACT[s] if s in EXACT else pat.sub(lambda m: tok[m.group(1)], s)
        _CONV[name] = conv
    return _CONV[name]

def begin(name):
    _CUR[0] = name

def install():
    if getattr(mtext.Text.set_text, "_fig_labels", False): return
    def set_text(self, s):
        if _CUR[0] is not None and s is not None: s = converter(_CUR[0])(s)
        return _ORIG(self, s)
    set_text._fig_labels = True
    mtext.Text.set_text = set_text

def _hit(a, b):
    return min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1

def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height
    ticklabels, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.majorTicks + axis.minorTicks: ticklabels.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks(): drawn.update((id(tk.label1), id(tk.label2)))
    left, cut, over = set(), set(), set()
    lgids = {id(x) for lg in list(fig.legends) + [a.get_legend() for a in fig.axes] if lg is not None for x in lg.get_texts()}
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible() or (id(t) in ticklabels and id(t) not in drawn): continue
        if CODE.search(s): left.add(s)
        b = t.get_window_extent(r)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1: cut.add(s + (" [legend]" if id(t) in lgids else " [axis text]"))
    boxes = [(ax, ax.get_window_extent(r)) for ax in fig.axes]
    for ax, _ in boxes:
        own = [t for axis in (ax.xaxis, ax.yaxis) for tk in axis.majorTicks + axis.minorTicks
               for t in (tk.label1, tk.label2) if id(t) in drawn]
        own += [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title]
        lg = ax.get_legend()
        for t in own + ([lg] if lg is not None else []):
            if not t.get_visible(): continue
            s = t.get_text() if isinstance(t, mtext.Text) else "legend"
            if not s.strip(): continue
            b = t.get_window_extent(r)
            if any(o is not ax and _hit(b, ob) for o, ob in boxes): over.add(s)
    if left: PROBLEMS.append(f"{name}: codes still in figure text: {sorted(left)}")
    if cut: PROBLEMS.append(f"{name}: text cut off at the figure edge: {sorted(cut)}")
    if over: PROBLEMS.append(f"{name}: text runs into another panel: {sorted(over)}")
    print(f"   labels {name}: " + ("OK, full names only, nothing cut off or overlapping" if not (left or cut or over) else "PROBLEM (listed at the end)"))

def report():
    if PROBLEMS:
        print("STOP: label problems (figures written for viewing only, not approved):")
        for p in PROBLEMS: print("   " + p)
        sys.exit(1)
    print("   labels: all figures use full names, nothing cut off or overlapping")
