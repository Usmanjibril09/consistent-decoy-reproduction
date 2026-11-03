#!/usr/bin/env python3
"""Read-only inventory of the raw datasets. Never writes inside the dataset folder."""
import gzip, hashlib, io, os, sys, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/ujw/Desktop/Dataset")
OUT = Path(__file__).resolve().parents[1] / "spec" / "data_inventory.txt"
lines = []

def w(s=""):
    print(s); lines.append(s)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def opener(p):
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", errors="replace")
    return open(p, "r", errors="replace")

def count_lines(p):
    with opener(p) as f:
        return sum(1 for _ in f)

def show_text(p, n=20):
    with opener(p) as f:
        for _ in range(n):
            s = f.readline()
            if not s:
                break
            w("      " + s.rstrip())

def peek_table(p):
    try:
        with opener(p) as f:
            head = "".join(f.readline() for _ in range(6))
        df = pd.read_csv(io.StringIO(head), sep=None, engine="python")
    except Exception as e:
        w(f"    could not parse as table: {e}")
        show_text(p, 5)
        return
    w(f"    columns ({df.shape[1]}): {list(df.columns)}")
    w(f"    lines incl. header: {count_lines(p)}")
    w("    first rows:")
    for s in df.head(3).to_string(max_cols=12).splitlines():
        w("      " + s)

if not ROOT.is_dir():
    sys.exit(f"STOP: dataset folder not found: {ROOT}")

w(f"dataset root: {ROOT}")
for d, subdirs, files in os.walk(ROOT):
    subdirs[:] = sorted(s for s in subdirs if not s.startswith("."))
    for name in sorted(files):
        if name.startswith("."):
            continue
        p = Path(d) / name
        rel = p.relative_to(ROOT)
        w(f"\n== {rel}  ({p.stat().st_size / 1e6:.2f} MB)")
        w(f"    sha256: {sha256(p)}")
        low = name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                w(f"    zip members ({len(names)}): {names[:30]}")
        elif "readme" in low or low.endswith(".md"):
            show_text(p)
        elif low.endswith((".csv", ".csv.gz", ".tsv", ".txt")):
            peek_table(p)
        else:
            w("    (not inspected: unknown type)")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
