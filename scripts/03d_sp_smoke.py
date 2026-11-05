"""SimProcess smoke test (D45): validation data only; no held-out file, no decoy output."""
import sys, time, warnings, logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "third_party/simprocess_7533f59"))

WIN = {"hai": 20, "qappd": None}            # None = H (D43)
EPS = {"hai": 0.3, "qappd": float("inf")}   # inf = no filter: bounds become (-inf, inf)
N_SANITY = {"hai": 60, "qappd": 120}
N_DET = 16
# D45 full-run window counts (features needed): normal 320 x 78 outputs, P-syn x 77 outputs
COUNTS = {"hai": 320 * 78 + 280 * 77, "qappd": 320 * 78 + 382 * 77}
NORMAL_STORED = 320 * 78

def _init(cols):
    warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
    import simprocess.core as c, simprocess.features as f
    st = {"resp": list(cols)}; c.SIGNAL_TYPES = st; f.SIGNAL_TYPES = st

def feats(args):
    w, cols, win, eps = args
    import simprocess.data as sd, simprocess.features as sf
    df = pd.DataFrame(w, columns=cols)
    nw = max(win // 5, 3)
    for c in cols:   # identical call to upstream process_csv_files (Kalman defaults 1e-5 / 1e-1)
        n = sd.extract_noise_signal(df, filter_type="kalman", window_size=nw, cutoff=0.1, fs=1.0,
                                    poly_order=min(2, nw - 1), keep_noise_only=True, target_column=c)
        df[f"{c}_noise_raw"] = n[c].values
    f, _ = sf.extract_window_features(df, window_size=win, epsilon=eps)
    return f

def expected_rows(w, win, eps):  # independent re-implementation of upstream's sign-aware filter
    k = 0
    for i in range(0, len(w), win // 5):
        s = w[i:i + win]
        if len(s) < win: continue
        m = s.mean(0)
        lo = np.where(m >= 0, m * (1 - eps), m * (1 + eps)); hi = np.where(m >= 0, m * (1 + eps), m * (1 - eps))
        with np.errstate(invalid="ignore"):
            bad = ((s < lo) | (s > hi)).any()
        k += not bad
    return k

def run(windows, cols, win, eps, procs):
    t = time.time()
    with ProcessPoolExecutor(procs, initializer=_init, initargs=(cols,)) as ex:
        out = list(ex.map(feats, [(w, cols, win, eps) for w in windows], chunksize=1))
    return out, time.time() - t

def arr(frames, names):
    A, bad = [], 0
    for f in frames:
        a = (f.reindex(columns=names).to_numpy(np.float64) if len(f) else np.zeros((0, len(names))))
        bad += int((~np.isfinite(a)).sum()); A.append(np.nan_to_num(a, nan=0, posinf=0, neginf=0))
    return A, bad

def rf_acc(Ar, Ad, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, GroupKFold
    n = len(Ar); cut = int(.6 * n); tr, te = range(cut), range(cut, n)
    X, y, g = [], [], []
    for i in tr:
        for A, lab in ((Ar[i], 1), (Ad[i], 0)):
            X.append(A); y += [lab] * len(A); g += [i] * len(A)
    X = np.concatenate(X); t = time.time()
    gs = GridSearchCV(RandomForestClassifier(random_state=seed, n_jobs=-1), {"n_estimators": [50, 200]},
                      cv=GroupKFold(5), scoring="accuracy").fit(X, np.array(y), groups=np.array(g))
    fit_s = time.time() - t
    def score(A): return float(gs.predict_proba(A)[:, 1].mean()) if len(A) else 0.0   # zero-pass -> P(real)=0
    hits = [(score(Ar[i]) > .5) == 1 for i in te] + [(score(Ad[i]) > .5) == 0 for i in te]
    return float(np.mean(hits)), fit_s, len(X), gs.best_params_["n_estimators"]

def main():
    from decoy import data
    if data.cfg_hash() != "59a11b235108": sys.exit("STOP: config hash " + data.cfg_hash())
    cfg = data.load_cfg(); fails = []; proj = {}
    for pn in ("hai", "qappd"):
        P = data.load_plant(cfg, pn); ncu = P.nc + P.ne
        cols = list(P.info["columns"][ncu:])
        mu, sd = np.asarray(P.info["mu"])[ncu:], np.asarray(P.info["sd"])[ncu:]
        V = P.va[:, P.W:, ncu:].astype(np.float64) * sd + mu          # physical units, horizon, responses
        win = WIN[pn] or P.H; eps = EPS[pn]
        print(f"== {pn}: val {V.shape}, window {win}, stride {win // 5}, eps {eps}", flush=True)
        # S3 determinism + speed
        det = V[np.linspace(0, len(V) - 1, N_DET).astype(int)]
        ref, secs = None, {}
        for p in (1, 4, 8):
            fr, s = run(det, cols, win, eps, p); secs[p] = s / len(det)
            a = np.concatenate([f.to_numpy(np.float64) for f in fr if len(f)])
            same = ref is None or (a.shape == ref.shape and np.array_equal(a, ref, equal_nan=True))
            print(f"   S3 procs {p}: {secs[p]:.3f} s/window  identical to 1 proc: {same}", flush=True)
            if not same: fails.append(f"{pn} procs {p} features differ")
            ref = a if ref is None else ref
        # S2 filter count
        rows = [len(f) for f in fr]; exp = [expected_rows(w, win, eps) for w in det]
        print(f"   S2 rows per window upstream {rows[:8]}... independent {exp[:8]}... match {rows == exp}", flush=True)
        if rows != exp: fails.append(f"{pn} filter count mismatch")
        # S4 sanity
        R = V[np.linspace(0, len(V) - 1, N_SANITY[pn]).astype(int)]
        rng = np.random.default_rng(7)
        noisy = R + rng.normal(0, 1, R.shape) * np.diff(V, axis=1).std((0, 1))
        fr_r, _ = run(R, cols, win, eps, 4); fr_n, t4 = run(noisy, cols, win, eps, 4)
        names = list(next(f for f in fr_r if len(f)).columns)
        Ar, b1 = arr(fr_r, names); An, b2 = arr(fr_n, names)
        pr, pn_ = np.mean([len(a) for a in Ar]), np.mean([len(a) for a in An])
        acc_n, fit_s, nrows, ne = rf_acc(Ar, An, 100)
        perm = rng.permutation(len(Ar)); h = len(Ar) // 2
        acc_r, _, _, _ = rf_acc([Ar[i] for i in perm[:h]], [Ar[i] for i in perm[h:2 * h]], 100)
        print(f"   S4 features {len(names)} | rows/window real {pr:.1f} noisy {pn_:.1f} | non-finite set to 0: {b1 + b2}", flush=True)
        print(f"   S4 real vs noisy accuracy {acc_n:.3f} (need >= 0.9) | real vs real {acc_r:.3f} (need 0.35-0.65)"
              f" | RF grid fit {fit_s:.1f} s on {nrows} rows (best n_estimators {ne})", flush=True)
        if acc_n < .9: fails.append(f"{pn} real-vs-noisy {acc_n:.3f}")
        if not .35 <= acc_r <= .65: fails.append(f"{pn} real-vs-real {acc_r:.3f}")
        # S7 projection
        rpw = max(pr, 1.0)
        gb = NORMAL_STORED * rpw * len(names) * 4 / 1e9
        rf_rows_full = 2 * 160 * rpw; rf_h = 77 * 5 * fit_s * max(rf_rows_full / nrows, 1) / 3600
        for p in (4, 8):
            hrs = COUNTS[pn] * secs[p] / 3600
            print(f"   S7 procs {p}: features {hrs:.1f} h | RF about {rf_h:.1f} h | features stored {gb:.2f} GB", flush=True)
        proj[pn] = (secs, gb, rf_h)
    tot = {p: sum(COUNTS[k] * proj[k][0][p] / 3600 + proj[k][2] for k in proj) for p in (4, 8)}
    gb = sum(v[1] for v in proj.values())
    print(f"== TOTAL: procs 4 {tot[4]:.1f} h | procs 8 {tot[8]:.1f} h | disk {gb:.2f} GB (cap 15)", flush=True)
    if gb > 15: fails.append(f"disk {gb:.1f} GB > 15")
    rec = 8 if tot[8] <= 0.85 * tot[4] else 4
    print(f"   recommended processes: {rec}")
    if fails:
        print("SMOKE FAILED:"); [print("  ", f) for f in fails]; sys.exit(1)
    print("SMOKE SIMPROCESS PASSED")

if __name__ == "__main__":
    main()
