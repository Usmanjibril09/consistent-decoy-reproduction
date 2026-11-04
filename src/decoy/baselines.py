"""Stage-1 baselines: B1 record-and-replay, B2 linear ARX (equation-error fit, free-run simulation)."""
from __future__ import annotations
import numpy as np


def b1_replay(tr, va, nc, ne, W, seed):
    """Each validation window receives the response part of a training window drawn at random; commands ignored."""
    pick = np.random.default_rng(seed).integers(0, len(tr), size=len(va))
    return tr[pick, W:, nc + ne:]


def _lagged(Y, U, na, nb, t0, L):
    parts = [Y[:, t0 - k:L - k] for k in range(1, na + 1)] + [U[:, t0 - k:L - k] for k in range(nb)]
    parts.append(np.ones(Y.shape[:1] + (L - t0, 1)))
    return np.concatenate(parts, -1)


def arx_fit(tr, nc, ne, na, nb, chunk=128):
    """y_t = sum_k A_k y_{t-k} (k=1..na) + sum_k B_k u_{t-k} (k=0..nb-1) + c, ridge 1e-6 * mean diagonal."""
    N, L, C = tr.shape
    ncu, nr = nc + ne, C - nc - ne
    t0 = max(na, nb - 1)
    F = nr * na + ncu * nb + 1
    XtX, XtY = np.zeros((F, F)), np.zeros((F, nr))
    for i in range(0, N, chunk):
        b = tr[i:i + chunk].astype(np.float64)
        U, Y = b[..., :ncu], b[..., ncu:]
        Phi = _lagged(Y, U, na, nb, t0, L).reshape(-1, F)
        Tg = Y[:, t0:].reshape(-1, nr)
        XtX += Phi.T @ Phi
        XtY += Phi.T @ Tg
    lam = 1e-6 * np.trace(XtX) / F
    return np.linalg.solve(XtX + lam * np.eye(F), XtY)


def arx_sim(theta, va, nc, ne, W, na, nb):
    """Free-run: real responses only in warm-up; afterwards past outputs are the model's own predictions."""
    N, L, C = va.shape
    ncu = nc + ne
    if W < max(na, nb - 1):
        raise ValueError("warm-up shorter than ARX orders")
    U = va[..., :ncu].astype(np.float64)
    Ys = va[..., ncu:].astype(np.float64).copy()
    one = np.ones((N, 1))
    with np.errstate(all="ignore"):
        for t in range(W, L):
            phi = np.concatenate([Ys[:, t - k] for k in range(1, na + 1)] + [U[:, t - k] for k in range(nb)] + [one], 1)
            Ys[:, t] = phi @ theta
    return Ys[:, W:]


def grey_sim(params, X, W, ncu):
    """Causal first-order-plus-delay model. No real future responses are used."""
    N, L, _ = X.shape
    out = np.empty((N, L-W, len(params)), dtype=np.float64)
    prev = X[:, W-1, ncu:].astype('float64').copy()
    for t in range(W, L):
        for j, p in enumerate(params):
            prev[:, j] = p['a'] * prev[:, j] + p['b'] * X[:, t-1-p['delay'], p['input']] + p['c']
        out[:, t-W] = prev
    return out.astype('float32')


def grey_fit(P, delays, max_eval=512):
    """Fit constrained equation error on all training windows, select input/delay on training free runs only."""
    ncu = P.nc + P.ne
    if max(delays) >= P.W:
        raise ValueError('B3 delay exceeds warm-up')
    ev = P.tr[np.unique(np.linspace(0, len(P.tr)-1, min(max_eval, len(P.tr))).astype(int))]
    params = []
    for j in range(P.nr):
        best = None
        for u in range(ncu):
            for d in delays:
                # Sufficient statistics in chunks, avoiding a huge least-squares design matrix.
                xx, xy = np.zeros((3,3)), np.zeros(3)
                for start in range(0, len(P.tr), 64):
                    X = P.tr[start:start+64].astype('float64'); t0=d+1
                    target = X[:,t0:,ncu+j].reshape(-1)
                    phi = np.stack([X[:,t0-1:-1,ncu+j].reshape(-1),
                                    X[:,:-t0,u].reshape(-1), np.ones_like(target)], 1)
                    xx += phi.T @ phi; xy += phi.T @ target
                coef = np.linalg.lstsq(xx, xy, rcond=1e-12)[0]
                a = float(np.clip(coef[0], 0, .999))
                bc = np.linalg.lstsq(xx[1:,1:], xy[1:] - a*xx[1:,0], rcond=1e-12)[0]
                p = dict(a=a,b=float(bc[0]),c=float(bc[1]),input=u,delay=d)
                prev = ev[:,P.W-1,ncu+j].astype('float64').copy(); err=0.
                for t in range(P.W, ev.shape[1]):
                    prev = a*prev + p['b']*ev[:,t-1-d,u]+p['c']
                    err += float(np.square(prev-ev[:,t,ncu+j]).sum())
                if best is None or err < best[0]: best=(err,p)
        params.append(best[1])
    return params
