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
