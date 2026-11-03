"""C1 command-conditioned GRU (A3: same without commands). Training with resumable checkpoints (D28)."""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split(b, nc, ne, W):
    """b: (B, W+H, nc+ne+nr), column order [command, exogenous, response] -> warm, u, e, y (horizon)."""
    return b[:, :W], b[:, W:, :nc], b[:, W:, nc:nc + ne], b[:, W:, nc + ne:]


class Seq2SeqGRU(nn.Module):
    """Encoder reads the warm-up (commands, exogenous, real responses). Decoder runs over the horizon from commands and
    exogenous inputs only, so predictions never see real horizon responses (free-run by construction)."""

    def __init__(self, nc, ne, nr, hidden, layers, use_cmd=True):
        super().__init__()
        self.use_cmd, self.ne = bool(use_cmd), int(ne)
        n_dec = (nc if use_cmd else 0) + ne
        self.enc = nn.GRU(nc + ne + nr, hidden, layers, batch_first=True)
        self.dec = nn.GRU(max(n_dec, 1), hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, nr)

    def forward(self, warm, u, e):
        _, h = self.enc(warm)
        parts = ([u] if self.use_cmd else []) + ([e] if self.ne else [])
        x = torch.cat(parts, -1) if parts else u.new_ones(u.shape[0], u.shape[1], 1)
        return self.head(self.dec(x, h)[0])


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


@torch.no_grad()
def _val_loss(model, X, nc, ne, W, dev, bs=512):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        warm, u, e, y = split(torch.from_numpy(X[i:i + bs]).to(dev), nc, ne, W)
        tot += F.mse_loss(model(warm, u, e), y, reduction="sum").item()
        n += y.numel()
    return tot / n


@torch.no_grad()
def predict(model, X, nc, ne, W, dev, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        warm, u, e, _ = split(torch.from_numpy(X[i:i + bs]).to(dev), nc, ne, W)
        out.append(model(warm, u, e).cpu().numpy())
    return np.concatenate(out)


def train(model, tr, va, nc, ne, W, lr, batch, max_epochs, patience, seed, ckpt, dev):
    """MSE on the free-run horizon; early stopping on validation loss; checkpoint written atomically every epoch."""
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    st = {"epoch": 0, "best": float("inf"), "bad": 0, "resumed_from": None, "history": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ckpt.exists():
        c = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        g.set_state(c["gen"])
        torch.set_rng_state(c["rng"])
        best_state, st = c["best_state"], c["state"]
        st["resumed_from"] = st["epoch"]
    Xtr = torch.from_numpy(tr)
    n = len(Xtr)
    while st["epoch"] < max_epochs and st["bad"] < patience:
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            warm, u, e, y = split(Xtr[perm[i:i + batch]].to(dev), nc, ne, W)
            loss = F.mse_loss(model(warm, u, e), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {st['epoch']}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
        vl = _val_loss(model, va, nc, ne, W, dev)
        st["epoch"] += 1
        st["history"].append((tot / n, vl))
        if vl < st["best"] - 1e-7:
            st["best"], st["bad"] = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            st["bad"] += 1
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(),
                    "rng": torch.get_rng_state(), "best_state": best_state, "state": st}, tmp)
        os.replace(tmp, ckpt)
    model.load_state_dict(best_state)
    return st
