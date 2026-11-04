"""Development-only CNN. Never reads held-out data; paired, purged temporal split."""
import numpy as np
import torch
from torch import nn


def partition(P):
    meta = P.info['window_index']['val']
    cut = int(.8 * len(meta))
    if cut < 2 or len(meta) - cut < 2:
        raise ValueError('K-dev needs at least two windows on each side')
    right = list(range(cut, len(meta)))
    first = {}
    for i in right:
        f, start = meta[i]
        first[f] = min(first.get(f, start), start)
    left = [i for i in range(cut) if meta[i][0] not in first or meta[i][1] + P.W + P.H <= first[meta[i][0]]]
    if len(left) < 2:
        raise ValueError('Not enough non-overlapping K-dev training windows')
    return np.array(left), np.array(right)


class CNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


def discriminate(P, generated, T, dev, seed=1729):
    real = P.va[:, P.W:]
    if generated.shape != real.shape or not np.isfinite(generated).all():
        raise ValueError('K-dev requires finite full-channel horizon windows')
    left, right = partition(P)
    # Fit classifier preprocessing only on its training partition, both classes equally.
    x = np.concatenate([real[left], generated[left]])
    mu, sd = x.mean((0, 1)), np.maximum(x.std((0, 1)), 1e-6)
    x = torch.from_numpy(((x - mu) / sd).astype('float32'))
    y = torch.cat([torch.ones(len(left)), torch.zeros(len(left))])
    xt = torch.from_numpy(((np.concatenate([real[right], generated[right]]) - mu) / sd).astype('float32'))
    yt = torch.cat([torch.ones(len(right)), torch.zeros(len(right))])
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(dev).startswith('cuda') else []):
        torch.manual_seed(seed)
        model = CNN(real.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=T['lr'])
        g = torch.Generator().manual_seed(seed)
        for epoch in range(T['epochs']):
            model.train()
            order = torch.randperm(len(x), generator=g)
            for idx in order.split(T['batch']):
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(x[idx].to(dev)), y[idx].to(dev))
                if not torch.isfinite(loss):
                    raise FloatingPointError('K-dev loss is non-finite')
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(b.to(dev)).cpu() for b in xt.split(T['batch'])])
        acc = float(((logits > 0) == yt.bool()).float().mean())
    return {'score': abs(acc - .5), 'accuracy': acc, 'train_pairs': len(left), 'eval_pairs': len(right),
            'purged': len(real) - len(left) - len(right), 'seed': seed, 'epochs': T['epochs']}
