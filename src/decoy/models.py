"""Learned models (D28, D29). A model maps a window b (B, W+H, nc+ne+nr), column order [command, exogenous, response],
to a horizon window (B, H, nc+ne+nr). Conditional models copy the real commands/exogenous inputs and predict responses;
unconditioned generators produce every channel. Real horizon responses are never model inputs at prediction time."""
from __future__ import annotations
import math
import os
import time

DEADLINE = None

def expired():
    return DEADLINE is not None and time.monotonic() >= DEADLINE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split(b, nc, ne, W):
    return b[:, :W], b[:, W:, :nc], b[:, W:, nc:nc + ne], b[:, W:, nc + ne:]


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def _bce(logits, v):
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, float(v)))


def _params(*mods):
    return [p for m in mods for p in m.parameters()]


class _Base(nn.Module):
    conditional = True
    adversarial = False
    n_phases = 1

    def _dims(self, d):
        self.nc, self.ne, self.nr, self.W = int(d["nc"]), int(d["ne"]), int(d["nr"]), int(d["W"])
        self.ncu, self.C = self.nc + self.ne, self.nc + self.ne + self.nr

    def window(self, b, g):
        return torch.cat([b[:, self.W:, :self.ncu], self.respond(b, g)], -1)


class Seq2SeqGRU(_Base):
    """C1 (A3 with use_cmd=False). Encoder reads the warm-up; decoder runs from commands/exogenous inputs only."""

    def __init__(self, d, hidden, layers, use_cmd=True):
        super().__init__()
        self._dims(d)
        self.use_cmd = bool(use_cmd)
        n_dec = (self.nc if use_cmd else 0) + self.ne
        self.enc = nn.GRU(self.C, hidden, layers, batch_first=True)
        self.dec = nn.GRU(max(n_dec, 1), hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, self.nr)

    def forward(self, warm, u, e):
        _, h = self.enc(warm)
        parts = ([u] if self.use_cmd else []) + ([e] if self.ne else [])
        x = torch.cat(parts, -1) if parts else u.new_ones(u.shape[0], u.shape[1], 1)
        return self.head(self.dec(x, h)[0])

    def respond(self, b, g=None):
        warm, u, e, _ = split(b, self.nc, self.ne, self.W)
        return self(warm, u, e)



def _base_loss(self, b):
    return F.mse_loss(self.respond(b), b[:, self.W:, self.ncu:])


_Base.loss = _base_loss


class CausalTransformer(_Base):
    """B8: causal Transformer over the whole window. Horizon responses are zeroed (flag=0), so no real response leaks in;
    the causal mask means no future commands are seen."""

    def __init__(self, d, d_model, layers):
        super().__init__()
        self._dims(d)
        self.inp = nn.Linear(self.C + 1, d_model)
        pos = torch.arange(2048).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(2048, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, 4, 2 * d_model, 0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, self.nr)

    def respond(self, b, g=None):
        B, L, _ = b.shape
        y = b[..., self.ncu:].clone()
        y[:, self.W:] = 0
        flag = torch.zeros(B, L, 1, device=b.device)
        flag[:, :self.W] = 1
        h = self.inp(torch.cat([b[..., :self.ncu], y, flag], -1)) + self.pe[:L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=b.device), 1)
        return self.head(self.enc(h, mask=mask)[:, self.W:])


class NeuPotSeq2Seq(_Base):
    """B9: LSTM seq2seq, Huber loss (delta 1), teacher forcing in training, autoregressive free-run at prediction (D29(6))."""

    def __init__(self, d, hidden, layers):
        super().__init__()
        self._dims(d)
        self.enc = nn.LSTM(self.C, hidden, layers, batch_first=True)
        self.dec = nn.LSTM(self.ncu + self.nr, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, self.nr)

    def _run(self, b, teacher):
        warm, _, _, y = split(b, self.nc, self.ne, self.W)
        _, hc = self.enc(warm)
        ue = b[:, self.W:, :self.ncu]
        if teacher:
            prev = torch.cat([warm[:, -1:, self.ncu:], y[:, :-1]], 1)
            return self.head(self.dec(torch.cat([ue, prev], -1), hc)[0])
        prev, out = warm[:, -1:, self.ncu:], []
        for t in range(ue.shape[1]):
            o, hc = self.dec(torch.cat([ue[:, t:t + 1], prev], -1), hc)
            prev = self.head(o)
            out.append(prev)
        return torch.cat(out, 1)

    def respond(self, b, g=None):
        return self._run(b, False)

    def loss(self, b):
        return F.huber_loss(self._run(b, True), b[:, self.W:, self.ncu:], delta=1.0)


def build(method, P, hp):
    d = {"nc": P.nc, "ne": P.ne, "nr": P.nr, "W": P.W}
    if method == "B4": return UnconditionalLSTM(d, hp["hidden"], hp["layers"])
    if method == "B5": return RCGAN(d, hp["hidden"], hp["noise"])
    if method == "B6": return LatentGAN(d, hp["latent"], hp["hidden"], P.H)
    if method == "B7": return TimeGAN(d, hp["hidden"], hp["layers"])
    if method in ("C1", "A3"):
        return Seq2SeqGRU(d, hp["hidden"], hp["layers"], use_cmd=(method == "C1"))
    if method == "B8":
        return CausalTransformer(d, hp["d_model"], hp["layers"])
    if method == "B9":
        return NeuPotSeq2Seq(d, hp["hidden"], hp["layers"])
    raise SystemExit(f"STOP: no model for {method}")


@torch.no_grad()
def _val_loss(model, X, dev, bs=32):
    """Free-run validation MSE for every model (selection criterion consistent with D28)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        b = torch.from_numpy(X[i:i + bs]).to(dev)
        y = b[:, model.W:, model.ncu:]
        tot += F.mse_loss(model.respond(b), y, reduction="sum").item()
        n += y.numel()
    if n == 0 or not math.isfinite(tot):
        raise FloatingPointError("Empty or non-finite validation loss; no checkpoint saved")
    return tot / n


@torch.no_grad()
def predict(model, X, dev, bs=32):
    model.eval()
    return np.concatenate([model.respond(torch.from_numpy(X[i:i + bs]).to(dev)).cpu().numpy() for i in range(0, len(X), bs)])


def train(model, tr, va, lr, batch, micro, max_epochs, patience, seed, ckpt, dev):
    """model.loss on training batches (micro-batches accumulated); early stopping on free-run val loss; atomic checkpoint per epoch."""
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("Training and validation windows must be nonempty")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    st = {"epoch": 0, "best": float("inf"), "bad": 0, "resumed_from": None, "history": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        g.set_state(c["gen"]); torch.set_rng_state(c["rng"])
        if str(dev).startswith("cuda") and c.get("cuda") is not None:
            torch.cuda.set_rng_state_all(c["cuda"])
        best_state, st = c["best_state"], c["state"]
        st["resumed_from"] = st["epoch"]
    Xtr = torch.from_numpy(tr)
    n, micro = len(Xtr), max(1, min(int(micro), int(batch)))
    while st["epoch"] < max_epochs and st["bad"] < patience:
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            tl = 0.0
            for j in range(0, len(idx), micro):
                bb = Xtr[idx[j:j + micro]].to(dev)
                loss = model.loss(bb) * (len(bb) / len(idx))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {st['epoch']}")
                loss.backward()
                tl += loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            tot += tl * len(idx)
        vl = _val_loss(model, va, dev)
        st["epoch"] += 1
        st["history"].append((tot / n, vl))
        if vl < st["best"] - 1e-7:
            st["best"], st["bad"] = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            st["bad"] += 1
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(),
                    "rng": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if str(dev).startswith("cuda") else None, "best_state": best_state, "state": st}, tmp)
        os.replace(tmp, ckpt)
        if expired(): raise TimeoutError("Time budget reached after saving epoch checkpoint")
    if not math.isfinite(st["best"]):
        raise FloatingPointError("No valid validation checkpoint available")
    model.load_state_dict(best_state)
    return st


class UnconditionalLSTM(_Base):
    """B4: autoregressive joint command/response LSTM; exogenous signals are replayed separately."""
    conditional = False
    def __init__(self, d, hidden, layers):
        super().__init__(); self._dims(d)
        self.rnn = nn.LSTM(self.nc+self.nr, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, self.nc+self.nr)
        self.register_buffer('noise_sd', torch.zeros(self.nc+self.nr))
    def values(self, b):
        return torch.cat([b[...,:self.nc], b[...,self.ncu:]], -1)
    def loss(self, b):
        x=self.values(b)
        return F.mse_loss(self.head(self.rnn(x[:,:-1])[0]), x[:,1:])
    def window(self, b, g=None):
        x=self.values(b); o,h=self.rnn(x[:,:self.W]); prev=self.head(o[:,-1:]); out=[]
        for t in range(b.shape[1]-self.W):
            if g is not None:
                prev=prev+torch.randn(prev.shape, generator=g, device=b.device)*self.noise_sd
            out.append(prev)
            if t+1 < b.shape[1]-self.W:
                o,h=self.rnn(prev,h);prev=self.head(o)
        z=torch.cat(out,1)
        return torch.cat([z[...,:self.nc], b[:,self.W:,self.nc:self.ncu], z[...,self.nc:]],-1)
    def respond(self,b,g=None): return self.window(b,g)[...,self.ncu:]
    @torch.no_grad()
    def fit_noise(self, tr, dev):
        sums=torch.zeros_like(self.noise_sd); squares=sums.clone(); n=0
        self.eval()
        for i in range(0,len(tr),32):
            x=self.values(torch.from_numpy(tr[i:i+32]).to(dev))
            e=x[:,1:]-self.head(self.rnn(x[:,:-1])[0])
            sums+=e.sum((0,1));squares+=e.square().sum((0,1));n+=e.shape[0]*e.shape[1]
        self.noise_sd.copy_((squares/n-(sums/n).square()).clamp_min(0).sqrt())


class RCGAN(_Base):
    adversarial=True
    def __init__(self,d,hidden,noise):
        super().__init__();self._dims(d);self.nz=noise
        self.enc=nn.LSTM(self.C,hidden,batch_first=True)
        self.gen=nn.LSTM(self.ncu+noise,hidden,batch_first=True)
        self.head=nn.Linear(hidden,self.nr)
        self.dis=nn.LSTM(self.C,hidden,batch_first=True)
        self.dhead=nn.Linear(hidden,1)
    def respond(self,b,g=None):
        _,h=self.enc(b[:,:self.W])
        z=torch.randn((len(b),b.shape[1]-self.W,self.nz),device=b.device,generator=g)
        x=torch.cat([b[:,self.W:,:self.ncu],z],-1)
        return self.head(self.gen(x,h)[0])
    def judge(self,x):return self.dhead(self.dis(x)[0])
    def optimizers(self,lr):
        return {'g':torch.optim.Adam(_params(self.enc,self.gen,self.head),lr=lr),
                'd':torch.optim.Adam(_params(self.dis,self.dhead),lr=lr)}
    def step(self,b,phase,opts,g):
        real=b[:,self.W:];fake=self.window(b,g)
        dl=_bce(self.judge(real),1)+_bce(self.judge(fake.detach()),0)
        update(dl,opts['d'])
        gl=_bce(self.judge(self.window(b,g)),1);update(gl,opts['g'])
        return {'d':float(dl.detach()),'g':float(gl.detach())}


class LatentGAN(_Base):
    """B6 GAN-through-autoencoder adaptation; NOT an exact NeuralPot reproduction."""
    adversarial=True;conditional=False;n_phases=2
    def __init__(self,d,latent,hidden,horizon):
        super().__init__();self._dims(d);self.H=horizon;self.nz=latent
        n=(self.nc+self.nr)*horizon
        self.enc=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Linear(hidden,latent))
        self.dec=nn.Sequential(nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,n))
        self.gen=nn.Sequential(nn.Linear(latent,hidden),nn.LeakyReLU(.2),nn.Linear(hidden,latent))
        self.dis=nn.Sequential(nn.Linear(latent,hidden),nn.LeakyReLU(.2),nn.Linear(hidden,1))
    def values(self,b):return torch.cat([b[:,self.W:,:self.nc],b[:,self.W:,self.ncu:]],-1)
    def window(self,b,g=None):
        z=torch.randn((len(b),self.nz),device=b.device,generator=g)
        x=self.dec(self.gen(z)).reshape(len(b),self.H,self.nc+self.nr)
        return torch.cat([x[...,:self.nc],b[:,self.W:,self.nc:self.ncu],x[...,self.nc:]],-1)
    def respond(self,b,g=None):return self.window(b,g)[...,self.ncu:]
    def optimizers(self,lr):
        return {'ae':torch.optim.Adam(_params(self.enc,self.dec),lr=lr),
                'g':torch.optim.Adam(self.gen.parameters(),lr=lr),'d':torch.optim.Adam(self.dis.parameters(),lr=lr)}
    def step(self,b,phase,opts,g):
        x=self.values(b).flatten(1)
        if phase==0:
            loss=F.mse_loss(self.dec(self.enc(x)),x);update(loss,opts['ae']);return {'ae':float(loss.detach())}
        h=self.enc(x).detach()
        z=torch.randn(h.shape,device=b.device,generator=g)
        dl=_bce(self.dis(h),1)+_bce(self.dis(self.gen(z).detach()),0);update(dl,opts['d'])
        z=torch.randn(h.shape,device=b.device,generator=g)
        gl=_bce(self.dis(self.gen(z)),1);update(gl,opts['g'])
        return {'d':float(dl.detach()),'g':float(gl.detach())}


class SequenceNet(nn.Module):
    def __init__(self,ni,hidden,no,layers,sigmoid=True):
        super().__init__();self.rnn=nn.GRU(ni,hidden,layers,batch_first=True);self.head=nn.Linear(hidden,no);self.sigmoid=sigmoid
    def forward(self,x):
        x=self.head(self.rnn(x)[0]);return x.sigmoid() if self.sigmoid else x


class TimeGAN(_Base):
    """B7: PyTorch implementation of the original three-phase objective (Yoon et al. 2019)."""
    adversarial=True;conditional=False;n_phases=3
    def __init__(self,d,hidden,layers):
        super().__init__();self._dims(d);n=self.nc+self.nr;self.nz=n
        self.embed=SequenceNet(n,hidden,hidden,layers)
        self.recover=SequenceNet(hidden,hidden,n,layers)
        self.gen=SequenceNet(n,hidden,hidden,layers)
        self.supervisor=SequenceNet(hidden,hidden,hidden,layers-1)
        self.dis=SequenceNet(hidden,hidden,1,layers,False)
        self.register_buffer('lo',torch.zeros(n));self.register_buffer('span',torch.ones(n))
    def values(self,b):return torch.cat([b[:,self.W:,:self.nc],b[:,self.W:,self.ncu:]],-1)
    @torch.no_grad()
    def fit_scale(self,tr):
        x=np.concatenate([tr[:,self.W:,:self.nc],tr[:,self.W:,self.ncu:]],-1)
        self.lo.copy_(torch.as_tensor(x.min((0,1)),device=self.lo.device))
        self.span.copy_(torch.as_tensor(np.maximum(x.max((0,1))-x.min((0,1)),1e-7),device=self.lo.device))
    def noise(self,b,g):return torch.rand((len(b),b.shape[1]-self.W,self.nz),device=b.device,generator=g)
    def window(self,b,g=None):
        x=self.recover(self.supervisor(self.gen(self.noise(b,g))))*self.span+self.lo
        return torch.cat([x[...,:self.nc],b[:,self.W:,self.nc:self.ncu],x[...,self.nc:]],-1)
    def respond(self,b,g=None):return self.window(b,g)[...,self.ncu:]
    def optimizers(self,lr):
        return {k:torch.optim.Adam(p,lr=lr) for k,p in {
          'e0':_params(self.embed,self.recover),'e':_params(self.embed,self.recover),
          's':_params(self.gen,self.supervisor),'g':_params(self.gen,self.supervisor),'d':list(self.dis.parameters())}.items()}
    def step(self,b,phase,opts,g):
        x=(self.values(b)-self.lo)/self.span
        if phase==0:
            loss=10*F.mse_loss(self.recover(self.embed(x)),x).clamp_min(1e-12).sqrt()
            update(loss,opts['e0']);return {'embedding':float(loss.detach())}
        if phase==1:
            h=self.embed(x).detach()
            loss=F.mse_loss(self.supervisor(h)[:,:-1],h[:,1:]);update(loss,opts['s'])
            return {'supervised':float(loss.detach())}
        # Two G/E updates for each D update, gamma=1, moment and supervised weights=100.
        for _ in range(2):
            h=self.embed(x).detach();e=self.gen(self.noise(b,g));hh=self.supervisor(e);xh=self.recover(hh)
            sl=F.mse_loss(self.supervisor(h)[:,:-1],h[:,1:])
            moment=(xh.mean(0)-x.mean(0)).abs().mean()+((xh.var(0,unbiased=False)+1e-6).sqrt()-(x.var(0,unbiased=False)+1e-6).sqrt()).abs().mean()
            gl=_bce(self.dis(hh),1)+_bce(self.dis(e),1)+100*sl.clamp_min(1e-12).sqrt()+100*moment
            update(gl,opts['g'])
            h=self.embed(x)
            el=10*F.mse_loss(self.recover(h),x).clamp_min(1e-12).sqrt()+.1*F.mse_loss(self.supervisor(h)[:,:-1],h[:,1:])
            update(el,opts['e'])
        h=self.embed(x).detach();e=self.gen(self.noise(b,g)).detach();hh=self.supervisor(e).detach()
        dl=_bce(self.dis(h),1)+_bce(self.dis(hh),0)+_bce(self.dis(e),0)
        if float(dl.detach())>.15:update(dl,opts['d'])
        return {'g':float(gl.detach()),'e':float(el.detach()),'d':float(dl.detach())}


def update(loss,opt):
    if not torch.isfinite(loss): raise FloatingPointError('Non-finite GAN objective')
    opt.zero_grad();loss.backward()
    for group in opt.param_groups:
        for p in group['params']:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError('Non-finite GAN gradient')
    opt.step()


def save_atomic(payload,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def train_gan(model,tr,lr,batch,iters,seed,ckpt,dev,every=200,stop_after=None):
    """Resume all optimizer, phase, sampling and CPU/CUDA RNG state; final iterate, not validation selection."""
    opts=model.optimizers(lr)
    g=torch.Generator(device=dev).manual_seed(seed+10000)
    cpu=torch.Generator().manual_seed(seed+20000)
    st={'step':0,'history':[],'resumed_from':None}
    if ckpt.exists():
        c=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(c['model'])
        for key in opts:opts[key].load_state_dict(c['optimizers'][key])
        g.set_state(c['noise']);cpu.set_state(c['sample']);torch.set_rng_state(c['rng'])
        if str(dev).startswith('cuda') and c.get('cuda') is not None:torch.cuda.set_rng_state_all(c['cuda'])
        st=c['state'];st['resumed_from']=st['step']
    elif hasattr(model,'fit_scale'):model.fit_scale(tr)
    total=model.n_phases*iters
    end=min(total,stop_after) if stop_after is not None else total
    model.train()
    while st['step']<end:
        phase=st['step']//iters
        idx=torch.randint(len(tr),(min(batch,len(tr)),),generator=cpu).numpy()
        b=torch.from_numpy(tr[idx]).to(dev)
        loss=model.step(b,phase,opts,g)
        st['step']+=1
        if st['step']%every==0 or st['step']==end or st['step']%iters==0 or expired():
            st['history'].append({'step':st['step'],'phase':phase,**loss})
            save_atomic({'model':model.state_dict(),'optimizers':{k:v.state_dict() for k,v in opts.items()},
                'noise':g.get_state(),'sample':cpu.get_state(),'rng':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all() if str(dev).startswith('cuda') else None,'state':st},ckpt)
            print(f'    GAN phase {phase+1}/{model.n_phases} step {st["step"]}/{total} {loss}',flush=True)
        if expired(): raise TimeoutError("Time budget reached after saving GAN checkpoint")
    return st


@torch.no_grad()
def generate(model,X,dev,seed=0,bs=32):
    model.eval();g=torch.Generator(device=dev).manual_seed(seed+30000)
    result=np.concatenate([model.window(torch.from_numpy(X[i:i+bs]).to(dev),g).cpu().numpy() for i in range(0,len(X),bs)])
    if not np.isfinite(result).all():raise FloatingPointError('Non-finite generation')
    return result
