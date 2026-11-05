"""Context-FID with a frozen TS2Vec encoder (official code, commit b0088e1, vendored unmodified in third_party/ts2vec;
MIT licence). Encoder trained on real training windows only (protocol E8, D44). FID as in the widely used
Context-FID code (mean/covariance of full-series embeddings, scipy sqrtm, real part)."""
import hashlib, random, sys
from pathlib import Path
import numpy as np, scipy.linalg, torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "ts2vec"
COMMIT = "b0088e14a99706c05451316dc6db8d3da9351163"
SHA = {"ts2vec.py": "2704502598dde115b55559bff8b56563a94eed5cdae4087ec8941a2a077f9972",
       "utils.py": "85ca67e8e84cad09933d2f3ba60de3bc18bafda78706bd5c4c4a509f577dea90",
       "LICENSE": "a0cc2d47952871dab9279d53700c02afb7a5b947afb85487d992f3c609486171",
       "models/__init__.py": "ea9eb54772c0e1bfbdfedf068b36079f44eed61ef09c68cab3b151682a88c2d9",
       "models/dilated_conv.py": "9c98e7313e78748670083c400a4ee4d1357a88f1a0457c7f44e780ac4165cdbb",
       "models/encoder.py": "41ceb3902466686dfb0aa119175761ebceb78dd882de7403b45890b965c2fe94",
       "models/losses.py": "c6f38f66a09b29689ab501062f2851bbd8139eb4bdb40dc73aa4bb0ce3bb2f8c"}
SETTINGS = dict(output_dims=320, hidden_dims=64, depth=10, lr=0.001, batch_size=16)   # official defaults


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ts2vec_module():
    for f, h in SHA.items():
        if _sha(VENDOR / f) != h:
            raise SystemExit(f"STOP: vendored TS2Vec file {f} differs from commit {COMMIT[:7]}")
    sys.dont_write_bytecode = True
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import models, utils, ts2vec
    for m in (models, utils, ts2vec):
        if not Path(m.__file__).resolve().is_relative_to(VENDOR.resolve()):
            raise SystemExit(f"STOP: {m.__name__} imported from {m.__file__}, not the vendored copy")
    return ts2vec


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def new_encoder(input_dims):
    return ts2vec_module().TS2Vec(input_dims=int(input_dims), device="cpu", **SETTINGS)


def load_encoder(path, input_dims):
    m = new_encoder(input_dims)
    m.load(str(path))
    return m


def embed(m, X, bs=64):
    X = np.ascontiguousarray(X, dtype=np.float32)
    return m.encode(X, encoding_window="full_series", batch_size=bs).astype(np.float64)


def fid(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm = scipy.linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2.0 * cm))
