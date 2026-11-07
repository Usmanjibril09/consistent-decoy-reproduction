"""EPIC arm seed shim (D48/D49). Active only when EPIC_SEED is set: seeds Python's random and
NumPy's global RNG at interpreter start, so unmodified upstream scripts become repeatable."""
import os as _os
_seed = _os.environ.get("EPIC_SEED")
if _seed is not None:
    import random as _random
    _random.seed(int(_seed))
    try:
        import numpy as _np
    except ImportError:
        pass
    else:
        _np.random.seed(int(_seed))
# Smoke part 2 (VRNN): also seed torch's global RNG when EPIC_TORCH_SEED is set.
# The power generators never set it, so smoke-1 behaviour is unchanged.
_tseed = _os.environ.get("EPIC_TORCH_SEED")
if _tseed is not None:
    import torch as _torch
    _torch.manual_seed(int(_tseed))
