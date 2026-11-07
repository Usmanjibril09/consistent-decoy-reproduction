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
