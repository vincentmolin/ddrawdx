import os

import jax
import pytest

os.environ["MPLBACKEND"] = "Agg"


@pytest.fixture(params=["eager", "jit"])
def execute(request):
    """Exercise public operations both directly and with caller-applied JIT."""
    return jax.jit if request.param == "jit" else lambda fn: fn
