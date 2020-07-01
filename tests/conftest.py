import time

import pytest
import torch


@pytest.fixture(scope="module")
def rtol():
    """Relative numerical tolerance."""
    return 1e-3


@pytest.fixture(scope="module")
def atol():
    """Absolute numerical tolerance."""
    return 1e-4


@pytest.fixture(scope="module", params=["random", "deterministic"])
def seed(request):
    rseed = 100 if request.param == "deterministic" else round(time.time() * 100)
    torch.manual_seed(rseed)
    return rseed


@pytest.fixture(
    scope="module",
    params=[(1e-6, 0.05, 2.0, 3.0), (0.01, 10.0, 2e11, 3e3)],
    ids=["(cm,kg)", "(m,N)"],
)
def ALYD(request):
    """Area, Length, Young, Density.

    Group of common properties of structures using consistent units.
    """
    return request.param


@pytest.fixture
def area(ALYD):
    return ALYD[0]


@pytest.fixture
def young(ALYD):
    return ALYD[1]


@pytest.fixture
def length(ALYD):
    return ALYD[2]


@pytest.fixture
def density(ALYD):
    return ALYD[3]
