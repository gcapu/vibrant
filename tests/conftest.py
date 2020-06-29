import time

import pytest
import torch


@pytest.fixture(params=['random', 'deterministic'])
def seed(request):
    rseed = 100 if request.param == 'deterministic' else round(time.time() * 100)
    torch.manual_seed(rseed)
    return rseed


@pytest.fixture(params=[(1e-6, 2.0, 0.05, 3.0), (0.01, 2e11, 10.0, 3e3)])
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
