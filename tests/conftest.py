import time

import pytest
import torch


@pytest.fixture(params=['random', 'deterministic'])
def seed(request):
    rseed = 100 if request.param == 'deterministic' else round(time.time() * 100)
    torch.manual_seed(rseed)
    return rseed