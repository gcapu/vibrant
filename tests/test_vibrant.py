import torch
import numpy as np

from vibrant import __version__
from vibrant.materials import Elastic
from vibrant.math_extensions import btdot


def test_version():
    assert __version__ == '0.0.2'


def describe_btdot():
    def size_6_batch_dot():
        large = torch.tensor(np.arange(2 * 2 * 3 * 6).reshape(2, 2, 3, 6))
        small = torch.tensor(np.arange(2 * 6).reshape(2, 6))
        result = btdot(large, small)
        assert result.size() == (2, 2, 3)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                for k in range(result.size(2)):
                    assert result[i, j, k] == large[i, j, k].numpy() @ small[i].numpy()

    def size_2_batch_double_dot():
        large = torch.tensor(np.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2))
        small = torch.tensor(np.arange(3 * 2 * 2).reshape(3, 2, 2))
        result = btdot(large, small)
        assert result.size() == (3, 4)
        for b in range(result.size(0)):
            bout = np.tensordot(large[b].numpy(), small[b].numpy())
            assert np.allclose(result[b].numpy(), bout)


def describe_elastic():
    def voigt_isotropic_no_poisson():
        diag = torch.randn(3)
        mat = Elastic(torch.diag(diag), 1)
        strain = torch.randn(4, 3)
        stress = mat.update(strain)
        assert torch.allclose(stress, diag * strain)

    # TODO: add tests for known materials.
