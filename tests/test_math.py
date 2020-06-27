import pytest
import torch
import numpy as np
from vibrant.math_extensions import btdot, assemble


def describe_btdot():
    def size_6_batch_dot_matches_numpy_result():
        large = torch.tensor(np.arange(2 * 2 * 3 * 6).reshape(2, 2, 3, 6))
        small = torch.tensor(np.arange(2 * 6).reshape(2, 6))
        result = btdot(large, small)
        assert result.size() == (2, 2, 3)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                for k in range(result.size(2)):
                    assert result[i, j, k] == large[i, j, k].numpy() @ small[i].numpy()

    def size_2_batch_double_dot_matches_numpy_result():
        large = torch.tensor(np.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2))
        small = torch.tensor(np.arange(3 * 2 * 2).reshape(3, 2, 2))
        result = btdot(large, small)
        assert result.size() == (3, 4)
        for b in range(result.size(0)):
            b_out = np.tensordot(large[b].numpy(), small[b].numpy())
            assert np.allclose(result[b].numpy(), b_out)


def describe_assemble():
    @pytest.mark.parametrize("input_dim", [1, 2, 3])
    @pytest.mark.parametrize("conn_dim", [1, 2, 3])
    @pytest.mark.parametrize("elements", [2, 5])
    @pytest.mark.parametrize("length", [3, 10])
    def simple_case(input_dim, conn_dim, elements, length):
        torch.manual_seed(100)
        conn = torch.randint(length, (elements, conn_dim))
        inputs = torch.rand(elements, conn_dim, input_dim)
        result = assemble(length, conn, inputs)
        real_result = torch.zeros(length, input_dim)
        for i, element in enumerate(conn):
            for j, node in enumerate(element):
                real_result[node] += inputs[i, j]
        assert torch.allclose(result, real_result)
