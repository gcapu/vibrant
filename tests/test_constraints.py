import pytest
import torch

from vibrant.constraints import ImposeVelocity
from vibrant.nodes import Nodes


class TestImposeVelocityConstraint:
    @pytest.fixture(params=[2, 3])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def nodes(self, seed, length, dim):
        num_nodes = 3
        return Nodes(
            X=length * torch.rand(num_nodes, dim),
            u=length * torch.rand(num_nodes, dim),
            v=length * torch.rand(num_nodes, dim),
        )

    @pytest.fixture(params=[torch.float, torch.int],)
    def vel(self, dim, request):
        dtype = request.param
        return torch.tensor([10] * dim, dtype=dtype)

    def test_constrain_velocity(self, nodes, vel, dim):
        constrained_ids = [1, 2]
        unconstrained_ids = list(set(range(len(nodes))) - set(constrained_ids))
        unconstrained_values = nodes.v[unconstrained_ids]
        vel_constraint = ImposeVelocity(nodes, constrained_ids, vel)
        vel_constraint()
        # check that the unconstrained nodes weren't modified
        assert torch.allclose(nodes.v[unconstrained_ids], unconstrained_values)
        # check that the constrained nodes now have the right value
        for i in constrained_ids:
            assert torch.allclose(nodes.v[i], torch.as_tensor(vel, dtype=torch.float))

    def test_constrain_displacement_fails(self, nodes, length, dim):
        vel_constraint = ImposeVelocity(nodes, [1, 2], [length / 10.0] * dim)
        original_position = nodes.u.clone()
        vel_constraint("u")
        assert torch.allclose(nodes.u, original_position)
