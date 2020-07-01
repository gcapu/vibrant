from math import cos, sin

import pytest
import torch

from vibrant.elements import Truss
from vibrant.materials import BasicMaterial
from vibrant.models import Model
from vibrant.nodes import Nodes


class TestTrussModel2D:
    @pytest.fixture
    def two_bars(self, young, density, area, length):
        X = torch.tensor([[0, 0], [length, 0], [length, length]])
        conn = torch.tensor([[0, 1], [1, 2]], dtype=int)
        nodes = Nodes(X)
        mat = BasicMaterial(lambda e: young * e, density)
        els = Truss(conn, nodes, area, mat)
        return Model(nodes, els)

    def test_mass(self, two_bars, young, area, length, density):
        m = two_bars.mass()
        assert m.size() == (3, 1)
        assert m[0, 0].item() == pytest.approx(m[1, 0].item() / 2)
        assert m[0, 0].item() == pytest.approx(m[2, 0].item())
        assert m[0, 0].item() == pytest.approx(density * length * area / 2)

    def test_force(self, two_bars, young, area, length, density):
        two_bars.nodes.u = torch.tensor(
            [[length * 0.1, 0], [0, 0], [length * sin(0.1), -length * (1 - cos(0.1))]]
        )
        f = two_bars.force()
        assert f[0, 0].item() == pytest.approx(-0.1 * young * area)  # stretch
        assert f[0, 1].item() == pytest.approx(0)  # transversal
        assert f[2, 1].item() == pytest.approx(0)  # rotation
        assert f[2, 1].item() == pytest.approx(0)  # rotation

    def test_acceleration(self, two_bars, young, area, length, density):
        two_bars.nodes.u = torch.tensor(
            [[length * 0.1, 0], [0, 0], [length * sin(0.1), -length * (1 - cos(0.1))]]
        )
        a = two_bars.acceleration()
        assert a[0, 0].item() == pytest.approx(
            -0.2 * young / density / length
        )  # stretch
        assert a[0, 1].item() == pytest.approx(0)  # transversal
        assert a[2, 1].item() == pytest.approx(0)  # rotation
        assert a[2, 1].item() == pytest.approx(0)  # rotation

    def test_damping_force(self, two_bars, young, area, length, density):
        torch.manual_seed(100)
        two_bars.damping = 2
        two_bars.nodes.v = torch.rand(3, 2)
        f = two_bars.force()
        v00 = two_bars.nodes.v[0, 0].item()
        m = density * length * area / 2
        assert f[0, 0].item() == pytest.approx(-m * v00 * two_bars.damping)


@pytest.mark.usefixtures("seed")
class TestTruss3D:
    def test_single_bar_force(self, length, area, young, rtol):
        # assign a random initial position and normalize to length
        X = torch.rand((2, 3)) + 100
        X = length * X / (X[1] - X[0]).norm()
        # align the bar to z direction and stretch it
        strain = 0.1
        u = torch.stack(
            [torch.zeros(3), X[0] - X[1] + length * torch.tensor([0, 0, 1 + strain])]
        )
        nodes = Nodes(X, u)
        mat = BasicMaterial(lambda e: young * e)
        conn = torch.tensor([[0, 1]], dtype=int)
        elements = Truss(conn, nodes, area, mat)
        model = Model(nodes, elements)
        assert model.force()[1, 2] == pytest.approx(-strain * young * area, rel=rtol)

    def pyramid_truss(self, length, area, young, density=1):
        X = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]])
        X = (X + 100.0) * length
        u = torch.rand(8, 3) * length / 2.0
        v = torch.rand(8, 3) * length / 2.0
        nodes = Nodes(X, u, v)
        conn = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 4], [2, 4], [3, 4]], dtype=int,
        )
        material = BasicMaterial(lambda e: young * e, density)
        elements = Truss(conn, nodes, area, material)
        model = Model(nodes, elements)  # zero damping
        return model

    def test_truss_3D_force(self, length, area, young):
        model = self.pyramid_truss(length, area, young)
        # obtainig the force at node 4
        analytic_result = torch.zeros((3,))
        for node_id in [1, 2, 3]:
            Xdiff = model.nodes.X[4] - model.nodes.X[node_id]
            xdiff = Xdiff + (model.nodes.u[4] - model.nodes.u[node_id])
            l0 = Xdiff.norm()
            l = xdiff.norm()
            stress = young * (l - l0) / l0
            analytic_result -= area * stress * xdiff / l
        assert torch.allclose(model.force()[4], analytic_result)
