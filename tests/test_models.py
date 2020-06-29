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
