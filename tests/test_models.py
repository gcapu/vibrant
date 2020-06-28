import pytest
import torch
from math import pi, cos, sin

from vibrant.models import Model
from vibrant.nodes import Nodes
from vibrant.elements import Truss
from vibrant.materials import BasicMaterial


@pytest.fixture(params=[2.0, 2e11])
def E(request):
    return request.param


@pytest.fixture(params=[3.0, 3e3])
def rho(request):
    return request.param


@pytest.fixture(params=[1e-6, 1.0])
def area(request):
    return request.param


@pytest.fixture(params=[5e-2, 5.0])
def l(request):
    return request.param


@pytest.fixture
def two_bars(E, rho, area, l):
    X = torch.tensor([[0, 0], [l, 0], [l, l]])
    conn = torch.tensor([[0, 1], [1, 2]], dtype=int)
    nodes = Nodes(X)
    mat = BasicMaterial(lambda e: E * e, rho)
    els = Truss(conn, nodes, area, mat)
    return Model(nodes, els)


def describe_2D_truss_model():
    def test_mass(two_bars, E, area, l, rho):
        m = two_bars.mass()
        assert m.size() == (3, 1)
        assert m[0, 0].item() == pytest.approx(m[1, 0].item() / 2)
        assert m[0, 0].item() == pytest.approx(m[2, 0].item())
        assert m[0, 0].item() == pytest.approx(rho * l * area / 2)

    def test_force(two_bars, E, area, l, rho):
        two_bars.nodes.u = torch.tensor(
            [[l * 0.1, 0], [0, 0], [l * sin(0.1), -(l - l * cos(0.1))]]
        )
        f = two_bars.force()
        assert f[0, 0].item() == pytest.approx(-0.1 * E * area)  # stretch
        assert f[0, 1].item() == pytest.approx(0)  # transversal
        assert f[2, 1].item() == pytest.approx(0)  # rotation
        assert f[2, 1].item() == pytest.approx(0)  # rotation

    def test_acceleration(two_bars, E, area, l, rho):
        two_bars.nodes.u = torch.tensor(
            [[l * 0.1, 0], [0, 0], [l * sin(0.1), -(l - l * cos(0.1))]]
        )
        a = two_bars.acceleration()
        assert a[0, 0].item() == pytest.approx(-0.2 * l * E / rho / l ** 2)  # stretch
        assert a[0, 1].item() == pytest.approx(0)  # transversal
        assert a[2, 1].item() == pytest.approx(0)  # rotation
        assert a[2, 1].item() == pytest.approx(0)  # rotation

    def test_damping_force(two_bars, E, area, l, rho):
        torch.manual_seed(100)
        two_bars.damping = 2
        two_bars.nodes.v = torch.rand(3, 2)
        f = two_bars.force()
        v00 = two_bars.nodes.v[0, 0].item()
        m = rho * l * area / 2
        assert f[0, 0].item() == pytest.approx(-m * v00 * two_bars.damping)
