from math import cos, pi, sin

import pytest
import torch

from vibrant.elements import Truss
from vibrant.materials import BasicMaterial
from vibrant.nodes import Nodes


class TestTrussWithMissingProperties:
    @pytest.fixture
    def nodes(self):
        return Nodes(torch.rand(3, 2))

    @pytest.fixture
    def conn(self):
        return torch.randint(3, (2, 2))

    @pytest.fixture
    def mat(self):
        return BasicMaterial(lambda strain: 5 * strain, 1)

    def test_without_material_mass_fails(self, seed, nodes, conn):
        elements = Truss(conn, nodes)
        with pytest.raises(TypeError):
            elements.mass()

    def test_without_material_force_fails(self, seed, nodes, conn):
        elements = Truss(conn, nodes)
        with pytest.raises(TypeError):
            elements.force()

    def test_without_nodes_mass_fails(self, seed, conn, mat):
        elements = Truss(conn, material=mat)
        with pytest.raises(TypeError):
            elements.mass()

    def test_without_nodes_force_fails(self, seed, conn, mat):
        elements = Truss(conn, material=mat)
        with pytest.raises(TypeError):
            elements.force()


class TestMirroredTruss:
    @pytest.fixture(params=[0, pi / 6, pi / 2])
    def angle(self, request):
        return request.param

    @pytest.fixture(params=[(1e-6, 2, 0.05, 3), (0.01, 2e11, 10, 3e3)])
    def AELD(self, request):
        """Area, E, Length, Density."""
        return request.param

    @pytest.fixture
    def area(self, AELD):
        return AELD[0]

    @pytest.fixture
    def E(self, AELD):
        return AELD[1]

    @pytest.fixture
    def length(self, AELD):
        return AELD[2]

    @pytest.fixture
    def density(self, AELD):
        return AELD[3]

    @pytest.fixture
    def rand_strain(self, seed):
        return torch.rand(1).item() / 10 - 0.05

    @pytest.fixture
    def nodes(self, seed, length, angle, rand_strain):
        """Three nodes mirrored vertically."""
        origin = length * 100 * torch.rand(2)
        p1 = origin + torch.tensor([length * cos(angle), length * sin(angle)])
        p2 = origin + torch.tensor([length * cos(angle), -length * sin(angle)])
        X = torch.stack([origin, p1, p2])
        # logitudinal stretch
        u1 = rand_strain * (X[1] - X[0])
        u2 = rand_strain * (X[2] - X[0])
        u = torch.stack([torch.zeros(2), u1, u2])
        return Nodes(X, u)

    @pytest.fixture
    def elements(self, nodes, E, density, area):
        material = BasicMaterial(lambda strain: E * strain, density)
        conn = torch.tensor([[0, 1], [0, 2]], dtype=int)
        return Truss(conn, nodes, area, material)

    def test_mass_matches_analytic(self, elements, area, length, density):
        # test the resulting mass is correct
        bar_weight = density * area * length
        real_mass = torch.tensor([[bar_weight], [bar_weight / 2], [bar_weight / 2]])
        nodal_mass = elements.mass()
        assert nodal_mass.size() == (3, 1)
        assert torch.allclose(nodal_mass, real_mass)

    @pytest.fixture
    def rtol(self):
        return 1e-3

    @pytest.fixture
    def atol(self):
        return 1e-4

    def test_force_matches_analytic(
        self, elements, nodes, area, E, length, rand_strain, rtol, atol,
    ):
        forces = elements.force()
        assert forces.size() == (3, 2)
        assert forces[1].norm() == pytest.approx(
            abs(rand_strain * E * area), rel=rtol, abs=atol
        )
        assert torch.allclose(
            -nodes.u[1] * E * area / length, forces[1], rtol=rtol, atol=atol
        )

    def test_force_components_match(self, elements, rtol, atol):
        """If the nodes are mirrored, so the forces must be mirrored."""
        forces = elements.force()
        assert forces[0, 1] / forces[1].norm() == pytest.approx(0, abs=atol)
        assert forces[1, 0] == pytest.approx(forces[2, 0].item(), rel=rtol, abs=atol)
        assert forces[1, 1] == pytest.approx(-forces[2, 1].item(), rel=rtol, abs=atol)

    def test_forces_match_for_same_position(self, elements, nodes, rtol, atol):
        """If both elements have the same end location, they produce the same force."""
        nodes.u[1] = nodes.X[2] - nodes.X[1] + nodes.u[2]
        forces = elements.force()
        assert torch.allclose(forces[1], forces[2], rtol=rtol, atol=atol)
