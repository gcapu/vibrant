import pytest
import torch
from math import cos, sin, pi

from vibrant.nodes import Nodes
from vibrant.materials import BasicMaterial
from vibrant.elements import Truss


def describe_truss():
    @pytest.mark.parametrize("func", ["force", "mass"])
    def without_material_it_fails(func):
        torch.manual_seed(100)
        # the two trusses are mirrored vertically.
        nodes = Nodes(torch.rand(3, 2))
        conn = torch.randint(3, (2, 2))
        elements = Truss(conn, nodes)
        with pytest.raises(TypeError):
            getattr(elements, func)()

    @pytest.mark.parametrize("func", ["force", "mass"])
    def without_nodes_it_fails(func):
        torch.manual_seed(100)
        # the two trusses are mirrored vertically.
        conn = torch.randint(3, (2, 2))
        elements = Truss(conn)
        elements.material = BasicMaterial(lambda strain: 5 * strain, 1)
        with pytest.raises(TypeError):
            getattr(elements, func)()

    @pytest.mark.parametrize("length", [0.1, 100])
    @pytest.mark.parametrize("angle", [0, pi / 6, pi / 4, pi / 2, 2 / 3 * pi, pi])
    def mirrored_trusses_2D(length, angle):
        torch.manual_seed(100)
        strain = torch.rand(1).item() / 10 - 0.05
        area = torch.rand(1).item() / 10
        density = torch.rand(1).item()
        E = torch.rand(1).item() * 1e6
        material = BasicMaterial(lambda strain: E * strain, density)
        # the two trusses are mirrored vertically.
        origin = torch.rand(2)
        p1 = origin + torch.tensor([length * cos(angle), length * sin(angle)])
        p2 = origin + torch.tensor([length * cos(angle), -length * sin(angle)])
        X = torch.stack([origin, p1, p2])
        # add logitudinal stretch
        u1 = strain * (p1 - origin)
        u2 = strain * (p2 - origin)
        u = torch.stack([torch.zeros(2), u1, u2])
        # create the elements
        nodes = Nodes(X, u)
        conn = torch.tensor([[0, 1], [0, 2]], dtype=int)
        elements = Truss(conn, nodes, area, material)
        # test the resulting mass is correct
        bar_weight = density * area * length
        real_mass = torch.tensor([[bar_weight], [bar_weight / 2], [bar_weight / 2]])
        nodal_mass = elements.mass()
        assert nodal_mass.size() == (3, 1)
        assert torch.allclose(nodal_mass, real_mass)
        # compare forces in first element with analytic result
        nodal_forces = elements.force()
        assert nodal_forces.size() == (3, 2)
        assert nodal_forces[1].norm() == pytest.approx(abs(strain * E * area), rel=1e-5)
        assert torch.allclose(-u1 * E * area / length, nodal_forces[1])
        # compare components of the two elements
        assert nodal_forces[0, 1] == pytest.approx(0)
        assert nodal_forces[1, 0] == pytest.approx(nodal_forces[2, 0].item())
        assert nodal_forces[1, 1] == pytest.approx(-nodal_forces[2, 1].item())
        # if both elements have the same end location, they produce the same force
        nodes.u[1] = p2 - p1 + u2
        nodal_forces = elements.force()
        assert torch.allclose(nodal_forces[1], nodal_forces[2])
