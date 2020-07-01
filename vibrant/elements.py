import torch

from vibrant.math_extensions import assemble


class Truss:
    """Truss elements."""

    def __init__(self, conn, nodes=None, area=1, material=None):
        self.conn = conn
        self.area = area
        self.material = material
        self.nodes = nodes
        self.strain = None
        self.stress = None

    def force(self):
        """Compute the nodal force."""
        if self.material is None:
            raise TypeError("Truss elements are missing material.")
        if self.nodes is None:
            raise TypeError("Truss elements are missing nodes.")
        Xdiff = self.nodes.X[self.conn[:, 1]] - self.nodes.X[self.conn[:, 0]]
        xdiff = Xdiff + (self.nodes.u[self.conn[:, 1]] - self.nodes.u[self.conn[:, 0]])
        L0 = (Xdiff).norm(dim=1)
        L = (xdiff).norm(dim=1)
        self.strain = (L - L0) / L0
        self.stress = self.material(self.strain)
        direction = xdiff / L[:, None]
        element_forces = self.stress[:, None] * self.area * direction
        element_forces = torch.stack((element_forces, -element_forces), dim=1)
        force = assemble(len(self.nodes), self.conn, element_forces)
        return force

    def mass(self):
        """Compute the nodal mass."""
        if self.material is None:
            raise TypeError("Truss elements are missing material.")
        if self.nodes is None:
            raise TypeError("Truss elements are missing nodes.")
        Xdiff = self.nodes.X[self.conn[:, 1]] - self.nodes.X[self.conn[:, 0]]
        L0 = (Xdiff).norm(dim=1)
        element_mass = L0 * self.area * self.material.density / 2
        element_mass = torch.stack((element_mass[:, None], element_mass[:, None]))
        mass = assemble(len(self.nodes), self.conn, element_mass)
        return mass
