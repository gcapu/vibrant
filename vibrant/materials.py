"""
Basic Materials.

Materials are objects that have the following members:
    A method named `update`. The particular arguments and outputs may
        change from material to material.
    A property or attribute named `density`, which returns the mass properties.
    A property or attribute named `matID`, which returns a unique integer ID for
        that material.
"""

from vibrant.math_extensions import btdot


class Elastic:
    """Basic elastic Material.

    Args:
        C (tensor): the stiffness matrix. It can have dimension 2 or 4, which
            corresopnd to the voigt and tensor form.
        density (float): the density of the material.
    """

    def __init__(self, C, density):
        # Adding a batch dimension
        self.C = C.unsqueeze(0)
        self.density = density
        self.matID = 1

    def update(self, strain):
        """Update the material points and return the state.

        Args:
            strain (tensor): It must have dimension 2 or 3. The first dimension
                contains the batch and the other one(s) contain the strain. If
                the dimension is 2, the strain is in voigt form, otherwise it is
                in tensor form.
        Returns:
            stress (tensor): It matches the dimension of the input strain.
        """
        # PyTorch's einsum does not support broadcasting yet, so in the meantime
        #   we use our own `btt`.
        #   For more info see https://github.com/pytorch/pytorch/issues/30194

        return btdot(self.C, strain)
