"""
Basic Materials.

Materials are objects that have the following members:
    A `__call__` method. The particular arguments and outputs may change from 
        material to material.
    A property or attribute named `density`, which returns the mass properties.
"""

import torch

from vibrant.math_extensions import btdot


class BasicMaterial:
    """Basic material that provides the stress and density
    
    It allows you to use a function as a material. It provides support for material 
      density."""

    def __init__(self, function, density):
        self.function = function
        self.density = density

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Elastic:
    """Basic elastic Material.

    Args:
        C (tensor): the stiffness matrix. It can have dimension 2 or 4, which
            corresopnd to the voigt and tensor form.
        density (float): the density of the material.
    """

    def __init__(self, C, density=1):
        # Adding a batch dimension
        self.C = C.unsqueeze(0)
        self.density = density

    def __call__(self, strain):
        """Get the stress.

        Args:
            strain (tensor): A tensor of dimension 2 or 3. The first dimension
                contains the batch and the other one(s) contain the strain. If
                the dimension is 2, each row is a strain is in voigt form,
                otherwise each batch contains a strain in tensor form.
        Returns:
            stress (tensor): It matches the shape of the input strain.
        """
        # PyTorch's einsum does not support broadcasting yet, so in the meantime
        #   we use our own batch dot multiplication.
        #   For more info see https://github.com/pytorch/pytorch/issues/30194

        return btdot(self.C, strain)


class Isotropic3D(Elastic):
    """Isotropic elastic material 3D in voigt form.

    Args:
        E (float): Young's modulus or elastic modulus.
        nu (float): Poisson's ratio. 
        density (float): the density of the material.
    """

    def __init__(self, E, nu, density=1):
        coef = E / (1 + nu) / (1 - 2 * nu)
        C = torch.zeros(6, 6)
        C.fill_diagonal_(1 - 2 * nu)
        C[:3, :3] += nu
        C[3:, 3:] /= 2
        C *= coef
        super().__init__(C, density)


class IsotropicPS(Elastic):
    """Isotropic elastic material in plane stress in voigt form.

    Args:
        E (float): Young's modulus or elastic modulus.
        nu (float): Poisson's ratio. 
        density (float): the density of the material.
    """

    def __init__(self, E, nu, density=1):
        coef = E / (1 - nu ** 2)
        C = coef * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        super().__init__(C, density)


class IsotropicPE(Elastic):
    """Isotropic elastic material in plane strain in voigt form.

    Args:
        E (float): Young's modulus or elastic modulus.
        nu (float): Poisson's ratio. 
        density (float): the density of the material.
    """

    def __init__(self, E, nu, density=1):
        coef = E / (1 + nu) / (1 - 2 * nu)
        C = coef * torch.tensor(
            [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]]
        )
        super().__init__(C, density)
