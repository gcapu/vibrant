import numpy as np
import pytest
import torch

from vibrant import materials
from vibrant.material_properties import mu_lambda


class TestElastic:
    def test_voigt_isotropic_no_poisson_2D_stress(self):
        torch.manual_seed(100)
        diag_elements = torch.randn(3)
        mat = materials.Elastic(torch.diag(diag_elements))
        strain = torch.randn(4, 3)
        stress = mat(strain)
        assert torch.allclose(stress, diag_elements * strain)

    def test_tensor_full_3D_stress(self):
        torch.manual_seed(100)
        stiffness = np.arange(3 ** 4).reshape(3, 3, 3, 3) + 0.5
        strain = np.arange(2 * 3 * 3).reshape(2, 3, 3) - 2.5
        mat = materials.Elastic(torch.tensor(stiffness))
        stress = mat(torch.tensor(strain))
        assert np.allclose(
            stress.numpy(), np.einsum("ijkl, bkl -> bij", stiffness, strain)
        )


class TestIsotropic3D:
    def test_stiffness(self):
        E = 2
        nu = 0.3
        coef = E / (1 + nu) / (1 - 2 * nu)
        stiffness = coef * torch.tensor(
            [
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
            ]
        )
        isomat = materials.Isotropic3D(E, nu)
        assert torch.allclose(isomat.C.squeeze(), stiffness)

    def test_stress(self):
        isomat = materials.Isotropic3D(2, 0.3)
        assert torch.allclose(isomat(torch.eye(6)), isomat.C.squeeze())

    def test_with_no_poisson_is_uncoupled(self):
        isomat = materials.Isotropic3D(2, 0)
        stress = isomat(torch.eye(6))
        assert (stress.diag().abs() > 0).all()
        assert torch.allclose(stress - stress.diag().diag(), torch.zeros(6, 6))


class TestIsotropic2D:
    def test_PE_stress(self):
        isomat = materials.IsotropicPE(2, 0.3)
        assert torch.allclose(isomat(torch.eye(3)), isomat.C.squeeze())

    def test_PS_stress(self):
        isomat = materials.IsotropicPS(2, 0.3)
        assert torch.allclose(isomat(torch.eye(3)), isomat.C.squeeze())


class TestMaterialProperties:
    def test_with_wrong_number_of_args_it_fails(self):
        with pytest.raises(ValueError):
            mu_lambda(E=1, nu=0.3, mu=4)
            mu_lambda(E=1)

    def test_works_for_E_and_nu(self):
        mu, lam = mu_lambda(E=2, nu=0.3)
        assert mu == pytest.approx(1 / 1.3)
        assert lam == pytest.approx(2 * 0.3 / 1.3 / 0.4)

    def test_works_for_E_and_mu(self):
        mu, lam = mu_lambda(E=2, mu=0.7)
        assert mu == pytest.approx(0.7)
        assert lam == pytest.approx(0.7 * 0.6 / (2.1 - 2))

    def test_works_for_E_and_lam(self):
        mu, lam = mu_lambda(E=2, lam=1)
        assert mu == pytest.approx((2 - 3 + 17 ** 0.5) / 4)
        assert lam == pytest.approx(lam)

    def test_works_for_nu_and_mu(self):
        mu, lam = mu_lambda(nu=0.3, mu=0.7)
        assert mu == pytest.approx(0.7)
        assert lam == pytest.approx(2 * 0.7 * 0.3 / 0.4)

    def test_works_for_nu_and_lamda(self):
        mu, lam = mu_lambda(nu=0.3, lam=1)
        assert mu == pytest.approx(0.4 / 0.6)
        assert lam == pytest.approx(1)

    def test_works_for_mu_and_lambda_themselves(self):
        mu, lam = mu_lambda(mu=2, lam=3)
        assert mu == pytest.approx(2)
        assert lam == pytest.approx(3)
