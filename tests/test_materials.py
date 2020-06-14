import pytest
import torch
import numpy as np


from vibrant import materials
from vibrant.math_extensions import btdot
from vibrant.material_properties import mu_lambda


def describe_btdot():
    def size_6_batch_dot_matches_numpy_result():
        large = torch.tensor(np.arange(2 * 2 * 3 * 6).reshape(2, 2, 3, 6))
        small = torch.tensor(np.arange(2 * 6).reshape(2, 6))
        result = btdot(large, small)
        assert result.size() == (2, 2, 3)
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                for k in range(result.size(2)):
                    assert result[i, j, k] == large[i, j, k].numpy() @ small[i].numpy()

    def size_2_batch_double_dot_matches_numpy_result():
        large = torch.tensor(np.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2))
        small = torch.tensor(np.arange(3 * 2 * 2).reshape(3, 2, 2))
        result = btdot(large, small)
        assert result.size() == (3, 4)
        for b in range(result.size(0)):
            b_out = np.tensordot(large[b].numpy(), small[b].numpy())
            assert np.allclose(result[b].numpy(), b_out)


def describe_elastic():
    def voigt_isotropic_no_poisson_2D_produces_correct_stress():
        torch.manual_seed(100)
        diag_elements = torch.randn(3)
        mat = materials.Elastic(torch.diag(diag_elements))
        strain = torch.randn(4, 3)
        stress = mat(strain)
        assert torch.allclose(stress, diag_elements * strain)

    def tensor_full_3D_produces_correct_stress():
        torch.manual_seed(100)
        stiffness = np.arange(3 ** 4).reshape(3, 3, 3, 3) + 0.5
        strain = np.arange(2 * 3 * 3).reshape(2, 3, 3) - 2.5
        mat = materials.Elastic(torch.tensor(stiffness))
        stress = mat(torch.tensor(strain))
        assert np.allclose(
            stress.numpy(), np.einsum('ijkl, bkl -> bij', stiffness, strain)
        )


def describe_isotropic3D():
    def produces_correct_stiffness():
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

    def produces_correct_stress():
        isomat = materials.Isotropic3D(2, 0.3)
        assert torch.allclose(isomat(torch.eye(6)), isomat.C.squeeze())

    def with_no_poisson_is_uncoupled():
        isomat = materials.Isotropic3D(2, 0)
        stress = isomat(torch.eye(6))
        assert (stress.diag().abs() > 0).all()
        assert torch.allclose(stress - stress.diag().diag(), torch.zeros(6, 6))


def describe_isotropicPE():
    def produces_correct_stress():
        isomat = materials.IsotropicPE(2, 0.3)
        assert torch.allclose(isomat(torch.eye(3)), isomat.C.squeeze())


def describe_isotropicPS():
    def produces_correct_stress():
        isomat = materials.IsotropicPS(2, 0.3)
        assert torch.allclose(isomat(torch.eye(3)), isomat.C.squeeze())


def describe_material_properties():
    def fails_with_wrong_number_of_args():
        with pytest.raises(ValueError):
            mu_lambda(E=1, nu=0.3, mu=4)
            mu_lambda(E=1)

    def works_for_E_and_nu():
        mu, lam = mu_lambda(E=2, nu=0.3)
        assert mu == pytest.approx(1 / 1.3)
        assert lam == pytest.approx(2 * 0.3 / 1.3 / 0.4)

    def works_for_E_and_mu():
        mu, lam = mu_lambda(E=2, mu=0.7)
        assert mu == pytest.approx(0.7)
        assert lam == pytest.approx(0.7 * 0.6 / (2.1 - 2))

    def works_for_E_and_lam():
        mu, lam = mu_lambda(E=2, lam=1)
        assert mu == pytest.approx((2 - 3 + 17 ** 0.5) / 4)
        assert lam == pytest.approx(lam)

    def works_for_nu_and_mu():
        mu, lam = mu_lambda(nu=0.3, mu=0.7)
        assert mu == pytest.approx(0.7)
        assert lam == pytest.approx(2 * 0.7 * 0.3 / 0.4)

    def works_for_nu_and_lamda():
        mu, lam = mu_lambda(nu=0.3, lam=1)
        assert mu == pytest.approx(0.4 / 0.6)
        assert lam == pytest.approx(1)

    def works_for_mu_and_lambda_themselves():
        mu, lam = mu_lambda(mu=2, lam=3)
        assert mu == pytest.approx(2)
        assert lam == pytest.approx(3)
