

def mu_lambda(E=None, nu=None, lam=None, mu=None):
    """Get lame's parameters.

    Build the lame constants of a material using the elastic constants for 
        isotropic materials. You must specify exactly two of the four available 
        constants. For example, you can provide E and mu as arguments.

    Args:
        E (float): Young's modulus or elastic modulus.
        nu (float): Poisson's ratio.
        mu (float): lame's second parameter or shear modulus G.
        lam (flaot): lame's first parameter lambda.
    Returns:
        (float, float): mu and lambda

    """
    number_of_inputs = sum(p is not None for p in (E, nu, lam, mu))
    if number_of_inputs != 2:
        raise ValueError(
            f"Two elastic constants are expected and received"
            + f" {number_of_inputs} instead."
        )
    if E is not None and nu is not None:
        lam = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
    elif E is not None:
        if mu is not None:
            lam = mu * (E - 2 * mu) / (3 * mu - E)
        elif lam is not None:
            R = (E ** 2 + 9 * lam ** 2 + 2 * E * lam) ** 0.5
            mu = (E - 3 * lam + R) / 4
    elif nu is not None:
        if mu is not None:
            lam = 2 * mu * nu / (1 - 2 * nu)
        elif lam is not None:
            mu = lam * (1 - 2 * nu) / (2 * nu)
    return mu, lam