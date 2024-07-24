from numpy import pi


def kraijenhoff_parameters(
    x: float, L: float, Ks: float, D: float, Sy: float
) -> tuple[float]:
    """Get Pastas parameters for Kraijenhoff van de Leur response function for
    an homogeneous aquifer between two parallel canals.

    Parameters
    ----------
    x : float
        The location in the aquifer in meters with x=0 in the center.
    L : float
        The aquifer length in meters.
    Ks : float
        The saturated hydraulic conductivity in meter/day.
    D : float
        The saturated thickness of the aquifer in meters.
    Sy : float
        The specific yield of the aquifer [-].

    Returns
    -------
    tuple[float]
        A tuple containing the response parameters A, a, and b.
    """
    N = 1.0e-3  # m/d
    b = x / L
    A = -N * L**2 / (2 * Ks * D) * (b**2 - (1 / 4))
    a = Sy * L**2 / (pi**2 * Ks * D)
    return A, a, b
