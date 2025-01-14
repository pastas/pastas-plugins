from numpy import pi, sqrt
from scipy.special import k0


def kraijenhoff_parameters(
    S: float, K: float, D: float, x: float, L: float, N: float = 1.0
) -> tuple[float]:
    """Get Pastas parameters for Kraijenhoff van de Leur response function for
    an homogeneous aquifer between two parallel canals.

    Parameters
    ----------
    S : float
        The storativity of the aquifer [-].
    K : float
        The saturated hydraulic conductivity [L/T].
    D : float
        The saturated thickness of the aquifer [L].
    x : float
        The location in the aquifer [L] with x=0 in the center.
    L : float
        The aquifer length [L].
    N : float
        The recharge flux [L/T].

    Returns
    -------
    tuple[float]
        A tuple containing the response parameters A, a, and b.
    """
    b = x / L
    A = -N * L**2 / (2 * K * D) * (b**2 - (1 / 4))
    a = S * L**2 / (pi**2 * K * D)
    return A, a, b


def exponential_parameters(S: float, c: float, N: 1.0) -> tuple[float]:
    """Get Pastas parameters for an Exponential response for a linear
    reservoir system.

    Parameters
    ----------
    S : float
        The storativity of the aquifer [-].
    c : float
        The drainage resistance [T].
    N : float
        The recharge flux [L/T].

    Returns
    -------
    tuple[float]
        A tuple containing the response parameters A and a.
    """

    A = N * c
    a = c * S
    return A, a


def hantush_parameters(
    S: float, K: float, D: float, c: float, r: float, Q: float = -1.0
) -> tuple[float]:
    """Get Pastas parameters for an Hantush response for a well in a confined aquifer.

    Parameters
    ----------
    S : float
        The storativity of the aquifer [-].
    K : float
        The saturated hydraulic conductivity [L/T].
    D : float
        The saturated thickness of the aquifer [L].
    c : float
        The drainage resistance [T].
    r : float
        The distance from the well resistance [L].
    Q : float
        The discharge of the well [L].

    Returns
    -------
    tuple[float]
        A tuple containing the response parameters A, a and b.
    """

    T = K * D
    lab = sqrt(T * c)
    b = r**2 / (4.0 * lab**2.0)
    a = c * S
    A = Q * k0(r / lab) / (2.0 * pi * T)

    return A, a, b


def theis_parameters(
    S: float, K: float, D: float, x: float, L: float, Q: float = -1.0
) -> tuple[float]:
    """Get Pastas parameters for an Theis response for a well between two parallel canals.

    Parameters
    ----------
    S : float
        The storativity of the aquifer [-].
    K : float
        The saturated hydraulic conductivity [L/T].
    D : float
        The saturated thickness of the aquifer [L].
    x : float
        The location in the aquifer [L] with x=0 in the center.
    L : float
        The aquifer length [L].
    Q : float
        The discharge of the well [L].

    Note
    ----
    Works only along the line y=0

    Returns
    -------
    tuple[float]
        A tuple containing the response parameters A, a and b.
    """
    T = K * D
    xw = 0.0  # with xw = 0
    A = Q / (4.0 * pi * T)
    a = S * L**2 / (pi**2.0 * T)
    b = (x - xw) / L
    return A, a, b
