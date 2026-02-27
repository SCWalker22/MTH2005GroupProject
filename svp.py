import numpy as np

def svp(X):
    """
    Calculation of Saturation Vapour Pressure (es)

    Parameters
    ----------
    X : array-like
        Input array where temperature (in Kelvin) is the SECOND element
        This function expects to have temperature as an input, in Kelvins,
        e.g. if X is a vector input to the function and temperature is the 
        second element, it would be defined as X[1]. You will need to modifiy
        to fit into your own modelling framework.

    Returns
    -------
    es : float or ndarray
        Saturation vapour pressure in Pascals (Pa).
    """

    # Convert input to NumPy array (allows list, tuple, ndarray, etc.)
    X = np.asarray(X)

    T = X[1]

    # Bolton (1980) approximation (pressure in hPa)
    es = 6.112 * np.exp(17.65 * (T - 273.0) / (T - 29.5))

    # Convert from hPa to Pa
    es = 100.0 * es

    return es
