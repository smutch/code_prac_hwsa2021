import numpy as np
from pytest_regtest import regtest

from code_prac import mass_function


def test_mass_function(regtest):

    redshifts = [0.0, 5.0, 10.0, 20.0, 30.0]
    masses = np.logspace(2, 15, 200)

    result = mass_function(redshifts, masses)

    with regtest:
        print(result)
