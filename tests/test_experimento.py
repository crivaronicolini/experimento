# import pytest
import numpy as np
import numpy.testing as nt
# from .experimento import experimento
from marco import experimento
import pytest

absdire = '/home/marco/.local/lib/python3.7/site-packages/marco/tests/example_medicion'
reldire = './example_medicion'

def test_absdire():
    expected = absdire
    actual = experimento(absdire).absdire
    assert actual == expected

def test_reldire():
    expected = absdire
    actual = experimento(reldire).absdire
    assert actual == expected

def error_comun(av,ae,bv,be):
    return np.sqrt((ae/av)**2 + (be/bv)**2) *av*bv
    # return np.sqrt((ae/av)**2 + (be/bv)**2)

def error(valv,vale):
    valv = np.array([valv])
    return experimento.set_e(valv, vale)

def error_exp(c,d):
    return experimento.get_e(c*d)

@pytest.mark.parametrize('lista', [[3e3,1e4,2e3,2e4],
                                   [3e-3,1e-5,2e-4,2e-6],
                                   [3,np.sqrt(2),2,np.pi]
                                  ])
def test_error(lista):
    av,ae,bv,be = lista
    expected = error_comun(av,ae,bv,be)
    actual = error_exp(error(av,ae),error(bv,be))
    nt.assert_allclose(actual, expected, verbose=True)

