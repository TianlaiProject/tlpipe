import numpy as np
from cora.util import hputil


## Test Packing and Unpacking of Healpix packed alms.g
def test_pack_unpack_half():

    pck1 = np.arange(10.0, dtype=np.complex128)
    pck2 = hputil.pack_alm(hputil.unpack_alm(pck1, 3))

    assert np.allclose(pck1, pck2)


def test_pack_unpack_full():

    pck1 = np.arange(10.0, dtype=np.complex128)
    pck2 = hputil.pack_alm(hputil.unpack_alm(pck1, 3, fullm=True))

    assert np.allclose(pck1, pck2)


if __name__ == '__main__':
    test_pack_unpack_half()
    test_pack_unpack_full()