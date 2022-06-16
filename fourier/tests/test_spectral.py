from unittest import TestCase
import numpy as np


class Test(TestCase):
    def test_cfft_1d(self):
        from spectral import cfft, cifft

        for i in range(10):
            n = np.random.randint(10, 100)
            x = np.random.random(n)
            np.testing.assert_allclose(x, cifft(cfft(x)))
            np.testing.assert_allclose(x, cifft(cfft(x, d=0), d=0))
            np.testing.assert_allclose(x, cifft(cfft(x, d=-1), d=-1))

    def test_cfft_2d(self):
        from spectral import cfft, cifft

        for i in range(10):
            n = np.random.randint(10, 100)
            m = np.random.randint(10, 100)
            x = np.random.random([n, m])
            np.testing.assert_allclose(x, cifft(cfft(x)))
            np.testing.assert_allclose(x, cifft(cfft(x, d=0), d=0))
            np.testing.assert_allclose(x, cifft(cfft(x, d=1), d=1))
            np.testing.assert_allclose(x, cifft(cfft(x, d=-1), d=-1))
