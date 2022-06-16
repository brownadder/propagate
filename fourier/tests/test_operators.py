from unittest import TestCase
import numpy as np


class Test(TestCase):
    def test_diffop_1D(self):
        from discretize import grid1d
        from operators import diffop

        # 1D test
        xrange = [0, 2*np.pi]
        x = grid1d(100, xrange)
        s = np.sin(x)

        d1s_numerical = diffop(0, 1, s, xrange)
        d1s_analytical = np.cos(x)
        np.testing.assert_allclose(d1s_analytical, d1s_numerical)

        d2s_numerical = diffop(0, 2, s, xrange)
        d2s_analytical = -np.sin(x)
        np.testing.assert_allclose(d2s_analytical, d2s_numerical)

    def test_diffop_2D(self):
        from discretize import grid, fixrange
        from operators import diffop

        # 2D square
        xrange = fixrange([0, 2*np.pi], 2)
        x = grid([100, 100], xrange)
        s = np.sin(x[0]+3*x[1])

        d1x0s_numerical = np.real(diffop(0, 1, s, xrange))
        d1x0s_analytical = np.cos(x[0]+3*x[1])
        np.testing.assert_allclose(d1x0s_analytical, d1x0s_numerical, rtol=1e-8, atol=1e-10)

        d1x1s_numerical = diffop(1, 1, s, xrange)
        d1x1s_analytical = 3 * np.cos(x[0] + 3 * x[1])
        np.testing.assert_allclose(d1x1s_analytical, d1x1s_numerical, rtol=1e-8, atol=1e-10)

        d2x1_numerical = diffop(1, 2, s, xrange)
        d2x1_analytical = -9 * np.sin(x[0] + 3 * x[1])
        np.testing.assert_allclose(d2x1_analytical, d2x1_numerical, rtol=1e-8, atol=1e-10)