from unittest import TestCase
import numpy as np


class Test(TestCase):
    def test_dim(self):
        from discretize import dim
        self.assertEqual(1, dim(np.array([1, 3, 2])))
        self.assertEqual(2, dim(np.array([[1, 3, 2], [5, -2, 1]])))

    def test_fixrange(self):
        from discretize import fixrange
        np.testing.assert_array_equal(np.array([[-1, 1]]), fixrange(-1, 1))
        np.testing.assert_array_equal(np.array([[-1, 1], [-1, 1]]), fixrange(-1, 2))
        np.testing.assert_array_equal(np.array([[0, 1], [0, 1]]), fixrange([0, 1], 2))
        np.testing.assert_array_equal(np.array([[0, 1], [4, 6]]), fixrange(np.array([[0, 1], [4, 6]]), 2))

    def test_grid1d(self):
        from discretize import grid1d
        np.testing.assert_array_equal(np.linspace(-9.0, 9.0, 10), grid1d(10, [-10, 10]))
        np.testing.assert_array_equal(np.linspace(3.4, 6.6, 5), grid1d(5, [3, 7]))

    def test_l2norm(self):
        from discretize import l2norm, grid1d
        self.assertAlmostEqual(1 / np.sqrt(2), l2norm(np.array([1, 0]), [0, 1]))
        self.assertAlmostEqual(1, l2norm(np.ones(10), [0, 1]))
        self.assertAlmostEqual(np.sqrt(5), l2norm(np.ones(10), [-1, 4]))
        xr = [0, 2 * np.pi]
        x = grid1d(100, xr)
        u = np.sin(x)
        self.assertAlmostEqual(np.sqrt(np.pi), l2norm(u, xr))

    def test_l2inner(self):
        from discretize import l2inner, l2norm, grid1d
        xr = [-1, 4]
        u = np.ones(10)
        self.assertAlmostEqual(l2norm(u, xr)**2, np.real(l2inner(u, u, xr)))
        u = np.array([1, 0, 1 + 2j])
        v = np.array([0, 4, 5 + 3j])
        np.testing.assert_allclose((5/3)*(11 - 7j), l2inner(u, v, xr))
        xr = [0, 2*np.pi]
        x = grid1d(100, xr)
        u = np.sin(x)
        v = np.cos(x)
        self.assertAlmostEqual(0, l2inner(u, v, xr))

    def test_grid(self):
        from discretize import grid, grid1d
        #1D
        np.testing.assert_array_equal([np.linspace(-9.0, 9.0, 10)], grid([10], [-10, 10]))

        #2D using meshgrid
        n = [5, 3]
        xr = np.array([[0, 1], [-2, -1]])
        x0 = grid1d(n[0], xr[0])
        x1 = grid1d(n[1], xr[1])
        xg = np.meshgrid(x0, x1)
        xg[0] = xg[0].T
        xg[1] = xg[1].T
        np.testing.assert_array_equal(xg, grid(n, xr))

        #2D against Matlab's ndgrid + grid1d
        xg = [np.array([[2., 4., 6., 8., 10.]])]
        xg[0] = np.concatenate((xg[0], xg[0])).T
        xg.append(np.array([[-3., -1.] for i in range(5)]))
        x = grid([5, 2], np.array([[1, 11], [-4, 0]]))
        np.testing.assert_array_equal(xg, x)
