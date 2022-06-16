import numpy as np


# Dimensions of a discretised function u
# u             discretised function
# output        scalar int
def dim(u):
    return len(list(u.shape))


# Create a valid xrange to describe domain
# input xrange can be
# -1            in which case set xrange to default: [-1, 1] in all dimensions
# list          this needs to be length 2 and is repeated across all dimensions
# ndarray       of size dims x 2 and is returned as it is
# output        dims x 2 ndarray
def fixrange(xrange, dims):
    if isinstance(xrange, int) and xrange == -1:
        return np.array([[-1, 1] for d in range(dims)])
    else:
        if isinstance(xrange, list):
            assert(len(xrange) == 2)
            return np.array([xrange for d in range(dims)])
        else:
            return xrange


# Create a simple one-dimensional grid
# later make a version that allows one end point to be included
# n         scalar int
# xrange    2 length array of reals
# output    n x 1 float
def grid1d(n, xrange):
    offset = (xrange[1] - xrange[0]) / (2 * n)
    return np.linspace(xrange[0] + offset, xrange[1] - offset, n)


# Computes the complex L2 inner product <u, v>
# which is conjugate linear in u and linear in v
# u,v       complex-valued discretised functions discretised
#           on domain described by xrange
# xrange    dims x 2
# output    scalar complex
def l2inner(u, v, xrange=-1):
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape)
    return s * np.inner(u.flatten().conj(), v.flatten())


# Computes the L2 norm ||u||
# u         complex-valued discretised function discretised
#           on domain described by xrange
# xrange    dims x 2
# output    scalar real
def l2norm(u, xrange=-1):
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape)
    return np.sqrt(s) * np.linalg.norm(u.flatten())


# Create an N-dimensional grid
# n         dim length array of int
# xrange    dim x 2 ndarray of reals
# output    dim length list of ndarrays, each of size n_1 x n_2 x ... x n_N
def grid(n, xrange):
    dims = len(n)
    xrange = fixrange(xrange, dims)
    xlist = []
    for i in range(dims):
        xlist.append(grid1d(n[i], xrange[i]))
    x = np.meshgrid(*xlist)
    for i in range(dims):
        x[i] = x[i].T
    return x








#
# def resample(u0, n, xrange=-1):
#     dims = dim(u0)
#     xrange = fixrange(xrange, dims)
#
#     a = xrange[:, 1]
#     b = xrange[:, 2]
#     n0 = u0.shape
#
#     if n == n0:
#         return u0
#     else:
#         uu = cfft(u0)
#
#         midpt = lambda m: (m / 2) + 1 - np.mod(m, 2) / 2
#         midpt0 = midpt(n0)
#         midptn = midpt(n)
#
#         print(midpt0)
#         print(midptn)
#
#         rs = []
#         if n < n0:     # downsample
#             for d in range(dims):
#                 rs.append(np.arange(midpt0[d] - midptn[d] + 1, midpt0[d] + (n[d] - midptn[d]) + 1))
#             un = uu(*rs)  # when both are even this works, otherwise check
#         else:           # upsample
#             for d in range(dims):
#                 rs.append(np.arange(midptn[d] - midpt0[d] + 1, midptn[d] + (n0[d] - midpt0[d]) + 1))
#
#             if dims==1:
#                 un = np.zeros(n,1)  # blank
#             else:
#                 un = np.zeros(n)    # blank
#             np.put(un, *rs, uu)
#
#
#         midgrid = lambda m: a + ((2/m) *  (1/2) * (1-np.mod(m, 2)) + 1) * (b-a)/2
#
#         # this gives the offset
#         o = midgrid(n)-midgrid(n0)
#
#         for d in range(dims):
#             c = -1j * o[d] * np.pi * (np.arange(1., n[d]) - midptn[d]) / ( (b[d]-a[d]) / 2 )
#             un = fourierproduct(np.exp, c, un, d) * np.sqrt(n[d]) / np.sqrt(n0[d])
#
#         return cifft(un)
#
#

