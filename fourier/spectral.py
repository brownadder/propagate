import numpy as np


# f         complex-valued data
# d         dimension in which to do fft. default (d = -1) for fft in all directions
# central fft - performs fourier transform while shifting frequency to centre
# NOTE: scaled different from matlab implementation
def cfft(f, d=-1):
    if d == -1:
        return np.fft.fftshift(np.fft.fftn(f))
    else:
        return np.fft.fftshift(np.fft.fft(f, axis=d), d)


# inverse of cfft
def cifft(f, d=-1):
    if d == -1:
        return np.fft.ifftn(np.fft.ifftshift(f))
    else:
        return np.fft.ifft(np.fft.ifftshift(f, d), axis=d)


def cfftmatrix(n):
    I = np.eye(n)
    F = np.fft.fftshift(np.fft.fft(I, axis=0), axes=0) / np.sqrt(n)
    return F


# n         scalar int
# xrange    2 length array of reals
# creates the fourier symbol in a single dimension
def fouriersymbol(n, xrange):
    lf = 2 / (xrange[1] - xrange[0])
    o = np.mod(n, 2)
    c = 1j * np.pi * lf * np.arange(-(n - o) / 2, (n - o) / 2 + o).T
    return c


#when a full grid of the fourier symbol is required - this is helpful if
#one is using cfftn (or cfft(u,1:D)), i.e. FFT is first run in all
#directions and then there is pointwise multiplication. This symbol can
#help compute the Laplacian efficiently.
# n         dim length array int
# xrange    dim x 2 length array of reals (for 1D xrange should be [[x0,x1]])
# storefn   for gpu/cpu storage
# creates the fourier symbol in all dimensions
def fouriersymbolfull(n, xrange, storefn = lambda x: x):
    dims = xrange.shape[0]
    if len(n) < dims:
        n = n + [max(n) for i in range(dims-len(n))]  #best to specify n for each dim, otherwise max(n) is used
    clist = [storefn(fouriersymbol(n[d], xrange[d])) for d in range(dims)]
    return np.meshgrid(*clist)


def laplaciansymbol(n, xrange, storefn = lambda x: x):
    c = fouriersymbolfull(n, xrange, storefn)
    dims = xrange.shape[0]
    lapsymb = np.zeros(c[0].shape)
    for d in range(dims):
        lapsymb = lapsymb + c[d]**2
    return lapsymb