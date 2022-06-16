from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from operators import *

# a = np.array([[1,2,3],[3,-5,2]])
# print(a[0])


xrange = [0, 2*np.pi]
x = grid1d(100, xrange)     #
s = np.sin(x)
ds = diffop(0,1,s,xrange)   #
d2s = diffop(0,2,s,xrange)  #

plt.plot(x, s)
plt.plot(x, np.real(ds))
plt.plot(x, np.real(d2s))
plt.show()

D2 = diffmatrix(2, 100, xrange)
plt.plot(np.real(D2.dot(s)))
plt.show()

print(np.linalg.norm(D2.dot(s)-d2s))
print(l2norm(D2.dot(s)-d2s, xrange))
print(np.sqrt(np.real(l2inner(D2.dot(s)-d2s, D2.dot(s)-d2s, xrange))))

# ----------------------------

a = np.array([[1,2,3],[3,-5,2]])
print(a)
np.put(a, [[0,1],[0,1]], np.array([[5,5],[5,5]]) )
print(a)

# ----------------------------

n = 200
xr = [-10, 10]
x = grid1d(n, xr)

x0 = -2.0
u = np.exp(-(x-x0)**2/(2*0.25))
V = x**4 - 10*x**2

eLu = lambda h, u: diffopexp(0, 2, 1j*h, u, xr)
eVu = lambda h, u: np.exp(-1j*h*V)*u
strang = lambda h, u: eVu(h/2, eLu(h, eVu(h/2, u)))

D2 = diffmatrix(2, n, xr)
H = -D2 + np.diag(V)
exact = lambda h, u: expm(-1j*h*H).dot(u)

T = 0.5
uref = exact(T, u)
ustrang = strang(T, u)

plt.plot(x, np.abs(uref))
plt.plot(x, np.abs(ustrang))
plt.show()

print(l2norm(uref-ustrang, xr))


def runstrang(T, N, u0):
    u = u0
    h = T/N
    for i in range(N):
        u = strang(h, u)
    return u


plt.plot(np.abs(runstrang(T,1000,u)))
plt.show()


Nlist = 2**np.arange(2,9)
hlist = T/Nlist
err = [l2norm(uref-runstrang(T,N,u)) for N in Nlist]
print(err)
print(Nlist)
print(hlist)
plt.loglog(hlist, err)
plt.loglog(hlist, hlist**2)
plt.xlabel('time step')
plt.ylabel('L2 error')
plt.legend(['error in strang', 'O(h^2)'])
plt.show()


# y = grid([100], np.array([xrange]))
# print(y)
#
# print(dim(s))
#
# #c = cfft(s)
# c = cfft(s,0)
# plt.plot(np.real(c))
# plt.show()
#
# #fs = fouriersymbol(4,[-1,1])
# #print(fs)
#
# # a = np.array([1,5,2])
# # n = len(a)
# # shp = [n if d==2 else 1 for d in range(5)]
# # print(a.reshape(shp))
#
#
#
# # plt.plot(x,s)
# # plt.show()
#
# n = [3,4]
# xrange = [[-2,-1],[0,1]]
# X,Y = grid(n,xrange)
# print(X)
# print(Y)
#
# print(list(X.shape))
# print(X.shape[1])
#
# print(dim(X))
#
#
# print(np.multiply(np.sin(X),Y))
#
#

