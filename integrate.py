import numpy as np
from scipy.special import roots_jacobi



def integral(v,t,g,q):
# =============================================================================
# # computes the integral of a function g on the mesh [v,t]
# v is an (nv by 2) matrix of the mesh vertices
# t is an (nt by 3) matrix of the mesh triangles 
#  output:   int_{Omega} f(x,y) dxdy
# # uses Stroud conical quadrature at Gauss-Jacobi nodes
#  see Ainsworth, Andriamaro, Davydov, SISC 2011
# test:  g = lambda x,y: 0.*x + 1.0 
#           q = 5
#           val = integral(v,t,g,q)
#          check that val = area of the region
# =============================================================================
    a = area(v,t)
    [t1,w1] = roots_jacobi(q,1,0) # Gauss Jacobi nodes/weights on [-1,1]
    [t2,w2] = roots_jacobi(q,0,0)
    wx = 0.5*w1  # Gauss Jacobi weights on [0,1]
    wy = 0.5*w2
    x,y = stroudnodes(v,t,q) # Stroud nodes on the mesh
    f = g(x,y) # evaluate g on the Stroud nodes
    qs = q*q # number of points per triangle
    w = np.outer(wy,wx).reshape(1,q,q)
    ntri = int(f.shape[0]/qs) # number of triangles
    nn = np.arange(f.size)
    idx = np.floor(nn/qs).astype(int) # find triangle containing Stroud point
    a1 = a[idx] # find area of triangle containing Stroud point
    a1 = a1.reshape(a1.size,1)
    ze = (a1*f).reshape(ntri,q,q)
    v = np.sum(np.tensordot(ze,w,axes=([1,2],[1,2]))) # Stroud quadrature
    return v


def stroudnodes(v,t,q):
    
# =============================================================================
#     tensorized computation of Stroud nodes on the mesh
#     this maps the Gauss-Jacobi points on the square [0,1]^2 to the physical triangles
#     computes Stroud nodes on all triangles at once as a tensor multiplication
#     v is an (nv by 2) matrix of the vertex coordinates
#     t is an (nt by 3) matrix of the mesh triangles
#     q is the number of Gauss-Jacobi points 
#     q^2 is the number of points per triangle
#     output: x, y coordinates of the Stroud points on the mesh.
# =============================================================================
    
    vt = v[t[:,:],:]
    [t11,w11] = roots_jacobi(q,1,0) # Gauss Jacobi nodes on [-1,1]
    [t22,w22] = roots_jacobi(q,0,0)
    t1 = 0.5 + 0.5*t11 # transform Gauss-Jacobi nodes to [0,1]
    t2 = 0.5 + 0.5*t22
    tx,ty = np.meshgrid(t1,t2)
    u = tx # barycentric coordinates at the Gauss-Jacobi nodes on square
    v = ty*(1-u)
    w = 1-u-v
    z = np.multiply.outer(vt[:,0,:],u)+np.multiply.outer(vt[:,1,:],v)+\
        np.multiply.outer(vt[:,2,:],w)
    x = z[:,0,:].flatten() # corresponding points on the mesh
    y = z[:,1,:].flatten()
    return x.reshape(x.size,1),y.reshape(y.size,1)
