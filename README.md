# integrate
Code to integrate a two dimensional function over a triangulation
Using Gauss-Jacobi quadrature 
Example:
import numpy as np
v = np.array([[0.,0.],[1.,0],[0.,1],[1.,1.]) # vertices of a triangle
t = np.array([[0,1,2],[1,3,2]]) # nodes defining the triangulation
q = 5
f = lambda x,y: np.exp(np.sin(x**2+y**2))
val = integral(v,t,f,q)

