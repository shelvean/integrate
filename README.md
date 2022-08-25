# integrate
Code to integrate a two dimensional function over a triangulation <br/>
Using Gauss-Jacobi quadrature  <br/>


# Example:
import numpy as np  <br/>
v = np.array([[0.,0.],[1.,0],[0.,1],[1.,1.]) # vertices of a triangle  <br/>
t = np.array([[0,1,2],[1,3,2]]) # nodes defining the triangulation  <br/>
q = 5  <br/>
f = lambda x,y: np.exp(np.sin(x**2+y**2))  <br/>
val = integral(v,t,f,q)  <br/>

