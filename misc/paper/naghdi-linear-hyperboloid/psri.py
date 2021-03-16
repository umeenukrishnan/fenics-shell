# Copyright (C) 2016 Matteo Brunetti.
#
# This file is part of fenics-shells.
#
# fenics-shells is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fenics-shells is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with fenics-shells. If not, see <http://www.gnu.org/licenses/>.

"""This demo program solves the Reissner-Mindlin-Naghdi linear equations.

Implement here the discretisation proposen by Arnold and Brezzi:
MATHEMATICS OF COMPUTATION, Volume 66, Number 217, January 1997, Pages 1-14
https://www.ima.umn.edu/~arnold//papers/shellelt.pdf
"""

import os
import sys
from dolfin import *

import numpy as np
import mshr
parameters.form_compiler.quadrature_degree = 4
parameters["form_compiler"]["representation"] = "uflacs"

Ndiv = 30
thickness = 1E-3
bl = 3*np.sqrt(thickness) # size of the inner region (boundary layer)
# bl = 6*np.sqrt(1E-4) # size of the inner region (boundary layer)
mesh = RectangleMesh(Point(0., 0.), Point(0.5*np.pi, -1.0), 2*Ndiv, Ndiv)
cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)

# Mesh refinement
hmin_ref = bl/18.0
hmin = mesh.hmin()

while (hmin > hmin_ref):
	for cell in cells(mesh):
		yc = MeshEntity.midpoint(cell).y()
		urv = -1.0 + bl
		if yc < urv:
			cell_markers[cell] = True
	refmesh = refine(mesh, cell_markers)
	hmin = refmesh.hmin()
	mesh = refmesh
	cell_markers = CellFunction("bool", mesh)
	cell_markers.set_all(False)


h_max = mesh.hmax()
h_min = mesh.hmin()

# Define the initial configuration
VSpace = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
X0 = Expression(('cos(x[0])*cosh(x[1])','sin(x[0])*cosh(x[1])','sinh(x[1])'), degree = 4)
X0 = interpolate(X0, VSpace)

# first fundamental form
gradX0 = grad(X0)
acov = gradX0.T*gradX0
acontra = inv(acov)
deta = det(acov)

g1 = X0.dx(0)
g2 = X0.dx(1)

g1contra = acontra[0,0]*g1 + acontra[0,1]*g2
g2contra = acontra[1,0]*g1 + acontra[1,1]*g2

# second fundamental form
N0 = cross(X0.dx(0), X0.dx(1))/sqrt(dot(cross(X0.dx(0), X0.dx(1)), cross(X0.dx(0), X0.dx(1))))
bcov = -0.5*(grad(X0).T*grad(N0) + grad(N0).T*grad(X0))
gradN0 = grad(N0)

# Define the element (Arnold and Brezzi)
lagr = FiniteElement("Lagrange", triangle, degree = 2)
bubble = FiniteElement("B", triangle, degree = 3)
en_element = lagr + bubble
mixed_element = MixedElement(3*[en_element] + 2*[en_element])

# Define the Function Space
U = FunctionSpace(mesh, mixed_element)
dofs = U.dim()
u = Function(U)
u_trial, u_t = TrialFunction(U), TestFunction(U)
z1, z2, z3, th1, th2 = split(u) # components
z1_t, z2_t, z3_t, th1_t, th2_t = split(u_t)

z = as_vector([z1, z2, z3]) # displacement vector
# theta = th1*g1 + th2*g2
theta = th1*g1contra + th2*g2contra

z_t = as_vector([z1_t, z2_t, z3_t])
theta_t = th1_t*g1 + th2_t*g2

# Define the geometric and material parameters
Y, nu = 2e11, 1./3.
mu = Y/(2.0*(1.0 + nu))
t = Constant(thickness)

## Define the linear Naghdi strain measures
e_naghdi = lambda v: 0.5*(gradX0.T*grad(v) + grad(v).T*gradX0) # stretching tensor
k_naghdi = lambda v, phi: -0.5*(gradX0.T*grad(phi) + grad(phi).T*gradX0) - 0.5*(gradN0.T*grad(v) + grad(v).T*gradN0) # curvature tensor
g_naghdi = lambda v, phi: gradX0.T*phi + grad(v).T*N0 # shear strain vector

# Define the Kinematics
e_eff = e_naghdi(z)
k_eff = k_naghdi(z, theta)
g_eff = g_naghdi(z, theta)

# Define the Constitutive properties and the Constitutive law (TOCHECK!, see Le Dret, a posteriori analysis)
lb = 2.0*mu*nu/(1.0 - 2.0*nu)
i, j, k, l = Index(), Index(), Index(), Index()
Am = as_tensor((((2.0*lb*mu)/(lb + 2.0*mu))*acontra[i,j]*acontra[k,l] + 1.0*mu*(acontra[i,k]*acontra[j,l] + acontra[i,l]*acontra[j,k])),[i,j,k,l])
A = Am
D = (1.0/12.0)*Am
G = mu*acontra

# Define the Generalized forces
N = as_tensor((A[i,j,k,l]*e_eff[k,l]),[i, j])
M = as_tensor((D[i,j,k,l]*k_eff[k,l]),[i, j])
T = as_tensor((G[i,j]*g_eff[j]), [i])

# Define the energies
psi_m = .5*inner(N, e_eff) # Membrane energy density
psi_b = .5*inner(M, k_eff) # Bending energy density
psi_s = .5*inner(T, g_eff) # Shear energy density

# bending, membrane, shearing energy
kappa = 1.0/t**2
circumradius = Circumradius(mesh)
alpha = circumradius**2
J_b = psi_b*sqrt(deta)*dx
J_m = alpha*psi_m*sqrt(deta)*dx
J_s = alpha*psi_s*sqrt(deta)*dx
# membrane, shearing energy reduced integration
dx_h = dx(metadata={'quadrature_degree': 2 }) # reduced integration
J_mh = (kappa -alpha)*psi_m*sqrt(deta)*dx_h
J_sh = (kappa -alpha)*psi_s*sqrt(deta)*dx_h
energy_PSR = J_b + J_m + J_mh + J_s + J_sh

# energy
energy = (t**3)*energy_PSR

# Define the boundary conditions
left_boundary = lambda x, on_boundary: abs(x[0]) <= DOLFIN_EPS and on_boundary
right_boundary = lambda x, on_boundary: abs(x[0] - np.pi/2) <= DOLFIN_EPS and on_boundary
bottom_boundary = lambda x, on_boundary: abs(x[1] + 1.0) <= DOLFIN_EPS and on_boundary
up_boundary = lambda x, on_boundary: abs(x[1]) <= DOLFIN_EPS and on_boundary

# - clamped boundary conditions
clamp = DirichletBC(U, project(Expression(("0.0", "0.0", "0.0", "0.0", "0.0"), degree = 4), U), bottom_boundary)

bc_u_right = DirichletBC(U.sub(0), project(Constant(0.0), U.sub(0).collapse()), right_boundary)
bc_th_right = DirichletBC(U.sub(3), project(Constant(0.0), U.sub(3).collapse()), right_boundary)

bc_u_up = DirichletBC(U.sub(2), project(Constant(0.0), U.sub(2).collapse()), up_boundary)
bc_th_up = DirichletBC(U.sub(4), project(Constant(0.0), U.sub(4).collapse()), up_boundary)

bc_u_left = DirichletBC(U.sub(1), project(Constant(0.0), U.sub(1).collapse()), left_boundary)
bc_th_left = DirichletBC(U.sub(3), project(Constant(0.0), U.sub(3).collapse()), left_boundary)

bcs = [clamp, bc_u_right, bc_th_right, bc_u_up, bc_th_up, bc_u_left, bc_th_left]

# Define the External Work
f = Expression("p0*cos(2.0*x[0])", p0=5.0*1E6, degree=4)
zn = dot(z, N0)
W_ext = f*zn*sqrt(deta)*dx

# Define the residual
Pi = energy - W_ext
dPi = derivative(Pi, u, u_t)
J = derivative(dPi, u, u_trial)

# Solve
Amat, bvec = assemble_system(J, dPi, bcs=bcs)
solver = LUSolver("mumps")
print("Number of dofs is %s", dofs)
solver.solve(Amat, u.vector(), bvec)

ux, uy, uz, thx, thy = u.split(deepcopy=True)
elastic_energy = assemble(energy)
whole_elastic_energy = 8.0*elastic_energy

output_dir = "3dplots/"
X0in = Expression(('cos(x[0])*cosh(x[1]) - x[0]','sin(x[0])*cosh(x[1]) - x[1]','sinh(x[1])'), degree = 4)
initial_conf = project(X0in, VSpace)
displacement = as_vector([ux, uy, uz])
scale_factor = 1e0 # scale factor to visualize the deformed configuration
deformed_conf = X0in + scale_factor*displacement
deformed_conf = project(deformed_conf, VSpace)
file1 = File(output_dir + "initial_configuration.pvd")
file1 << initial_conf
file2 = File(output_dir + "deformed_configuration.pvd")
file2 << deformed_conf