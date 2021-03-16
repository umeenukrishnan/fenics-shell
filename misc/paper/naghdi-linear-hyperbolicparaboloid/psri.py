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

# Define the mesh
mesh = RectangleMesh(Point(-0.5, -0.5), Point(0.5, 0.5), 160, 160)
h_max = mesh.hmax()
h_min = mesh.hmin()

# Define the initial configuration
VSpace = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
X0 = Expression(('x[0]','x[1]','x[0]*x[0] - x[1]*x[1]'), degree = 4)
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
# TOCHECK - bathe-chapelle write theta wrt to the CONTRAVARIANT components. Try it.
# theta = th1*g1 + th2*g2
theta = th1*g1contra + th2*g2contra

z_t = as_vector([z1_t, z2_t, z3_t])
# theta_t = th1_t*g1 + th2_t*g2
theta_t = th1_t*g1contra + th2_t*g2contra

# Define the geometric and material parameters
Y, nu = 2e8, 0.3
mu = Y/(2.0*(1.0 + nu))
t = Constant(1E-2)

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
J_b = psi_b*sqrt(deta)*dx
J_m = psi_m*sqrt(deta)*dx
J_s = psi_s*sqrt(deta)*dx
# membrane, shearing energy reduced integration
dx_h = dx(metadata={'quadrature_degree': 2}) # reduced integration
J_mh = psi_m*sqrt(deta)*dx_h
J_sh = psi_s*sqrt(deta)*dx_h
# Arnold and Brezzi Partial Selective Reduced energy
alpha = 1.0
kappa = 1.0/t**2
energy_PSR = J_b + alpha*J_s + (kappa - alpha)*J_sh + alpha*J_m + (kappa - alpha)*J_mh
# energy
energy = (t**3)*energy_PSR

# Define the boundary conditions
left_boundary = lambda x, on_boundary: abs(x[0] + 0.5) <= DOLFIN_EPS and on_boundary

# - clamped boundary conditions
clamp = DirichletBC(U, project(Expression(("0.0", "0.0", "0.0", "0.0", "0.0"), degree = 4), U), left_boundary)
bcs = [clamp]

# Define the External Work
body_force = 8.*t
f = Constant(body_force)
W_ext = f*z3*sqrt(deta)*dx

# Define the residual
#### TOCHECK - sign of the residual (see slack chat)
Pi = energy - W_ext
dPi = derivative(Pi, u, u_t)
J = derivative(dPi, u, u_trial)

# Solve
Amat, bvec = assemble_system(J, dPi, bcs=bcs)
solver = LUSolver("mumps")
print("Number of dofs is %s", dofs)
solver.solve(Amat, u.vector(), bvec)

ux, uy, uz, thx, thy = u.split(deepcopy=True)

# output
elastic_energy = assemble(energy)
scaled_elastic_energy = 1e3*elastic_energy # just to compare with bathe energy (given in Nm)
disp = uz(.5, 0.)

X0in = Expression(('x[0]-x[0]','x[1]-x[1]','x[0]*x[0] - x[1]*x[1]'), degree = 4)
initial_conf = project(X0in, VSpace)
displacement = as_vector([ux, uy, uz])
scale_factor = 1e0 # scale factor to visualiza the deformed configuration
deformed_conf = X0in + scale_factor*displacement
deformed_conf = project(deformed_conf, VSpace)
file1 = File("initial_configuration.pvd")
file1 << initial_conf
file2 = File("deformed_configuration.pvd")
file2 << deformed_conf


# Computing energies
output_dir = "3dplots/"
bending_energy = assemble(1e3*t**3*J_b) #1e3 for comparison with bathe
bending_energy_density = project(1e3*t**3*psi_b, FunctionSpace(mesh, 'CG', 1))
reference_displacement = uz(0.5, 0.0)
file4 = File(output_dir + "bending_energy_density1e-2.pvd")
file4 << bending_energy_density