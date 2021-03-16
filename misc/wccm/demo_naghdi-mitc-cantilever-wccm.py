# Copyright (C) 2016 Matteo Brunetti
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
 
"""This demo program solves the Reissner-Mindlin-Naghdi equations on a
cantilever beam subject to an end moment that results in the beam rolling up
completely on itself.

We use the Duran-Liberman projection operator expressed in pure UFL where extra
Lagrange multipliers exist on the edge of each element to enforce the
compatibility condition.
"""

from dolfin import *
from fenics_shells import *
parameters.form_compiler.quadrature_degree = 2


# Define the mesh
P1, P2 = Point(0.0, -.5), Point(12.0, .5)
mesh = RectangleMesh(P1, P2, 48, 4, "crossed")

# In-plane displacements, rotations, out-of-plane displacements
# shear strains and Lagrange multiplier field.
element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

# Define the Function Space
U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space
U_P = U.projected_space

# Define the material parameters (see Bathe et al., The MITC3+ shell element in geometric nonlinear analysis, 2015)
Y, nu = Constant(1.2E6), Constant(0.0) 
mu = Y/(2.0*(1.0 + nu))
lb = 2.0*mu*nu/(1.0 - 2.0*nu) 
thickness = 1E-1
eps = Constant(0.5*thickness) 

# Define the Trial and Test functions of U. Define a function on U.
u, u_t, u_ = TrialFunction(U_F), TestFunction(U_F), Function(U_F)
v_, theta_, w_, Rgamma_, p_ = split(u_)
z_ = as_vector([v_[0], v_[1], w_])

# Define subdomains for Dirichlet boundary conditions
left = lambda x, on_boundary: x[0] <= DOLFIN_EPS and on_boundary
bc_v = DirichletBC(U.sub(0), Constant((0.0,0.0)), left)
bc_a = DirichletBC(U.sub(1), Constant((0.0,0.0)), left)
bc_w = DirichletBC(U.sub(2), Constant(0.0), left)
bcs = [bc_v, bc_a, bc_w]

# Naghdi strain measures
#~ Director vector
d = lambda theta: as_vector([sin(theta[1])*cos(theta[0]), -sin(theta[0]), cos(theta[1])*cos(theta[0])])
# Deformation gradient
F = lambda u: as_tensor([[1.0, 0.0],[0.0, 1.0],[Constant(0), Constant(0)]]) + grad(u)
# Stretching tensor (1st Naghdi strain measure)
G = lambda F0: 0.5*(F0.T*F0 - Identity(2))
# Curvature tensor (2nd Naghdi strain measure)
K = lambda F0, d: 0.5*(F0.T*grad(d) + grad(d).T*F0)
# Shear strain vector (3rd Naghdi strain measure)
g = lambda F0, d: F0.T*d

# Dual stress tensor (constitutive law)
S = lambda X: 2.0*mu*X + ((2.0*mu*lb)/(2.0*mu + lb))*tr(X)*Identity(2) 

# Define subdomain for boundary condition on tractions
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 12.0) <= DOLFIN_EPS and on_boundary        

right_tractions = Right()
# Create mesh function over cell facets
exterior_facet_domains = FacetFunction("size_t", mesh)
exterior_facet_domains.set_all(0)
right_tractions.mark(exterior_facet_domains, 1)

# Define the measure
ds = Measure("ds")(subdomain_data=exterior_facet_domains)

# Define the traction 
m_right = Expression(('-c'), c = 1.0)

#~ Membrane energy density
psi_G = (2.0*eps)*inner(S(G(F(z_))), G(F(z_)))
#~ Bending energy density
psi_K = ((2.0*eps)**3/12.0)*inner(S(K(F(z_), d(theta_))), K(F(z_), d(theta_)))
#~ Shear energy density
psi_g = eps*2.0*mu*inner(Rgamma_, Rgamma_)
# Stored strain energy density
psi = 0.5*(psi_G + psi_K + psi_g)

# Define external work
W_ext = m_right*theta_[1]*ds(1)

# Compute the Residual and Jacobian.
# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.
L_R = inner_e(g(F(z_), d(theta_)) - Rgamma_, p_)
L_el = psi*dx
L = L_el + L_R - W_ext
F = derivative(L, u_, u_t) # First variation of Pi
J = derivative(F, u_, u) # Compute Jacobian of F

# Create nonlinear variational problem and solve
u_p_ = Function(U_P)
problem = ProjectedNonlinearProblem(U_P, F, u_, u_p_, bcs=bcs, J=J)
solver = NewtonSolver()

# Solver parameters
prm = solver.parameters
prm['error_on_nonconvergence'] = False
prm['maximum_iterations'] = 20
prm['linear_solver'] = "mumps"
prm['absolute_tolerance'] = 1E-7

output_dir = "output/naghdi-mitc-cantilever/"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Solution
import numpy as np
w_ls = []
v_ls = []
Mmax = 50.*np.pi/3.
loadings = np.linspace(0.0, Mmax, 50)

fid = File(output_dir + "solution.pvd")
for j in loadings:
    m_right.c = j
    solver.solve(problem, u_p_.vector())
    v_h, theta_h, w_h, Rgamma_h, p_h = u_.split(deepcopy=True) # extract components
    z_h = project(z_, VectorFunctionSpace(mesh, "CG", 1, dim=3))
    z_h.rename('z', 'z')
    fid << z_h, j
    w_ls.append(w_h(12.0, 0.0))
    v_ls.append(abs(v_h(12.0, 0.0)[0]))

# analytical solution
load_analytical = np.linspace(1E-3, 1.0, 100)
inplane_analytical = 12.0*(1.0 - np.sin(2.0*np.pi*load_analytical)/(2.0*np.pi*load_analytical))
outplane_analytical = 12.0*(1.0 - np.cos(2.0*np.pi*load_analytical))/(2.0*np.pi*load_analytical)

np.savetxt(output_dir + "v.csv", v_ls)
np.savetxt(output_dir + "w.csv", w_ls)
np.savetxt(output_dir + "load.csv", loadings)