# Copyright (C) 2015 X
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
 
"""This demo program solves the out-of-plane Reissner-Mindlin-von Karman equations 
for a cantilever (beam-like) plate (TODO: ADD REF).

We use the Duran-Liberman projection operator expressed in pure UFL where extra
Lagrange multipliers exist on the edge and in the interior of each element to
enforce the compatibility of the rotations and the reduced rotations.
"""

from dolfin import *
from fenics_shells import *
import numpy as np
import mshr
import matplotlib.pyplot as plt

# Define the mesh
Lx, Ly = 2.0, 1.0
aspect_ratio = (Lx/Ly)
ndiv = 10
ndiv_x, ndiv_y = int(aspect_ratio)*ndiv, ndiv
mesh = RectangleMesh(Point(-0.5*Lx, -0.5*Ly), Point(0.5*Lx, 0.5*Ly), ndiv_x, ndiv_y, 'crossed')
h_max = mesh.hmax()

# Define the function spaces
U_L = DuranLiberman(mesh, space_type="lagrange")
V = VectorFunctionSpace(mesh, 'CG', 1)
VK = MixedFunctionSpace([V, U_L.R, U_L.V, U_L.RR, U_L.Q_1])

# Define the material parameters
poisson = 0.3
shear_factor = 5./6.
nu = Constant(poisson)
kappa = Constant(shear_factor)
eps = 0.01

# Define the Trial and Test functions of VK. Define a function on VK.
z, z_t, z_ = TrialFunction(VK), TestFunction(VK), Function(VK)
# Split the Trial and Test functions
v, theta, w, R_theta, p = split(z) 
v_t, theta_t, w_t, R_theta_t, p_t = split(z_t)
v_, theta_, w_, R_theta_, p_ = split(z_)
# Join the transverse unknown for the reduced operator
u = (theta, w, R_theta, p) 
u_t = (theta_t, w_t, R_theta_t, p_t)

# Define the shear energy
R_gamma = gamma(R_theta, w) # shear strain
T = T(R_gamma, E=1.0, nu=nu, kappa=kappa, t=1.0) # shear generalized stress
Pi_s = shear_energy(R_gamma, T)*dx # shear energy
Pi_s = action(12.0*(1.0 - nu**2)*(1.0/eps**2)*Pi_s, z_)

# Define the bending energy in terms of the standard rotation space.
ke = k(theta)
M_l = M(ke, E=1.0, nu=nu, t=1.0)
Pi_b = bending_energy(ke, M_l)*dx
Pi_b = action(12.0*(1.0 - nu**2)*Pi_b, z_)

# Define the (von karman) membrane energy in terms of the standard displacement space
e = von_karman_e(v, grad(w))
N_l = N(e, E=1.0, nu=nu, t=1.0)
Pi_m = membrane_energy(e, N_l)*dx
Pi_m = action(12.0*(1.0 - nu**2)*(1.0/eps**2)*Pi_m, z_)

# The reduction operator. Details about how this works can be found in the
# documentation of the class DuranLiberman. Here a measure is attached on the entire mesh,
# on the assumption that you wish to cure locking everywhere!
# R_h = U_L.reduction_operator(u, u_t)

# Define the lagrangian multiplier energy term
def b_e(theta, R_theta, p_t, U):
    r"""Return bilinear form related to constructing reduced rotation on element edges.

    .. note:: For use in conjunction with DuranLiberman or MITC7 with space_type='lagrange'.

    .. math::
        b_e(\theta, R_h(\theta); \tilde{p}) := \sum_{e} \int_{e} ((\theta -
        R_h(\theta)) \cdot t) (\tilde{p} \cdot t) \; ds = 0 \quad \forall
        \tilde{p} \in Q_1
    """
    n = FacetNormal(U.mesh())
    t = as_vector((-n[1], n[0]))
    return (inner(theta - R_theta, t)*inner(p_t, t))('+')*dS + inner(theta - R_theta, t)*inner(p_t, t)*ds

Pi_R = b_e(theta_, R_theta_, p_, U_L)

# Define subdomains for Dirichlet boundary conditions
allboundary = lambda x, on_boundary: on_boundary
left = lambda x, on_boundary: abs(x[0] + 0.5*Lx) <= DOLFIN_EPS and on_boundary
right = lambda x, on_boundary: abs(x[0] - 0.5*Lx) <= DOLFIN_EPS and on_boundary
# Define Dirichlet boundary conditions
bc_v = DirichletBC(VK.sub(0), Constant((0.0, 0.0)), left)
bc_R = DirichletBC(VK.sub(1), Constant((0.0, 0.0)), left)
bc_w = DirichletBC(VK.sub(2), Constant(0.0), left)
# Collect Dirichlet boundary conditions
bcs = [bc_v, bc_R, bc_w]

# The class DuranLiberman provides a helper utility to automatically construct 
# boundary conditions on the reduced rotations and Lagrange multipliers.
# TO CHECK HOW TO USE IT!
# bcs += U_L.auxilliary_dirichlet_conditions(bcs)

# Define subdomain for boundary condition on tractions
class RightB(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.5*Lx) <= DOLFIN_EPS and on_boundary        

right_tractions = RightB()
# Create mesh function over cell facets
exterior_facet_domains = FacetFunction("size_t", mesh)
exterior_facet_domains.set_all(0)
right_tractions.mark(exterior_facet_domains, 1)
# Define the measure
ds = Measure("ds")[exterior_facet_domains]

# Define the traction 
t_right = Expression(('-c'), c = .1)

# Define imperfection and external work
f = Constant(1E-4)
L = f*w_*dx + t_right*v_[0]*ds(1)

# Compute the Residual and Jacobian
Pi = Pi_m + Pi_s + Pi_b + Pi_R - L
F = derivative(Pi, z_, z_t)
J = derivative(F, z_, z)

# Initial guess
init = Function(VK)
z_.assign(init)

# Analytical buckling load (cantilever beam)
inertia = Ly*eps**3/12.0
F_cr = 0.25*np.pi**2*inertia/Lx**2
f_cr = F_cr*(12.0*(1.0 - poisson**2)*(1.0/eps**3)) 

# Define the loadings vector
loadings = np.linspace(0.0, 1.025*f_cr , 60)

# Set the problem and the solver parameters
problem = NonlinearVariationalProblem(F, z_, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.line_search =  'basic'
solver.parameters.snes_solver.maximum_iterations =  15
solver.parameters.snes_solver.error_on_nonconvergence = False

# Solve
ls_w = []
for i in loadings:
	t_right.c = i
	solver.solve()

	v_h, theta_h, w_h, R_theta_h, p_h = z_.split(deepcopy=True)
	ls_w.append(w_h(Lx/2.0, 0.0))

# Post-process
loadings = np.delete(loadings, 0)
ls_w = np.delete(ls_w, 0)
plt.plot(ls_w, loadings, color='0.25', marker='o', linewidth=3.5, label='FE (Reissner-Mindlin-von Karman)')
plt.axhline(y=f_cr, color='b', linewidth=1.5, label='Analytical critical load')
plt.xlabel('free-end displacement', fontsize=15)
plt.ylabel('traction', fontsize=15)
plt.xlim(ls_w[0], ls_w[-1])
plt.grid()
plt.legend(loc=4)
plt.savefig("rm_vk_buckling.png")
plt.show()
    