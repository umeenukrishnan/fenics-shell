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
for the minimal ridge problem (TODO: ADD REF).

We use the Duran-Liberman projection operator expressed in pure UFL where extra
Lagrange multipliers exist on the edge and in the interior of each element to
enforce the compatibility of the rotations and the reduced rotations.
"""

from dolfin import *
from fenics_shells import * 

# Create the mesh (1/4 of a rectangle)
L_x, L_y = 4.0, 1.0
ndiv_x, ndiv_y = 80, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(0.5*L_x, 0.5*L_y), ndiv_x, ndiv_y, 'crossed')

# Define the spaces
U_L = DuranLiberman(mesh, space_type="lagrange")
V = VectorFunctionSpace(mesh, 'CG', 1)
VK = MixedFunctionSpace([V, U_L.R, U_L.V, U_L.RR, U_L.Q_1])

# Material parameters 
E = Constant(1.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.01)

# Define the Trial and Test functions of VK. Define a function on VK.
z, z_t, z_ = TrialFunction(VK), TestFunction(VK), Function(VK)
# Split the Trial and Test functions
v, theta, w, R_theta, p = split(z) 
v_t, theta_t, w_t, R_theta_t, p_t = split(z_t)
v_, theta_, w_, R_theta_, p_ = split(z_)
# Join the transverse unknown for the reduced operator
u = (theta, w, R_theta, p) 
u_t = (theta_t, w_t, R_theta_t, p_t)

# Stabilization parameter for shear energy (see Kere, Lyly, 2005)
# alpha = 1.0
# stab = t**2/(t**2 + alpha*h_max**2)
stab = 1.0

# Define the shear energy
R_gamma = gamma(R_theta, w) # shear strain
T = T(R_gamma, E=E, nu=nu, kappa=kappa, t=t) # shear generalized stress
Pi_s = stab*shear_energy(R_gamma, T)*dx # shear energy
Pi_s = action(Pi_s, z_)

# Define the bending energy in terms of the standard rotation space.
ke = k(theta)
M_l = M(ke, E=E, nu=nu, t=t)
Pi_b = bending_energy(ke, M_l)*dx
Pi_b = action(Pi_b, z_)

# Define the (von karman) membrane energy in terms of the standard displacement space
e = von_karman_e(v, grad(w))
N_l = N(e, E=E, nu=nu, t=t)
Pi_m = membrane_energy(e, N_l)*dx
Pi_m = action(Pi_m, z_)

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

# Define the Subdomains for Dirichlet boundary conditions
# allboundary = lambda x, on_boundary: on_boundary
# right = lambda x, on_boundary: abs(x[0] - 0.5*L_x) <= DOLFIN_EPS and on_boundary 
bottom = lambda x, on_boundary: x[1] <= DOLFIN_EPS and on_boundary 
left = lambda x, on_boundary: x[0] <= DOLFIN_EPS and on_boundary
up = lambda x, on_boundary: abs(x[1] - 0.5*L_y) <= DOLFIN_EPS and on_boundary 

# Set the value of the transverse displacement in the middle of the edge
w_exp = Expression("-c*2.0*x[0]/Lx + c", c = .1, Lx = L_x)

# Boundary conditions on transverse displacement and rotation
bc_v_bottom = DirichletBC(VK.sub(0).sub(1), Constant(0.0), bottom)
bc_v_left = DirichletBC(VK.sub(0).sub(0), Constant(0.0), left)
bc_w_up = DirichletBC(VK.sub(2), w_exp, up)
bc_R_left = DirichletBC(VK.sub(1).sub(0), Constant(0.0), left)
bc_R_bottom = DirichletBC(VK.sub(1).sub(1), Constant(0.0), bottom)
bcs = [bc_v_left, bc_v_bottom, bc_w_up, bc_R_left, bc_R_bottom]

# The class DuranLiberman provides a helper utility to automatically construct 
# boundary conditions on the reduced rotations and Lagrange multipliers.
# TO CHECK! THIS IS NOT NEEDED BECAUSE RR AND Q_1 ARE EMPTY !?
# bcs += U_L.auxilliary_dirichlet_conditions(bcs)

# Compute the Residual and Jacobian
Pi = Pi_m + Pi_s + Pi_b + Pi_R
F = derivative(Pi, z_, z_t)
J = derivative(F, z_, z)

# Initial guess
init = Function(VK)
z_.assign(init)

# Solution
w_exp.c = 0.01
problem = NonlinearVariationalProblem(F, z_, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.maximum_iterations =  50
solver.solve()
v_h, theta_h, w_h, R_theta_h, p_h = z_.split(deepcopy=True)
plot(w_h, interactive=True)

