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
 
"""This demo program solves the out-of-plane Reissner-Mindlin equations on the
unit square with uniform transverse loading with fully clamped boundary conditions.

We use the Duran-Liberman projection operator expressed in pure UFL where extra
Lagrange multipliers exist on the edge and in the interior of each element to
enforce the compatibility of the rotations and the reduced rotations.
"""

from dolfin import *
from fenics_shells import * 

# Create mesh 
L_x, L_y = 4.0, 1.0
ndiv_x, ndiv_y = 80, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(0.5*L_x, 0.5*L_y), ndiv_x, ndiv_y, 'crossed')

# Duranl-Liberman Lagrange problem space 
# - (rotations, transverse displacement, reduced rotations, facet lagrange multipliers, cell lagrange multipliers)
U_L = DuranLiberman(mesh, space_type="lagrange")

# Material parameters 
E = Constant(1.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.001)

# Define trial, test and function on the function space
u, u_t, u_ = TrialFunction(U_L), TestFunction(U_L), Function(U_L)
theta, w, R_theta, p = split(u)
theta_t, w_t, R_theta_t, p_t = split(u_t)
theta_, w_, R_theta_, p_ = split(u_)

# Define the shear strain on reduced rotations
R_gamma = gamma(R_theta, w)
T = T(R_gamma, E=E, nu=nu, kappa=kappa, t=t)
Pi_s = shear_energy(R_gamma, T)*dx
Pi_s = action(1./t**2*Pi_s, u_)

# Define the Lagrangian multiplier term
# TODO this should be imported as U_L.b_e (as the U_L.reduction_operator)
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

# Define the bending energy in terms of the standard rotation space.
k = k(theta)
M = M(k, E=E, nu=nu, t=t)
Pi_b = bending_energy(k, M)*dx
Pi_b = action(Pi_b, u_)

# Define a uniform transverse loading (when applied)
f = Constant(0.0) 
L = t**3*f*u_t[2]*dx 

# Define the total energy and compute the derivatives
Pi = Pi_s + Pi_b + Pi_R - L
F = derivative(Pi, u_, u_t)
J = derivative(F, u_, u)

# Define subdomains for Dirichlet boundary conditions
allboundary = lambda x, on_boundary: on_boundary
left = lambda x, on_boundary: x[0] <= DOLFIN_EPS and on_boundary 
bottom = lambda x, on_boundary: x[1] <= DOLFIN_EPS and on_boundary 
right = lambda x, on_boundary: abs(x[0] - 0.5*L_x) <= DOLFIN_EPS and on_boundary 
up = lambda x, on_boundary: abs(x[1] - 0.5*L_y) <= DOLFIN_EPS and on_boundary 

# Define the value of the transverse displacement in the middle of the edge
w_exp = Expression("-c*2.0*x[0]/Lx + c", c = 1., Lx = L_x)

# Boundary conditions on transverse displacement and rotation
bc_w_up = DirichletBC(U_L.V, w_exp, up)
bc_R_left = DirichletBC(U_L.R.sub(0), Constant(0.0), left)
bc_R_bottom = DirichletBC(U_L.R.sub(1), Constant(0.0), bottom)
bcs = [bc_w_up, bc_R_left, bc_R_bottom]

# The class DuranLiberman provides a helper utility to automatically construct 
# boundary conditions on the reduced rotations and Lagrange multipliers.
# TO CHECK! THIS IS NOT NEEDED BECAUSE RR AND Q_1 ARE EMPTY !?
# bcs += U_L.auxilliary_dirichlet_conditions(bcs)

# Define the problem
problem = NonlinearVariationalProblem(F, u_, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.maximum_iterations =  10
solver.parameters.snes_solver.absolute_tolerance = 1E-20
solver.parameters.snes_solver.error_on_nonconvergence = True

# Solve the problem
w_exp.c = L_x/2.
solver.solve()
theta_h, w_h, R_theta_h, p_h = u_.split(deepcopy=True)