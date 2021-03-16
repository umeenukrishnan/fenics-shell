# Copyright (C) 2015 Jack S. Hale
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

We use the Duran-Liberman projection operator to alleviate the problem of shear
locking in the Kirchhoff limit."""

from dolfin import *
from fenics_shells import * 

class VonKarmanEquation(NonlinearProblem):
    def __init__(self, a_b, a_s, c, d, bcs, u_f_, u_p_, U_F, R_e, F, F_s):
        NonlinearProblem.__init__(self)
        self.a_b = a_b
        self.a_s = a_s
        self.c = c
        self.d = d
        self.bcs = bcs
        self.u_f_ = u_f_
        self.u_p_ = u_p_
        self.U_F = U_F
        self.R_e = R_e
        self.F = F
        self.F_s = F_s
    def form(self, A, b, x):
        # Assembling A
        A = projected_assemble(self.a_b, self.a_s, self.c, self.d)
        [bc.apply(A) for bc in self.bcs]
        # Assembling L
        assign(self.u_f_.sub(0), self.u_p_.sub(0))
        assign(self.u_f_.sub(1), self.u_p_.sub(1))
        RR = self.U_F.extract_sub_space([2]).collapse()
        R_theta = TrialFunction(RR)
        R_theta_t = TestFunction(RR)
        R_theta_ = Function(RR)
        P = assemble(self.R_e(R_theta, R_theta_t))
        b_theta = assemble(self.R_e(u_f_.sub(0), R_theta_t))
        solver = LUSolver("umfpack")
        solver.solve(P, R_theta_.vector(), b_theta)
        assign(self.u_f_.sub(2), R_theta_)
        b = projected_assemble(self.F, self.F_s, self.c, self.d)
        [bc.apply(b) for bc in self.bcs]
    def F(self, b, x):
        pass
    def J(self, A, x):
        pass

# Create a mesh and define the full function space
# and the primal function space
mesh = UnitSquareMesh(15, 15)
mesh.init()
# Full problem space (rotations, transverse displacement, shear)
U_F = DuranLiberman(mesh, space_type="full")

# Material parameters 
E = Constant(1.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.01)

# Now we define our bending energy on the problem space associated with only
# the primal variables. We will not apply our projection opeator here as this
# part of the energy of the plate does not contribute to shear locking.

# Primal problem space (where our final linear system is constructed)
U = DuranLiberman(mesh, space_type="primal")
u, u_t, u_p_ = TrialFunction(U), TestFunction(U), Function(U)
theta, w = split(u)
theta_t, w_t = split(u_t) # test functions are suffixed with _t

# Bending energy
M_l = M(k(theta), E=E, nu=nu, t=t)
Pi_b = bending_energy(k(theta), M_l)*dx
Pi_b = action(Pi_b, u_p_)
F_b = derivative(Pi_b, u_p_, u_t) 
a_b = derivative(F_b, u_p_, u)

# Uniform transverse loading
f = Constant(1.0) 
L = t**3*f*w_t*dx 

F = F_b - L

# Now boundary conditions. These should be defined on the primal space.
all_boundary = lambda x, on_boundary: on_boundary

# Clamped everywhere on boundary.
bcs = [DirichletBC(U, Constant((0.0, 0.0, 0.0)), all_boundary)]

# Full problem space
u, u_t, u_f_ = TrialFunction(U_F), TestFunction(U_F), Function(U_F)
theta, w, R_theta = split(u)
theta_t, w_t, R_theta_t = split(u_t) # test functions are suffixed with _t

# The first thing we will do is construct our Duran-Liberman reduction
# operator. We do this by constructing the reduction operator which takes the
# shear strain defined by the generalised displacements (theta, w) to the
# reduced rotation space R_theta.
# c = R_e(gamma(theta, w), R_theta_t, mesh) + R_e(gamma(theta_t, w_t), R_theta, mesh)
# d = R_e(R_theta, R_theta_t, mesh)
c = R_e(gamma(theta, w), R_theta_t) + R_e(gamma(theta_t, w_t), R_theta)
d = R_e(R_theta, R_theta_t)

# Then, we define the shear energy in terms of the reduced rotation space. Note
# that we must attach out own measure dx. This is so you can easily define
# multi-material problems if you want to.
R_gamma = gamma(R_theta, w)
T_l = T(R_gamma, E=E, nu=nu, kappa=kappa, t=t)
Pi_s = shear_energy(R_gamma, T_l)*dx
Pi_s = action(Pi_s, u_f_)
# residual
F_s = derivative(Pi_s, u_f_, u_t)
# Jacobian. We will assemble this later.
a_s = derivative(F_s, u_f_, u)

# Now we use projected_assemble to assemble a_b in the normal way, and apply
# the action of our projection R on a_s. This returns a matrix defined on
# the primal space (theta, z) only.
# to add consistent residuals 
A = PETScMatrix()
b = PETScVector()
problem = VonKarmanEquation(a_b, a_s, c, d, bcs, u_f_, u_p_, U_F, R_e, F, F_s)

# Solver
solver = PETScSNESSolver()

prm = solver.parameters
prm['linear_solver'] = "umfpack"
prm['relative_tolerance'] = 1E-10
prm['absolute_tolerance'] = 1E-10
prm['maximum_iterations'] = 20
prm['line_search'] = "basic"
prm['solution_tolerance'] = 1E-10
prm['report'] = True
# PETScOptions.set("snes_view", True)
#PETScOptions.set("snes_convergence_test", "skip")

# import sys
# sys.exit("STOP!")
solver.solve(problem, u_p_.vector())
