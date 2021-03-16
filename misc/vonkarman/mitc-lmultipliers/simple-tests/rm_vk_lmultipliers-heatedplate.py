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
for the heated plate problem (TODO: ADD REF).

We use the Duran-Liberman projection operator expressed in pure UFL where extra
Lagrange multipliers exist on the edge and in the interior of each element to
enforce the compatibility of the rotations and the reduced rotations.
"""

from dolfin import *
from fenics_shells import *
import numpy as np
import mshr
import matplotlib.pyplot as plt

# Create a mesh and define the function space
radius = 1.0
n_div = 10
centre = Point(0.,0.)
domain_area = np.pi*radius**2
geom = mshr.Circle(centre, radius)
mesh = mshr.generate_mesh(geom, n_div)
h_max = mesh.hmax()
# mesh.init()

# Define the spaces
U_L = DuranLiberman(mesh, space_type="lagrange")
V = VectorFunctionSpace(mesh, 'CG', 1)
VK = MixedFunctionSpace([V, U_L.R, U_L.V, U_L.RR, U_L.Q_1])

# Material parameters 
E = Constant(1.0)
t = Constant(1.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
eps = 0.1

# Defines the lenticular thinning of the plate
th_f = Expression('(1.0 - (x[0]*x[0])/(R*R) - (x[1]*x[1])/(R*R))', R=radius)
# th_f = Expression('1.0', R=radius)

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
T = T(R_gamma, E=E, nu=nu, kappa=kappa, t=t*th_f) # shear generalized stress
Pi_s = shear_energy(R_gamma, T)*dx # shear energy
Pi_s = action(1./eps**2*Pi_s, z_)

# Define the target inelastic curvature
k_T = as_tensor(Expression((("1.0*c","0.0*c"),("0.0*c","0.95*c")), c=1.0))

# Define the bending energy in terms of the standard rotation space.
ke = k(theta) - k_T
M_l = M(ke, E=E, nu=nu, t=t*th_f)
Pi_b = bending_energy(ke, M_l)*dx
Pi_b = action((12.*(1.0 - nu**2))*Pi_b, z_)

# Define the (von karman) membrane energy in terms of the standard displacement space
e = von_karman_e(v, grad(w))
N_l = N(e, E=E, nu=nu, t=t*th_f)
Pi_m = membrane_energy(e, N_l)*dx
Pi_m = action(1./eps**2*Pi_m, z_)

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

Pi_R = 1./eps**2*b_e(theta_, R_theta_, p_, U_L)

# Define the boundary conditions
center = lambda x, on_boundary: x[0]**2 + x[1]**2 < (1.E-6*h_max)**2
bc_v = DirichletBC(VK.sub(0), Constant((0.0,0.0)), center, method="pointwise")
bc_R = DirichletBC(VK.sub(1), Constant((0.0,0.0)), center, method="pointwise")
bc_w = DirichletBC(VK.sub(2), Constant(0.0), center, method="pointwise")
bc_RR = DirichletBC(VK.sub(3), Constant((0.0,0.0)), center, method="pointwise")
bc_Q_1 = DirichletBC(VK.sub(4), Constant((0.0,0.0)), center, method="pointwise")
# bcs = [bc_v, bc_R, bc_w, bc_RR, bc_Q_1]
bcs = [bc_v, bc_R, bc_w]

# The class DuranLiberman provides a helper utility to automatically construct 
# boundary conditions on the reduced rotations and Lagrange multipliers.
# TO CHECK HOW TO USE IT!
# bcs += U_L.auxilliary_dirichlet_conditions(bcs)

# Compute the Residual and Jacobian
Pi = Pi_m + Pi_s + Pi_b + Pi_R
F = derivative(Pi, z_, z_t)
J = derivative(F, z_, z)

# Initial guess
init = Function(VK)
z_.assign(init)

# Solution
kx = []
ky = []
kxy = []
ls_load = []

# Analytical critical inelastic curvature (see E. H. Mansfield, 1962)
# c_cr = 5.16
# loadings = np.linspace(0.0, 1.5*c_cr, 500)
loadings = np.linspace(0.0, 4., 500)

# Set problem and solver
problem = NonlinearVariationalProblem(F, z_, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.line_search =  'basic'
solver.parameters.snes_solver.maximum_iterations =  15
solver.parameters.snes_solver.absolute_tolerance = 1E-9
solver.parameters.snes_solver.error_on_nonconvergence = True

# Set plot options
plt.xlabel('load')
plt.ylabel('Average curvatures')
plt.ion()
plt.grid()
plt.show()

for i in loadings:
    k_T.c = i
    solver.solve()

    v_h, theta_h, w_h, R_theta_h, p_h = z_.split(deepcopy=True)
    K_h = project(sym(grad(theta_h)), TensorFunctionSpace(mesh, 'CG', 1))
    Kxx = assemble(K_h[0,0]*dx)/domain_area
    Kyy = assemble(K_h[1,1]*dx)/domain_area
    Kxy = assemble(K_h[0,1]*dx)/domain_area
    ls_load.append(i)
    kx.append(Kxx)
    ky.append(Kyy)
    kxy.append(Kxy)
    plt.plot(ls_load, kx,'k',linewidth=3.5)
    plt.plot(ls_load, ky,'r',linewidth=3.5)
    plt.xlabel('Temperature gradient', fontsize=15)
    plt.ylabel('Average curvatures', fontsize=15)
    plt.draw()

# plt.savefig("output/mansfield.png")
