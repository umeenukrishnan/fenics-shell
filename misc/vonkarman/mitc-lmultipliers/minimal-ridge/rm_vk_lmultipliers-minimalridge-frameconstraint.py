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
import numpy as np

# Create the mesh (1/4 of a rectangle)
L_x, L_y = 4.0, 1.0
ndiv = 10
aspect_ratio = L_x/L_y
ndiv_x, ndiv_y = int(aspect_ratio)*ndiv, ndiv

# Rectangular mesh
# mesh = RectangleMesh(Point(0.0, 0.0), Point(0.5*L_x, 0.5*L_y), ndiv_x, ndiv_y, 'crossed')

# Meshes
# meshfile = 'eta4-h001-1-10-50-50.xml'
# meshfile = 'eta4-h001-2-20-100-100.xml'
# meshfile = 'eta4-h001-3-30-150-150.xml'
meshfile = 'eta4-h001-4-40-200-200.xml'
# meshfile = 'eta4-h001-5-50-250-250.xml'
# meshfile = 'eta4-h001-10-100-500-500.xml'
mesh = Mesh(meshfile)

# Mesh properties
h_max = mesh.hmax()
h_min = mesh.hmin()

# Define the spaces
U_L = DuranLiberman(mesh, space_type="lagrange")
V = VectorFunctionSpace(mesh, 'CG', 1)
VK = MixedFunctionSpace([V, U_L.R, U_L.V, U_L.RR, U_L.Q_1])

# Define the material parameters
poisson = 0.3
shear_factor = 5./6.
nu = Constant(poisson)
kappa = Constant(shear_factor)
eps = 0.001 # thickness
lmbd = np.sqrt(eps**2/(12.0*(1 - poisson**2))) # small parameter as in (Lobkovsky, 1996)

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
Pi_s = action((1.0/lmbd**2)*Pi_s, z_)

# Define the bending energy in terms of the standard rotation space.
ke = k(theta)
M_l = M(ke, E=1.0, nu=nu, t=1.0)
Pi_b = bending_energy(ke, M_l)*dx
Pi_b = action((eps**2/lmbd**2)*Pi_b, z_)

# Define the (von karman) membrane energy in terms of the standard displacement space
e = von_karman_e(v, grad(w))
N_l = N(e, E=1.0, nu=nu, t=1.0)
Pi_m = membrane_energy(e, N_l)*dx
Pi_m = action((1.0/lmbd**2)*Pi_m, z_)

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
bottom = lambda x, on_boundary: x[1] <= DOLFIN_EPS and on_boundary
left = lambda x, on_boundary: x[0] <= DOLFIN_EPS and on_boundary
up = lambda x, on_boundary: abs(x[1] - 0.5*L_y) <= DOLFIN_EPS and on_boundary 

# Set the value of the transverse displacement in the middle of the edge
w_exp = Expression("(l - x[0])*sin(c)", c = .1, l = L_x/2.0)
u_exp = Expression("l*(cos(c) - 1.0) + (l - x[0]) + (x[0] - l)*cos(c)", c = .1, l = L_x/2.0)

# Boundary conditions on transverse displacement and rotation
bc_v_bottom = DirichletBC(VK.sub(0).sub(1), Constant(0.0), bottom)
bc_v_left = DirichletBC(VK.sub(0).sub(0), Constant(0.0), left)
bc_v_up = DirichletBC(VK.sub(0).sub(0), u_exp, up)
bc_w_up = DirichletBC(VK.sub(2), w_exp, up)
bc_R_left = DirichletBC(VK.sub(1).sub(0), Constant(0.0), left)
bc_R_bottom = DirichletBC(VK.sub(1).sub(1), Constant(0.0), bottom)
bcs = [bc_v_up, bc_v_left, bc_v_bottom, bc_w_up, bc_R_left, bc_R_bottom]

# The class DuranLiberman provides a helper utility to automatically construct 
# boundary conditions on the reduced rotations and Lagrange multipliers.
# TO CHECK! 
# bcs += U_L.auxilliary_dirichlet_conditions(bcs)

# Compute the Residual and Jacobian
Pi = Pi_m + Pi_s + Pi_b + Pi_R
F = derivative(Pi, z_, z_t)
J = derivative(F, z_, z)

# Initial guess
init = Function(VK)
# init = Function(VK, 'nomefile.xml') # to load previous solution
z_.assign(init)

# Frame angle
angle_max = 45.0*np.pi/180.0 # base angle of the minimal ridge
angle_incr = np.linspace(0.0, angle_max, 200)

# Solution
problem = NonlinearVariationalProblem(F, z_, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.maximum_iterations =  15
solver.parameters.snes_solver.absolute_tolerance = 1E-7
solver.parameters.snes_solver.error_on_nonconvergence = False

# Solution
print '________________________________________________'
print 'Numerical solution of the Minimal Ridge Problem'
print 'Numerical procedure: Duran-Liberman with lagrangian multipliers'
print '- Ridge length = %r' % L_y
print '- Thickness/Length ratio (see DiDonna and Witten, 2001) = %r' % eps
print '- Minimum mesh size = %r' % h_min
print '- Maximum mesh size = %r' % h_max

ls_angle = []
ls_energy = []
ls_sag = []

for count,i in enumerate(angle_incr):

    print '------------------------------------------------'
    print 'Now solving load increment ' + repr(count) + ' of ' + repr(len(angle_incr))
    w_exp.c = i
    u_exp.c = i
    # Solving
    solver.solve()
    v_h, theta_h, w_h, R_theta_h, p_h = z_.split(deepcopy=True)
    energy_m = assemble(Pi_m)
    energy_b = assemble(Pi_b)
    energy_s = assemble(Pi_s)
    energy_p = assemble(Pi_R)
    print("Energy - m:  %s, b: %s, s: %s, p: %s" %(energy_m, energy_b, energy_s, energy_p))

    # Comparison with analytical scaling for the sag
    current_angle = i
    # The terms 2.0*current_angle provides the alpha angle of fig. 9.8, pag. 351 
    # in Audoly and Pomeau, 2010. For the scaling law see (9.68) pag. 361, ibidem.
    # - analytical energy and sag
    energy = 1.505*lmbd**(-1./3.)*(2.0*current_angle)**(7./3.)
    sag = 0.163*L_y*lmbd**(1./3.)*(2.0*current_angle)**(2./3.)
    # - numerical energy and sag (4.0 is because we consider a quarter of the domain)
    energy_h = 4.0*(energy_m + energy_b + energy_s)
    sag_h = (w_h(0.0, 0.5*L_y) - w_h(0.0, 0.0))
    # energy percentage error
    energy_error = 100*(energy_h - energy)/energy
    print '-..-..-..-..-..-..-..-..-..-..-..-..-..-'
    print 'The analytical energy is: %r' % energy
    print 'The numerical energy is: %r' % energy_h
    print 'Energy percentage error is: %r' % energy_error
    print '-..-..-..-..-..-..-..-..-..-..-..-..-..-'
    # sag percentage error
    sag_error = 100*(sag_h - sag)/sag
    print 'Sag percentage error is: %r' % sag_error

    # Angle vs Elastic energy plot 
    ls_angle.append(2.0*current_angle)
    ls_energy.append(energy_h)
    ls_sag.append(sag_h)

# Storing data
np.save('sag' + '-mesh_' + meshfile[:-4], ls_sag)
np.save('angles' + '-mesh_' + meshfile[:-4], ls_angle)
np.save('energy' + '-mesh_' + meshfile[:-4], ls_energy)
np.save('meshdata' + '-mesh_' + meshfile[:-4], [z_.vector().array().size, mesh.hmax(), mesh.hmin()])

# Storing the solution
File('zsol' + '_angle-' + str(round(angle_max, 2)).replace('.', '_') + '_mesh-' + meshfile[:-4] + '.xml') << z_

# -----------------------------
# Plot of ENERGY SCALING
import matplotlib.pyplot as plt
ls_angle.pop(0)
ls_energy.pop(0)
# - energy scaling with numerical prefactor (see Audoly and Pomeau, 2010)
ls_energy_scaling_wp = [1.505*lmbd**(-1./3.)*i**(7./3.) for i in ls_angle]
# - energy scaling with numerical prefactor (see Lobkovsky, 1996)
ls_energy_scaling_wpl = [0.246*lmbd**(-1./3.)*i**(7./3.) for i in ls_angle]
plt.loglog(ls_angle, ls_energy,'ko',linewidth=3.5, label=r'FE')
plt.loglog(ls_angle, ls_energy_scaling_wp,'k',linewidth=3.5, label=r'$1.505\, \lambda^{1/3} \alpha^{7/3}$')
plt.loglog(ls_angle, ls_energy_scaling_wpl,'k',linewidth=3.5, linestyle='--', label=r'$0.246\, \lambda^{1/3} \alpha^{7/3}$')
plt.xlabel(r"$\alpha$", fontsize=20)
plt.ylabel(r"$U\,(\lambda=c)/D$", fontsize=20)
plt.grid()
plt.legend(loc='upper center')
plt.savefig('energyscale' + '-mesh_' + meshfile[:-4] + '.pdf')
# plt.show()
