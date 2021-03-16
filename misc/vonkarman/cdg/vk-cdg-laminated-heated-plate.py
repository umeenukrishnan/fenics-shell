# Copyright (C) 2015 Matteo Brunetti
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

"""This demo program solves the von Karman equations on a composite plate with elliptic
planform and lenticular cross-section. The plate is free on the boundary. 

The analytical solution can be found in the paper:

- A. Fernandes, C. Maurini and S. Vidoli, "Multiparameter actuation and shape 
control of bistable composite plates." International Journal of Solids and Structures,  Vol. 47. Issue 10. Elsevier, 2010.

In this demo the stacking sequence is [-45/+45/+45/-45/+45/-45/-45/+45] with 
carbon-epoxy layers. For further information about the material, see:

- C. Maurini, A. Vincenti, and S. Vidoli, "Modelling and design of anisotropic 
multistable shells." ECCM 2010, Paris, France, 2010.

We use the Continuous/Discontinuous Galerkin formulation to exploit standard
Lagrangian elements for the transverse displacement discretization
"""

from dolfin import *
from fenics_shells import *
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required to run this demo.") 

try:
    import mshr
except ImportError:
    raise ImportError("mshr is required to run this demo.")

# Define relevant parameters and parse from command-line
user = Parameters("user")
user.add("thickness", 1.0)
user.add("nelements", 80)
user.add("interpolation_u", 2)
user.add("interpolation_w", 2)
try:
    parameters.add(user)
except:
    pass
parameters.parse()

# Information about parameters
info(parameters, False)

p_user = parameters["user"]

# Elliptical or circular plate
centre = Point(0.,0.)
a_rad = 1.0
b_rad = 0.5
domain_area = np.pi*a_rad*b_rad
geom = mshr.Ellipse(centre, a_rad, b_rad)
mesh = mshr.generate_mesh(geom, p_user.nelements)
mesh.init()
h_max = mesh.hmax()
h_min = mesh.hmin()

# Problem space (in-plane and transverse displacement)
V = VectorFunctionSpace(mesh, "CG", p_user.interpolation_u)
W = ContinuousDiscontinuous(mesh, n = p_user.interpolation_w)

U = MixedFunctionSpace([V, W])

# Trial and Test function
u = TrialFunction(U)
u_t = TestFunction(U)
u_ = Function(U)
(v, w) = split(u)

# Fix the value in the centre to eliminate the nullspace
def center(x,on_boundary):
    return x[0]**2 + x[1]**2 < (0.5*h_max)**2

bc_v = DirichletBC(U.sub(0), Constant((0.0,0.0)), center, method="pointwise")
bc_w = DirichletBC(U.sub(1), Constant(0.0), center, method="pointwise")
bcs = [bc_v, bc_w]

# Von Karman kinematics
theta = kirchhoff_love_theta(w)
e_ = von_karman_e(v, theta)
k_ = k(theta)

# Properties of the composite (lenticular thickness, stacking sequence and material)
h = interpolate(Expression('(1.0 - (x[0]*x[0])/(a*a) - (x[1]*x[1])/(b*b))', a=a_rad, b=b_rad), FunctionSpace(mesh, 'CG', 2))
thetas = [np.pi/4., -np.pi/4., -np.pi/4., np.pi/4., -np.pi/4., np.pi/4., np.pi/4., -np.pi/4.]
n_layers= len(thetas) # number of layers
hs = h*np.ones(n_layers)/n_layers # thickness of the layers (assuming they are equal)
E2 = interpolate(Expression("1."), FunctionSpace(mesh, 'DG', 0)) # reference Young modulus

# Calculate the laminates matrices (ufl matrices)
A, B, D = laminates.ABD(40.038*E2, E2, 0.5*E2, 0.25, hs, thetas)

# Target inelastic curvature
k_T = as_tensor(Expression((("1.0*c","0.0*c"),("0.0*c","1.0*c")), c=1.0))

# Define the energy using Voigt notation
ev = strain_to_voigt(e_)
kv = strain_to_voigt(k_)
kv_T = strain_to_voigt(k_T)

# Interpolation of the stiffness matrices
Ai = project(A, mesh=mesh)
Bi = project(B, mesh=mesh)
Di = project(D, mesh=mesh)

# Elastic energy
Pi_m = .5*dot(Ai*ev, ev)*dx
Pi_b = .5*dot(Di*(kv - kv_T), (kv - kv_T))*dx
Pi_mb = dot(Bi*(kv - kv_T), ev)*dx
Pi_elastic = Pi_m + Pi_b + Pi_mb

# Discontinuos contribution to the elastic energy
Mv = Di*(kv - kv_T) + Bi*ev # Bending moment in voigt notation
M_ = stress_from_voigt(Mv) # Convertion to tensor notation

# Set the stabilization constant alpha as the norm of the bending stiffness matrix D
alpha = Constant(1.0*np.linalg.norm(Di(0,0)))
Pi_cdg = cdg_energy(theta, M_, alpha, mesh)

# Total energy
Pi = Pi_elastic + Pi_cdg

# Residuals and Jacobian
Pi = action(Pi, u_)
F = derivative(Pi, u_, u_t)
J = derivative(F, u_, u)

# Initial guess
init = Function(U)
u_.assign(init)

# Solver settings
problem = NonlinearVariationalProblem(F, u_, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.maximum_iterations =  30
solver.parameters.snes_solver.absolute_tolerance = 1E-8

# Analytical solution (see A. Fernandes et al., 2010)
# - dimensionless material parameters
beta = Ai(0,0)[4]/Ai(0,0)[0]
nu = Ai(0,0)[1]/Ai(0,0)[0]
gamma = Ai(0,0)[8]/Ai(0,0)[0]
rho = gamma/(1.0 - (nu**2/beta))
mu = nu/np.sqrt(beta) 
eta = gamma/np.sqrt(beta)
Dstar = np.array([[1.0, mu, 0.0], [mu, 1.0, 0.0], [0.0, 0.0, eta]])
# - analytical dimensionless critical values (see A. Fernandes et al., 2010)
hQ = np.sqrt(2*eta)*(1.0 + 2.0*eta + mu)/(1.0 + mu) # since kTx = kTy = h, hQ as in the reference paper
hP_pos = 2*np.sqrt(1.0 - mu)/(1.0 + mu) # since kTx = kTy = h, hP+ as in the reference paper
h_cr = np.minimum(hQ, hP_pos)
# - characteristic radius for lenticular cross-section (see my notebook)
ratio = b_rad/a_rad
psiq = (1.0/(48.0*np.pi**2))*(1.0 - (nu**2/beta))*ratio**2/(5.0 + 2.0*((1.0/rho) - (nu/beta))*ratio**2 + (5.0/beta)*ratio**4)
R0 = 2.0*np.sqrt(psiq*Ai(0,0)[0]/Di(0,0)[0])*np.pi*a_rad*b_rad
# - critical curvature
c_cr = h_cr/R0

# Solution
loadings = np.linspace(0.0, 1.2*c_cr, 40)
# plt.ion()
# plt.grid()
# plt.show()

kx = []
ky = []
kxy = []
ls_load = []
ls_bend_el = [] # elastic contribution to the bending energy
ls_bend_cdg = [] # cdg contribution to the bending energy
ls_bend = [] # bending energy
ls_mem = [] # membrane energy

for count,i in enumerate(loadings):
    print '------------------------------------------------'
    print 'Now solving load increment ' + repr(count) + ' of ' + repr(len(loadings))
    # Load increment
    k_T.c = i
    solver.solve()
    v_h, w_h = u_.split(deepcopy=True)
    plot(w_h, key='u', wireframe = False)
    u_.assign(u_)
    # Post-processing
    # - list load
    ls_load.append(i*R0)
    # - assemble and list curvatures
    K_h = project(grad(grad(w_h)), TensorFunctionSpace(mesh, 'DG', 0))
    Kxy = assemble(K_h[0,1]*dx)/domain_area
    Kxx = 2.0*w_h((1.0 - 1E-6)*a_rad, 0.0)/(a_rad*a_rad) 
    Kyy = 2.0*w_h(0.0, (1.0 - 1E-6)*b_rad)/(b_rad*b_rad)     
    kx.append(Kxx*R0/np.sqrt(beta))
    ky.append(Kyy*R0)
    kxy.append(Kxy*R0/(beta**(1.0/4.0)))
    # - assemble and list energies
    bend_el = assemble(action(Pi_b, u_))
    bend_cdg = assemble(action(Pi_cdg, u_))
    bend = bend_el + bend_cdg
    mem = assemble(action(Pi_m, u_))
    en_scale = (4.0*R0**2)/(beta*np.pi*a_rad*b_rad*Di(0,0)[0])
    ls_bend_el.append(en_scale*bend_el)
    ls_bend_cdg.append(en_scale*bend_cdg)
    ls_bend.append(en_scale*bend)
    ls_mem.append(en_scale*mem)

    # Plotting
    # plt.plot(ls_load, kx, color = '0.0', linewidth=3.5, label=r"$K_x$")
    # plt.plot(ls_load, ky, color = '0.0', linewidth=3.5, label=r"$K_y$")
    # plt.plot(ls_load, kxy, color = '0.5', linewidth=3.5, label=r"$K_{xy}$")
    # plt.draw()

#plt.savefig("output/laminateplate.png")

# alpha value
alpha_value = project(alpha, mesh=mesh)
print 'The weight parameter is: %r' % alpha_value(0.0, 0.0)

# Analytical solution
h_before = np.linspace(0.0, 0.9999*h_cr, 100)
h_after= np.linspace(1.00001*h_cr, 1.2*h_cr, 100)
lsh = [i for i in h_before] + [j for j in h_after]
ls_f = [(1.0 + mu)*(np.sqrt(3.0)*np.sqrt(27*j**2 + 4.0*(1.0 + mu)) + 9.0*j) for j in h_before]
ls_K1 = [(2.0**(1.0/3.0)*i**(2.0/3.0) - 2.0*3.0**(1.0/3.0)*(1.0 + mu))/(6.0**(2.0/3.0)*i**(1.0/3.0)) for i in ls_f]
ls_K2 = [j/2.0*(1.0 + mu) + np.sqrt(j*j/4.0*(1.0 + mu)*(1.0 + mu) - (1.0 - mu)) for j in h_after]
ls_K3 = [j/2.0*(1.0 + mu) - np.sqrt(j*j/4.0*(1.0 + mu)*(1.0 + mu) - (1.0 - mu)) for j in h_after]
lskx = ls_K1 + ls_K2
lsky = ls_K1 + ls_K3
cur = np.array([[i, 0., 0.] for i in lskx]) + np.array([[0., j, 0.] for j in lsky]) - np.array([[k, k, 0.0] for k in lsh])
bend_an = [0.5*np.einsum('ij,j,i -> ', Dstar, j, j) for j in cur]
mem_an = [0.5*(i*j)**2 for (i,j) in zip(lskx, lsky)]

class AnalyticSolution(Expression):
    def __init__(self, kx, ky):
        self.kx = kx
        self.ky = ky

    def eval(self, value, x):
        value[0] = (1.0/2.0)*self.kx*x[0]**2 + (1.0/2.0)*self.ky*x[1]**2

# last solution error
w_an = AnalyticSolution(lskx[-1]/R0, lsky[-1]/R0)
error_w = errornorm(w_an, w_h, norm_type='l2')/norm(w_h, norm_type='l2')
print 'The L2 error in w is: %r' % error_w

# Figure 1: numerical vs analytical bending energy
plt.figure(1)
# plt.ion()
plt.plot(lsh, bend_an, color = 'y', linewidth=3.5, label=r"$B$")
plt.plot(ls_load, ls_bend, color = '0.0', linewidth=3.5, label=r"$B_h$")
plt.xlabel(r"$H_x = H_y = h$", fontsize=20)
plt.ylabel(r"$\bar{B}$", fontsize=20)
plt.xlim(lsh[0], lsh[-1])
plt.grid()
plt.legend(loc=2)
plt.show()

# Figure 2: bending energy with elastic and discontinuous contribution
plt.figure(2)
plt.plot(lsh, bend_an, color = 'y', linewidth=3.5, label=r"$B$")
plt.plot(ls_load, ls_bend, color = '0.0', linewidth=3.5, label=r"$B_h$")
plt.plot(ls_load, ls_bend_el, color = '0.25', linewidth=3.5, label=r"$B_h^e$")
plt.plot(ls_load, ls_bend_cdg, color = '0.5', linewidth=3.5, label=r"$B_h^d$")
plt.xlabel(r"$H_x = H_y = h$", fontsize=20)
plt.ylabel(r"$\bar{B}$", fontsize=20)
plt.xlim(lsh[0], lsh[-1])
plt.grid()
plt.legend(loc=2)
plt.show()

# Figure 3: numerical vs analytical stretching energy
plt.figure(3)
plt.plot(lsh, mem_an, color = 'y', linewidth=3.5, label=r"$M$")
plt.plot(ls_load, ls_mem, color = '0.0', linewidth=3.5, label=r"$M_h$")
plt.xlabel(r"$H_x = H_y = h$", fontsize=20)
plt.ylabel(r"$\bar{M}$", fontsize=20)
plt.xlim(lsh[0], lsh[-1])
plt.grid()
plt.legend(loc=2)
plt.show()

# Figure 4: numerical vs analytical curvatures
plt.figure(4)
plt.plot(ls_load, kx, color = '0.0', linewidth=3.5, label=r"$K_x$")
plt.plot(ls_load, ky, color = '0.25', linewidth=3.5, label=r"$K_y$")
plt.plot(ls_load, kxy, color = '0.5', linewidth=3.5, label=r"$K_{xy}$")
plt.plot(h_before, ls_K1, color = 'y', linewidth=3.5)
plt.plot(h_after, ls_K2, color = 'y', linewidth=3.5)
plt.plot(h_after, ls_K3, color = 'y', linewidth=3.5)
plt.xlabel(r"$H_x = H_y = h$", fontsize=20)
plt.ylabel(r"$K_x, K_y, K_{xy}$", fontsize=20)
plt.xlim(h_before[0], h_after[-1])
plt.grid()
plt.legend(loc=2)
plt.show()