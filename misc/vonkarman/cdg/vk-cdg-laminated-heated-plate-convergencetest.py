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
# user.add("nelements", 80)
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

def vonkarman_laminates(mesh_div, alpha_coef):

# Elliptical or circular plate
	centre = Point(0.,0.)
	a_rad = 1.0
	b_rad = 0.5
	domain_area = np.pi*a_rad*b_rad
	geom = mshr.Ellipse(centre, a_rad, b_rad)
	mesh = mshr.generate_mesh(geom, mesh_div)
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
		return x[0]**2 + x[1]**2 < (0.0000005*h_max)**2

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
	alpha = Constant(alpha_coef*np.linalg.norm(Di(0,0)))
	Pi_cdg = cdg_energy(theta, M_, alpha, mesh)

# Total energy
	Pi = Pi_elastic + Pi_cdg

# Residuals and Jacobian
	Pi = action(Pi, u_)
	F = derivative(Pi, u_, u_t)
	J = derivative(F, u_, u)

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
# - analytical solution (on post-critical branch, see A. Fernandes et al., 2010)
	h_a = 1.2*h_cr
	K2 = h_a/2.0*(1.0 + mu) + np.sqrt(h_a*h_a/4.0*(1.0 + mu)*(1.0 + mu) - (1.0 - mu))
	K3 = h_a/2.0*(1.0 + mu) - np.sqrt(h_a*h_a/4.0*(1.0 + mu)*(1.0 + mu) - (1.0 - mu))
	Kvec = np.array([K2 - h_a, K3 - h_a, 0.])
# - analytical energies
	Be_an = 0.5*np.einsum('ij,j,i -> ', Dstar, Kvec, Kvec)
	Me_an = 0.5*(K2*K3)**2
	En_an = Be_an + Me_an

# Initial guess (we use as initial guess the analytical solution)
	init = Function(U)
	init = interpolate(Expression(('0.0', '0.0', '0.5*kx*x[0]*x[0] + 0.5*ky*x[1]*x[1] + kxy*x[0]*x[1]'), kx=K2/R0, ky=K3/R0, kxy=0.0), U)
	u_.assign(init)

# Solver settings
	problem = NonlinearVariationalProblem(F, u_, bcs, J = J)
	solver = NonlinearVariationalSolver(problem)
	solver.parameters.nonlinear_solver = 'snes'
	solver.parameters.snes_solver.linear_solver =  'umfpack'
	# solver.parameters.snes_solver.linear_solver =  'superlu_dist'
	# solver.parameters.snes_solver.line_search =  'bt'
	solver.parameters.snes_solver.maximum_iterations =  15
	solver.parameters.snes_solver.error_on_nonconvergence = False
	# solver.parameters.snes_solver.absolute_tolerance = 1E-8

# Solution
	k_T.c = h_a/R0
	solver.solve()
	v_h, w_h = u_.split(deepcopy=True)
# plot(w_h, key='u', wireframe = False)

	class KarmanSolution(Expression):
		def __init__(self, kx, ky):
			self.kx = kx
			self.ky = ky

		def eval(self, value, x):
			value[0] = (1.0/2.0)*self.kx*x[0]**2 + (1.0/2.0)*self.ky*x[1]**2

	w_an = KarmanSolution(K2/R0, K3/R0)

# Post-processing
# - energy scale
	en_scale = (4.0*R0**2)/(beta*np.pi*a_rad*b_rad*Di(0,0)[0])
# - numerical bending energy
	h = CellSize(mesh)
	h_avg = (h('+') + h('-'))/2.0
	n = FacetNormal(mesh)
	edge_weight = assemble(Constant(1.0)('+')/h_avg*dS)
	Pi_j = -inner(jump(theta, n), avg(inner(M_, outer(n, n))))*dS # jump term
	Pi_p = (1.0/2.0)*alpha('+')/h_avg*inner(jump(theta, n), jump(theta, n))*dS # pen. term
	Be_j = assemble(action(Pi_j, u_))
	Be_p = assemble(action(Pi_p, u_))
	Be_el = assemble(action(Pi_b, u_))
	Be_cdg = assemble(action(Pi_cdg, u_)) # this should be Be_j + Be_p
	Be_h = en_scale*(Be_el + Be_cdg)
# - numerical membrane energy
	Me_el = assemble(action(Pi_m, u_))
	Me_h = en_scale*Me_el
# - numerical coupling energy
	BMe_el = assemble(action(Pi_mb, u_))
	BMe_h = en_scale*BMe_el
# - numerical potential energy
	En_h = Be_h + Me_h + BMe_h

# Comparisons
	# error_Be = 100*(Be_an - Be_h)/Be_an
	# error_Me = 100*(Me_an - Me_h)/Me_an
	init_u, init_w = init.split(deepcopy=True)
	# error_w = errornorm(init_w, w_h, norm_type='l2')/norm(w_h, norm_type='l2')
	# print '------------------------------------------------------'
	# print 'Percentage error on bending energy: %r' % error_Be
	# print 'Percentage error on stretching energy: %r' % error_Me
	# print 'L2-error norm in w: %r' % error_w
	# print '------------------------------------------------------'

	result = {}
	result['hmax'] = h_max
	result['hmin'] = h_min
	result['w_l2'] = errornorm(w_an, w_h, norm_type='l2') #/norm(w_h, norm_type='l2')
	result['w_h1'] = errornorm(w_an, w_h, norm_type='h1')
	result['bending_error'] = 100*(Be_an - Be_h)/Be_an
	result['membrane_error'] = 100*(Me_an - Me_h)/Me_an
	result['energy_error'] = 100*(En_an - En_h)/En_an
	result['dofs'] = U.dim()
	result['transverse_disp'] = w_h
	result['transverse_disp_analytical'] = w_an
	result['Benergy_elastic'] = en_scale*Be_el
	result['Benergy_cdg'] = en_scale*Be_cdg
	result['Benergy_cdg_jump'] = en_scale*Be_j
	result['Benergy_cdg_penal'] = en_scale*Be_p
	result['Benergy'] = Be_h
	result['Menergy'] = Me_h
	result['BMenergy'] = BMe_h
	result['Benergy_analytical'] = Be_an
	result['Menergy_analytical'] = Me_an
	result['edge_weight'] = edge_weight

	return result

# vk_test = vonkarman_laminates(80, 1.0)

ls_al = [1.0, 10., 100., 1000.]
# ls_al = np.linspace(1., 500, 4)
# ls_div = [10, 20, 50, 80, 100]
ls_div = [10, 20, 40, 60, 80, 100]
ls_h = []
ls_en = []
ls_w_l2 = []
ls_w_h1 = []
ls_Bel = []
ls_Bj = []
ls_Bp = []
ls_B = []
ls_M = []
ls_Ban = []
ls_Man = []
ls_N = []
ls_ew = []

# ndiv = 40
# for i in ls_al:
# 	vk_test = vonkarman_laminates(ndiv, i)
# 	ls_N.append(1.0/vk_test['hmax'])
# 	ls_en.append(np.abs(vk_test['energy_error']))
# 	ls_Bj.append(vk_test['Benergy_cdg_jump'])
# 	ls_Bp.append(vk_test['Benergy_cdg_penal'])
# 	ls_Bel.append(vk_test['Benergy_elastic'])
# 	ls_B.append(vk_test['Benergy'])
# 	ls_Ban.append(vk_test['Benergy_analytical'])
# 	ls_Man.append(vk_test['Menergy_analytical'])
# 	ls_M.append(vk_test['Menergy'])
# 	ls_w_l2.append(vk_test['w_l2'])
# 	ls_w_h1.append(vk_test['w_h1'])
# 	ls_ew.append(vk_test['edge_weight'])

# plt.figure(1)
# plt.loglog(ls_al, ls_w_l2, color = 'k', marker = 'o', linewidth=2.5, label=r"$|w - w_h|_{L_2}$")
# plt.loglog(ls_al, ls_w_h1, color = 'b', marker = 'o', linewidth=2.5, label=r"$|w - w_h|_{H_1}$")
# plt.xlabel(r"$\alpha/\alpha_D$", fontsize=20)
# plt.ylabel(r"$w_e$", fontsize=20)
# plt.xlim(ls_al[0], ls_al[-1])
# plt.grid()
# plt.legend(loc=2)
# plt.savefig('wnorms-1overh' + str(int(ls_N[0])) + '.pdf')
# plt.show()

# plt.figure(2)
# plt.plot(ls_al, ls_B, color = 'k', marker='o', linewidth=2.5, label=r"$B$")
# plt.plot(ls_al, ls_Bel, color = 'b', marker='o', linewidth=2.5, label=r"$b_e$")
# plt.plot(ls_al, ls_Ban, color = 'r', linewidth=2.5, label=r"$B_a$")
# plt.xlabel(r"$\alpha/\alpha_D$", fontsize=20)
# plt.ylabel(r"$B$", fontsize=20)
# plt.xlim(ls_al[0], ls_al[-1])
# plt.grid()
# plt.legend(loc=2)
# plt.savefig('bendingenergy-1overh' + str(int(ls_N[0])) + '.pdf')
# plt.show()

# plt.figure(3)
# plt.plot(ls_al, ls_Bj, color = 'g', marker='o', linewidth=2.5, label=r"$b_j$")
# plt.plot(ls_al, ls_Bp, color = 'y', marker='o', linewidth=2.5, label=r"$b_p$")
# plt.xlabel(r"$\alpha/\alpha_D$", fontsize=20)
# plt.ylabel(r"$B_{cdg}$", fontsize=20)
# plt.xlim(ls_al[0], ls_al[-1])
# plt.grid()
# plt.legend(loc=2)
# plt.savefig('cdgenergy-1overh' + str(int(ls_N[0])) + '.pdf')
# plt.show()

alpha_coeff = 1.0
for i in ls_div:
	vk_test = vonkarman_laminates(i, alpha_coeff)
	ls_N.append(1.0/vk_test['hmax'])
	ls_en.append(np.abs(vk_test['energy_error']))
	ls_Bj.append(vk_test['Benergy_cdg_jump'])
	ls_Bp.append(vk_test['Benergy_cdg_penal'])
	ls_Bel.append(vk_test['Benergy_elastic'])
	ls_B.append(vk_test['Benergy'])
	ls_Ban.append(vk_test['Benergy_analytical'])
	ls_Man.append(vk_test['Menergy_analytical'])
	ls_M.append(vk_test['Menergy'])
	ls_w_l2.append(vk_test['w_l2'])
	ls_w_h1.append(vk_test['w_h1'])
	ls_ew.append(vk_test['edge_weight'])

plt.figure(1)
plt.loglog(ls_N, ls_w_l2, color = 'k', marker = 'o', linewidth=2.5, label=r"$|w - w_h|_{L_2}$")
plt.loglog(ls_N, ls_w_h1, color = 'b', marker = 'o', linewidth=2.5, label=r"$|w - w_h|_{H_1}$")
plt.xlabel(r"$1/h$", fontsize=20)
plt.ylabel(r"$w_e$", fontsize=20)
plt.xlim(ls_N[0], ls_N[-1])
plt.grid()
plt.legend(loc=2)
plt.savefig('wnorms-alpha' + str(alpha_coeff).replace('.', '_') + '.pdf')
plt.show()

plt.figure(2)
plt.plot(ls_N, ls_B, color = 'k', marker='o', linewidth=2.5, label=r"$B$")
plt.plot(ls_N, ls_Bel, color = 'b', marker='o', linewidth=2.5, label=r"$b_e$")
plt.plot(ls_N, ls_Ban, color = 'r', linewidth=2.5, label=r"$B_a$")
plt.xlabel(r"$1/h$", fontsize=20)
plt.ylabel(r"$B$", fontsize=20)
plt.xlim(ls_N[0], ls_N[-1])
plt.grid()
plt.legend(loc=2)
plt.savefig('bendingenergy-alpha' + str(alpha_coeff).replace('.', '_') + '.pdf')
plt.show()

plt.figure(3)
plt.plot(ls_N, ls_Bj, color = 'g', marker='o', linewidth=2.5, label=r"$b_j$")
plt.plot(ls_N, ls_Bp, color = 'y', marker='o', linewidth=2.5, label=r"$b_p$")
plt.xlabel(r"$1/h$", fontsize=20)
plt.ylabel(r"$B_{cdg}$", fontsize=20)
plt.xlim(ls_N[0], ls_N[-1])
plt.grid()
plt.legend(loc=2)
plt.savefig('cdgenergy-alpha' + str(alpha_coeff).replace('.', '_') + '.pdf')
plt.show()

plt.figure(4)
plt.plot(ls_N, ls_M, color = 'k', marker='o', linewidth=2.5, label=r"$M$")
plt.plot(ls_N, ls_Man, color = 'r', linewidth=2.5, label=r"$M_a$")
plt.xlabel(r"$1/h$", fontsize=20)
plt.ylabel(r"$M$", fontsize=20)
plt.xlim(ls_N[0], ls_N[-1])
plt.grid()
plt.legend(loc=2)
plt.savefig('membraneenergy-alpha' + str(alpha_coeff).replace('.', '_') + '.pdf')
plt.show()