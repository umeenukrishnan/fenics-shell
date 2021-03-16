import os, sys
from dolfin import *
from mshr import *
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

parameters.form_compiler.quadrature_degree = 4
output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def hypar_solver(divisions, integration_points, alpha_exp):

	# Data
	L = 1.0
	Y, nu = 2e8, 0.3
	mu = Y/(2.0*(1.0 + nu))
	lb = 2.0*mu*nu/(1.0 - 2.0*nu)
	t = Constant(1E-4)

	# Mesh
	P1, P2 = Point(-L/2, -L/2), Point(L/2, L/2) 
	mesh = RectangleMesh(P1, P2, divisions, divisions)
	# mesh = RectangleMesh(P1, P2, divisions, divisions, "crossed")
	# mesh = generate_mesh(Rectangle(P1, P2), divisions)
	# With a crossed mesh it seems to work the 1 point integration rule too.

	# Initial shape
	initial_shape = Expression(('x[0]','x[1]','x[0]*x[0] - x[1]*x[1]'), degree = 4)
	V_y = FunctionSpace(mesh, VectorElement("P", triangle, degree=2, dim=3))
	yI = project(initial_shape, V_y)

	# First form and bases
	aI = grad(yI).T*grad(yI)
	aI_contra, jI = inv(aI), det(aI)
	g0, g1 = yI.dx(0), yI.dx(1)
	g0_c, g1_c = aI_contra[0,0]*g0 + aI_contra[0,1]*g1, aI_contra[1,0]*g0 + aI_contra[1,1]*g1

	# Normal
	def normal(y):
	    n = cross(y.dx(0), y.dx(1))
	    return n/sqrt(inner(n,n))

	V_normal = FunctionSpace(mesh, VectorElement("P", triangle, degree = 1, dim = 3))
	nI = project(normal(yI), V_normal)

	# Spaces
	P2 = FiniteElement("P", triangle, degree = 2)
	bubble = FiniteElement("B", triangle, degree = 3)

	Z = FunctionSpace(mesh, MixedElement(3*[P2 + bubble ] + 2*[P2 + bubble]))
	z_ = Function(Z)
	z, zt = TrialFunction(Z), TestFunction(Z)

	u0_, u1_, u2_, th0_, th1_ = split(z_)
	u0t, u1t, u2t, th0t, th1t = split(zt)
	u0, u1, u2, th0, th1 = split(z)

	u_, u, ut = as_vector([u0_, u1_, u2_]), as_vector([u0, u1, u2]), as_vector([u0t, u1t, u2t])
	theta_, theta, thetat = th0_*g0_c + th1_*g1_c, th0*g0_c + th1*g1_c, th0t*g0_c + th1t*g1_c

	# Strain measures
	e_naghdi = lambda v: 0.5*(grad(yI).T*grad(v) + grad(v).T*grad(yI))
	k_naghdi = lambda v, t: -0.5*(grad(yI).T*grad(t) + grad(t).T*grad(yI)) - 0.5*(grad(nI).T*grad(v) + grad(v).T*grad(nI))
	g_naghdi = lambda v, t: grad(yI).T*t + grad(v).T*nI 

	# Constitutive law
	i, j, k, l = Index(), Index(), Index(), Index()
	A_hooke = as_tensor((((2.0*lb*mu)/(lb + 2.0*mu))*aI_contra[i,j]*aI_contra[k,l] + 1.0*mu*(aI_contra[i,k]*aI_contra[j,l] + aI_contra[i,l]*aI_contra[j,k])),[i,j,k,l])

	# Stress measures
	N = as_tensor((t*A_hooke[i,j,k,l]*e_naghdi(u_)[k,l]),[i, j])
	M = as_tensor(((t**3/12.0)*A_hooke[i,j,k,l]*k_naghdi(u_, theta_)[k,l]),[i, j])
	T = as_tensor((t*mu*aI_contra[i,j]*g_naghdi(u_, theta_)[j]), [i])

	# Energy densities
	psi_m = .5*inner(N, e_naghdi(u_)) 
	psi_b = .5*inner(M, k_naghdi(u_, theta_)) 
	psi_s = .5*inner(T, g_naghdi(u_, theta_))

	# PSRI
	dx_h = dx(metadata={'quadrature_degree': integration_points})
	h = CellDiameter(mesh)
	alpha = project((t/h)**alpha_exp, FunctionSpace(mesh,'DG',0))
	# alpha = project((t/L)**alpha_exp, FunctionSpace(mesh,'DG',0))
	
	#########################################################################
	# t/L is suggested by A&B and it works fine in the linear case. 
	# t/h seems to work better in the nonlinear case.
	#########################################################################

	kappa = 1.0
	shear_energy = alpha*psi_s*sqrt(jI)*dx + (kappa - alpha)*psi_s*sqrt(jI)*dx_h
	membrane_energy = alpha*psi_m*sqrt(jI)*dx + (kappa - alpha)*psi_m*sqrt(jI)*dx_h
	bending_energy =  psi_b*sqrt(jI)*dx

	# Totale elastic energy
	elastic_energy = bending_energy + membrane_energy + shear_energy

	# External work
	body_force = 8.*t
	f = Constant(body_force)
	external_work = f*u2_*sqrt(jI)*dx

	# Problem statement
	Pi_total = elastic_energy - external_work
	residual = derivative(Pi_total, z_, zt)
	hessian = derivative(residual, z_, z)

	# Boundary conditions
	left_boundary = lambda x, on_boundary: abs(x[0] + L/2) <= DOLFIN_EPS and on_boundary
	clamp = DirichletBC(Z, project(Expression(("0.0", "0.0", "0.0", "0.0", "0.0"), degree = 1), Z), left_boundary)
	bcs = [clamp]

	# Solver and solution
	output_dir = "output/"
	A, b = assemble_system(hessian, residual, bcs=bcs)
	solver = LUSolver("mumps")
	solver.solve(A, z_.vector(), b)
	u0_h, u1_h, u2_h, th0_h, th1_h = z_.split(deepcopy=True)

	return [Z.dim(), abs(u2_h(L/2,0.)), alpha(0.,0.)]

ndivs = [16, 32, 64, 128]

dofs = []
disp_0, al_0 = [], []
disp_2, al_2 = [], []
disp_4, al_4 = [], []

print("Solving for n = 0")
for count, i in enumerate(ndivs):
	degrees_of_freedom, tip_displacement, alpha = hypar_solver(i, 2, 0)
	dofs.append(degrees_of_freedom)
	disp_0.append(tip_displacement)
	al_0.append(alpha)

print("Solving for n = 2")
for count, i in enumerate(ndivs):
	degrees_of_freedom, tip_displacement, alpha = hypar_solver(i, 2, 2)
	disp_2.append(tip_displacement)
	al_2.append(alpha)

print("Solving for n = 4")
for count, i in enumerate(ndivs):
	degrees_of_freedom, tip_displacement, alpha = hypar_solver(i, 2, 4)
	disp_4.append(tip_displacement)
	al_4.append(alpha)

err0 = [1 - j/disp_0[-1] for j in disp_0]
err2 = [abs(1 - j/disp_2[-1]) for j in disp_2]
err4 = [1 - j/disp_4[-1] for j in disp_4]

fig = plt.figure(figsize=(5.0, 5.0/1.648))
ax = plt.gca()
ax.loglog(dofs[:-1], err0[:-1], '-ob', label=r"$\alpha = 1$")
ax.loglog(dofs[:-1], err2[:-1], '-or', label=r"$\alpha = (t/h)^2$")
ax.loglog(dofs[:-1], err4[:-1], '-og', label=r"$\alpha = (t/h)^4$")
ax.set_xlabel('Number of degrees of freedom')
ax.set_ylabel('Relative Error $1-E_h/E$')
ax.legend()
plt.tight_layout()
plt.savefig("hypar_convergence-alpha.pdf")

# Fixed mesh
ns = [0,2,4,8]
dofs = []
disp = []
al = []

# # reference_solution = hypar_solver(128, 0)[1]
for count, i in enumerate(ns):
	degrees_of_freedom, tip_displacement, alpha = hypar_solver(64, 2, i)
	dofs.append(degrees_of_freedom)
	disp.append(tip_displacement)
	al.append(alpha)

bathe_value = 0.53 # t = 1E-4!
err0 = [1 - j/bathe_value for j in disp]

fig = plt.figure(figsize=(5.0, 5.0/1.648))
ax = plt.gca()
ax.loglog(al, err0, '-ob', label=r"$n=64, t=10^{-4}$")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'Relative Error $1-E_h/E_{Bathe}$')
ax.legend()
plt.tight_layout()
plt.savefig("hypar_convergence-alpha-fixedmesh.pdf")
