import os, sys
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
# parameters.plotting_backend = "matplotlib"
from mpl_toolkits.mplot3d import Axes3D
parameters.form_compiler.quadrature_degree = 4
output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import mshr as mh

# Data
R_e = 10
R_i = 6
Y, nu = Constant(21.E6), Constant(0.0) 
mu = Y/(2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu) 
thickness = 0.03
eps = Constant(0.5*thickness) 

# Define the mesh
cut = 0.001*R_e
C = Point(0.0,0.0)
P2, P3 = Point(R_e*np.sin(cut/2.0), R_e*np.cos(cut/2.0)), Point(-R_e*np.sin(cut/2.0), R_e*np.cos(cut/2.0))
geom = mh.Circle(C, R_e, 100)-mh.Circle(C, R_i, 100)-mh.Polygon([C,P2,P3])
mesh = mh.generate_mesh(geom, 28)
h_max = mesh.hmax()
h_min = mesh.hmin()

# Spaces
P2 = FiniteElement("P", triangle, degree = 2)
bubble = FiniteElement("B", triangle, degree = 3)

Z = FunctionSpace(mesh, MixedElement(3*[P2 + bubble ] + 2*[P2]))
z_ = Function(Z)
z, zt = TrialFunction(Z), TestFunction(Z)

u0_, u1_, u2_, beta0_, beta1_ = split(z_)
u0t, u1t, u2t, beta0t, beta1t = split(zt)
u0, u1, u2, beta0, beta1 = split(z)

u_, u, ut = as_vector([u0_, u1_, u2_]), as_vector([u0, u1, u2]), as_vector([u0t, u1t, u2t])
beta_, beta, betat = as_vector([beta0_, beta1_]), as_vector([beta0, beta1]), as_vector([beta0t, beta1t])


# Kinematics
def director(beta):
    return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])

F = grad(u_) + as_tensor([[1.0, 0.0],[0.0, 1.0],[Constant(0), Constant(0)]])
d = director(beta_)

e_elastic = lambda F: 0.5*(F.T*F - Identity(2))
k_elastic = lambda F, d: -0.5*(F.T*grad(d) + grad(d).T*F)
g_elastic = lambda F, d: F.T*d

# Constitutive properties and the Constitutive law
aI_contra = Identity(2)
i, j, k, l = Index(), Index(), Index(), Index()
A_hooke = as_tensor((((2.0*lmbda*mu)/(lmbda + 2.0*mu))*aI_contra[i,j]*aI_contra[k,l]
                + 1.0*mu*(aI_contra[i,k]*aI_contra[j,l] + aI_contra[i,l]*aI_contra[j,k]))
                ,[i,j,k,l])

# The normal stress (``N``), bending moment (``M``),
# and shear stress (``T``) tensors are (they are purely Lagrangian stress measures,
# similar to the so called 2nd Piola stress tensor in 3D elasticity) ::
t = thickness
N = as_tensor(t*A_hooke[i,j,k,l]*e_elastic(F)[k,l],[i, j])
M = as_tensor((t**3/12.0)*A_hooke[i,j,k,l]*k_elastic(F,d)[k,l],[i, j])
T = as_tensor(t*mu*aI_contra[i,j]*g_elastic(F,d)[j], [i])

# Hence, the contributions to the elastic energy density due to membrane (``psi_m``),
# bending (``psi_b``), and shear (``psi_s``) are
# (they are per unit surface in the initial configuration) ::

psi_m = 0.5*inner(N, e_elastic(F))
psi_b = 0.5*inner(M, k_elastic(F,d))
psi_s = 0.5*inner(T, g_elastic(F,d))

# Shear and membrane locking is treated using the partial reduced
# selective integration proposed in Arnold and Brezzi [1]. In this approach
# shear and membrane energy are splitted as a sum of two contributions
# weighted by a factor ``alpha``. One of the two contributions is
# integrated with a reduced integration. While [1] suggests a 1-point
# reduced integration, we observed that this leads to spurious modes in
# the present case. We use then :math:`2\times 2`-points Gauss integration
# for a portion :math:`1-\alpha` of the energy, whilst the rest is
# integrated with a :math:`4\times 4` scheme. We further refine the
# approach of [1] by adopting an optimized weighting factor
# :math:`\alpha=(t/h)^2`, where :math:`h` is the mesh size. ::

h = CellSize(mesh)
alpha = project(t**2/h**2,FunctionSpace(mesh,'DG',0))
dx_h = dx(metadata={'quadrature_degree': 1}) # reduced integration
shear_energy = alpha*psi_s*dx + (1. - alpha)*psi_s*dx_h
membrane_energy = alpha*psi_m*dx + (1. - alpha)*psi_m*dx_h
bending_energy =  psi_b*dx


# Define subdomain for boundary condition on tractions
class FreeB(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] + R_e*np.sin(cut/2.0)) >= DOLFIN_EPS and (x[0] + R_i*np.sin(cut/2.0)) <= DOLFIN_EPS and (x[1] - R_e*np.cos(cut/2.0)) <= DOLFIN_EPS and (x[1] - R_i*np.cos(cut/2.0)) >= DOLFIN_EPS and on_boundary   

tractions = FreeB()
# Create mesh function over cell facets
exterior_facet_domains = FacetFunction("size_t", mesh)
exterior_facet_domains.set_all(0)
tractions.mark(exterior_facet_domains, 1)

# Define the measure
ds = Measure("ds")[exterior_facet_domains]

# Define the traction 
t_right = Expression(('c'), c = 1.0, degree=0)

# Define external work
external_work = t_right*u2_*ds(1)

# Hence the total elastic energy and its first and second derivatives are ::

Pi_total = bending_energy + membrane_energy + shear_energy - external_work
residual = derivative(Pi_total, z_, zt)
hessian = derivative(residual, z_, z)

# The boundary conditions prescribe a full clamping on the top boundary,
# while on the left and the right side the normal component of the
# rotation and the transverse displacement are blocked. ::

# bc_clamped = DirichletBC(Z, project(z_, Z), up_boundary)
# bc_u = DirichletBC(Z.sub(2), project(Constant(0.), Z.sub(2).collapse()), leftright_boundary)
# bc_beta = DirichletBC(Z.sub(4), project(z_[4], Z.sub(4).collapse()), leftright_boundary)
# bcs = [bc_clamped, bc_u, bc_beta]


# Define subdomains for Dirichlet boundary conditions
bottom = lambda x, on_boundary: (x[0] - R_e*np.sin(cut/2.0)) <= DOLFIN_EPS  and (x[0] - R_i*np.sin(cut/2.0)) >= DOLFIN_EPS and (x[1] - R_e*np.cos(cut/2.0)) <= DOLFIN_EPS  and (x[1] - R_i*np.cos(cut/2.0)) >= DOLFIN_EPS  and on_boundary
# bc_v = DirichletBC(Z.sub(0), project(Constant((0.0,0.0,0.0)), Z.sub(0).collapse()), bottom)
# bc_a = DirichletBC(Z.sub(1), project(Constant((0.0,0.0)), Z.sub(1).collapse()), bottom)
bcs = [DirichletBC(Z, project(Constant((0.0,0.0,0.0,0.0,0.0)), Z), bottom)]

# We use a standard Newton solver and setup the files for the writing the
# results to disk ::

# Initial guess
init = Function(Z)
z_.assign(init)

class MyNonlinearProblem(NonlinearProblem):

    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def F(self, b, x):
        assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A, x)

problem = MyNonlinearProblem(residual, hessian, bcs)
solver = NewtonSolver()

# Solver parameters
prm = solver.parameters
prm['error_on_nonconvergence'] = True
prm['maximum_iterations'] = 30
prm['linear_solver'] = "mumps"
prm['absolute_tolerance'] = 1E-7

output_dir = "output/naghdi-mitc-annular/"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Solution
import numpy as np
w_ls = []
v_ls = []
Tmax = 0.8
loadings = np.linspace(0.0, 0.5*Tmax, 40)

wlistA = []
wlistB = []
# Inner and outer point of the free edge
eval_pointA = (-1.01*R_i*np.sin(cut/2.),1.005*R_i*np.cos(cut/2.))
eval_pointB = (-1.0005*R_e*np.sin(cut/2.),0.999*R_e*np.cos(cut/2.))

for j in loadings:
    t_right.c = j
    solver.solve(problem, z_.vector())
    u0_h, u1_h, u2_h, b0_h, b1_h = z_.split(deepcopy=True)
    wlistA.append(u2_h(eval_pointA))
    wlistB.append(u2_h(eval_pointB))

    # v_h, theta_h, w_h, Rgamma_h, p_h = u_.split(deepcopy=True) # extract components
    # z_h = project(z_, VectorFunctionSpace(mesh, "CG", 1, dim=3))
    # z_h.rename('z', 'z')
    # fid << z_h, j
    # wlistA.append(-w_h(eval_pointA))
    # wlistB.append(-w_h(eval_pointB))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5.0, 5.0/1.648))
reference_Sze_inner = np.array([
    np.array([0., 1.305, 2.455, 3.435, 4.277, 5.007, 5.649, 6.725, 7.602, 8.34, 8.974, 
        9.529, 10.023, 10.468, 10.876, 11.257, 11.620, 11.97, 12.31, 12.642, 12.966, 13.282, 13.59, 13.891]),
    0.8*np.array([0., .025, .05,.075, .1, .125, .15, .2, .25, .3, .35, .4, .45, .5, .55, 
        .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    ])
reference_Sze_outer = np.array([
    np.array([0., 1.789, 3.370, 4.720, 5.876, 6.872, 7.736, 9.160, 10.288, 11.213, 11.992, 12.661,
        13.247, 13.768, 14.240, 14.674, 15.081, 15.469, 15.842, 16.202, 16.550, 16.886, 17.212, 17.528]),
    0.8*np.array([0., .025, .05,.075, .1, .125, .15, .2, .25, .3, .35, .4, .45, .5, .55, 
        .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    ])
# plt.plot(wlistA, loadings,  "x", color='green', label='FEniCS-shells inner point')
# plt.plot(wlistB, loadings,  "o", color='r', label='FEniCS-shells outer point')
plt.plot(wlistA, loadings,  "x", color='green', label=r'$w_h(A)$')
plt.plot(wlistB, loadings,  "o", color='r', label=r'$w_h(B)$')
plt.plot(*reference_Sze_inner, "-", color='b', label='Sze (Abaqus S4R)')
plt.plot(*reference_Sze_outer, "-", color='b')
plt.xlabel("Vertical deflections")
plt.ylabel("Force/Unit Length")
plt.legend()
plt.tight_layout()
plt.savefig("annular_disp.pdf")

disp = as_vector([u0_h, u1_h, u2_h])
disp_pro = project(disp, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
file = File("annular-semi.pvd")
file << disp_pro