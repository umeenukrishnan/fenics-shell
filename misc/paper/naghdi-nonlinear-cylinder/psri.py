# 
# ..    # vim: set fileencoding=utf8 :
# 
# .. _CylindricalPointForce:
# 
# Clamped semi-cylindrical shell under point load
# ===============================================
# 
# This demo is implemented in the single Python file
# :download:`demo_nonlinear-naghdi-cylindrical.py`.
# 
# This demo program solves the nonlinear Naghdi shell equations for a
# semi-cylindrical shell loaded by a point force. This problem is a standard
# reference for testing shell finite element formulations, see [2].
# The numerical locking issue is cured using enriched finite
# element including cubic bubble shape functions and Partial Selective
# Reduced Integration [1].
# 
# .. image:: configuration.png
# 
# To follow this demo you should know how to:
# 
# -  Define a MixedElement and EnrichedElement and a FunctionSpace from
#    it.
# -  Write variational forms using the Unified Form Language.
# -  Automatically derive Jacobian and residuals using derivative().
# -  Apply Dirichlet boundary conditions using DirichletBC and apply().
# -  Solve non-linear problems using NonlinearProblem.
# -  Output data to XDMF files with XDMFFile.
# 
# This demo then illustrates how to:
# 
# -  Define and solve a nonlinear Naghdi shell problem with a curved
#    stress-free configuration given as analytical expression in terms
#    of two curvilinear coordinates.
# 
# We start with importing the required modules, setting ``matplolib`` as
# plotting backend, and generically set the integration order to 4 to
# avoid the automatic setting of FEniCS which would lead to unreasonably
# high integration orders for complex forms. ::

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

# We consider a semi-cylindrical shell of radius ``rho``, axis length
# ``L``. The shell is made of a linear elastic isotropic homogeneous
# material with Young modulus ``Y`` and Poisson ratio ``nu``. The
# (uniform) shell thickness is denoted by ``t``.
# The Lamé moduli ``lmbda``, ``mu`` are introduced to write later
# the 2D constitutive equation in plane-stress::

rho = 1.016  # radius
L = 3.048
Y, nu = 2.0685E7, 0.3
mu = Y/(2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu)
t = Constant(0.03)

# The midplane of the initial (stress-free) configuration
# :math:`\mathcal{S}_I` of the shell is given in the form of an analytical
# expression
# 
# .. math:: y_I:x\in\mathcal{M}\subset R^2\to y_I(x)\in\mathcal{S}_I\subset \mathcal R^3
# 
# in terms of the curvilinear coordinates :math:`x`. In the specific case
# we adopt the cylindrical coordinates :math:`x_0` and :math:`x_1`
# representing the angular and axial coordinates, respectively.
# Hence we mesh the two-dimensional domain
# :math:`\mathcal{M}\equiv [0,L_y]\times [-\pi/2,\pi/2]`. ::

P1, P2 = Point(-np.pi/2., 0.), Point(np.pi/2., L)
ndiv = 21
mesh = generate_mesh(Rectangle(P1, P2), ndiv)
plot(mesh); plt.xlabel(r"$x_0$"); plt.ylabel(r"$x_1$")
plt.savefig("output/mesh.png")


# .. image:: mesh.png
# 
# We provide the analytical expression of the initial shape as an
# ``Expression`` that we represent on a suitable ``FunctionSpace`` (here
# :math:`P_2`, but other are choices are possible)::

initial_shape = Expression(('r*sin(x[0])','x[1]','r*cos(x[0])'), r=rho, degree = 4)
V_y =  FunctionSpace(mesh, VectorElement("P", triangle, degree = 2, dim = 3))
yI = project(initial_shape, V_y)

# Given the midplane, we define the corresponding unit normal as below and
# project on a suitable function space (here :math:`P_1` but other choices
# are possible)::

def normal(y):
    n = cross(y.dx(0), y.dx(1))
    return n/sqrt(inner(n,n))

V_normal = FunctionSpace(mesh, VectorElement("P", triangle, degree = 1, dim = 3))
nI = project(normal(yI), V_normal)

# The kinematics of the Nadghi shell model is defined by the following
# vector fields :
# 
# - ``y``: the position of the midplane
# - ``d``: the director, a unit vector giving the orientation of the microstructure
# 
# We parametrize the director field by two angles, which correspond to spherical coordinates,
# so as to explicitly resolve the unit norm constraint (see [3])::

def director(beta):
    return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])

# We assume that in the initial configuration the director coincides with
# the normal. Hence, we can define the angles ``beta`` for the initial
# configuration as follows: ::

betaI_expression = Expression(["atan2(-n[1], sqrt(pow(n[0],2) + pow(n[2],2)))",
                               "atan2(n[0],n[2])"], n = nI, degree=4)

V_beta = FunctionSpace(mesh, VectorElement("P", triangle, degree = 2, dim = 2))
betaI = project(betaI_expression, V_beta)

# The director in the initial configuration is then written as ::

dI = director(betaI)

# We can visualize the shell shape and its normal with this
# utility function ::

def plot_shell(y,n=None):
    y_0, y_1, y_2 = y.split(deepcopy=True)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(y_0.compute_vertex_values(),
                    y_1.compute_vertex_values(),
                    y_2.compute_vertex_values(),
                    triangles=y.function_space().mesh().cells(),
                    linewidth=1, antialiased=True, shade = False)
    if n:
        n_0, n_1, n_2 = n.split(deepcopy=True)
        ax.quiver(y_0.compute_vertex_values(),
              y_1.compute_vertex_values(),
              y_2.compute_vertex_values(),
              n_0.compute_vertex_values(),
              n_1.compute_vertex_values(),
              n_2.compute_vertex_values(),
              length = .2, color = "r")
    ax.view_init(elev=20, azim=80)
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")
    plt.xticks([-1,0,1])
    plt.yticks([0,pi/2])
    return ax

plot_shell(yI,project(dI,V_normal))
plt.savefig("output/initial_cofiguration.png")

# .. image:: initial_cofiguration.png
# 
# 
# In our 5-parameter Naghdi shell model the configuration of the shell is
# assigned by
# 
# - the 3-component vector field ``u_`` representing the displacement
#   with respect to the initial configuration ``yI``
# 
# - the 2-component vector field ``beta_`` representing the angle variation
#   of the director ``d`` with respect to the initial configuration
# 
# Following [1], we use a ``P2+bubble`` element for ``y_`` and a ``P2``
# element for ``beta_``, and collect them in the state vector
# ``z_=[u_,beta_]``. We further define ``Function``, ``TestFunction``, and
# ``TrialFucntion`` and their different split views, which are useful for
# expressing the variational formulation. ::

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


# The gradient of the transformation and the director in the current
# configuration are given by ::

F = grad(u_) + grad(yI)
d = director(beta_+betaI)

# With the following definition of the natural metric and curvature ::

aI = grad(yI).T*grad(yI)
bI = -0.5*(grad(yI).T*grad(dI) + grad(dI).T*grad(yI))

# The elastic extensional (``e_elastic``), bending (``k_elastic``) , and
# shear (``g_elastic``) deformations in the Naghdi model are defined by ::

e_elastic = lambda F: 0.5*(F.T*F - aI)
k_elastic = lambda F, d: -0.5*(F.T*grad(d) + grad(d).T*F) - bI
g_elastic = lambda F, d: F.T*d-grad(yI).T*dI

# Using curvilinear coordinates,  and denoting by ``aI_contra`` the
# contravariant components of the metric tensor (in the initial curved configuration)
# the constitutive equation is written in terms of the matrix ``A_hooke`` below,
# representing the contravariant components of the constitutive tensor
# for isotropic elasticity in plane stress (see *e.g.* [4]).
# We use the index notation offered by ``UFL`` to express
# operations between tensors ::

aI_contra = inv(aI)
jI = det(aI)
# Constitutive properties and the Constitutive law
i, j, k, l = Index(), Index(), Index(), Index()
A_hooke = as_tensor((((2.0*lmbda*mu)/(lmbda + 2.0*mu))*aI_contra[i,j]*aI_contra[k,l]
                + 1.0*mu*(aI_contra[i,k]*aI_contra[j,l] + aI_contra[i,l]*aI_contra[j,k]))
                ,[i,j,k,l])

# The normal stress (``N``), bending moment (``M``),
# and shear stress (``T``) tensors are (they are purely Lagrangian stress measures,
# similar to the so called 2nd Piola stress tensor in 3D elasticity) ::

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
dx_h = dx(metadata={'quadrature_degree': 2}) # reduced integration
shear_energy = alpha*psi_s*sqrt(jI)*dx + (1. - alpha)*psi_s*sqrt(jI)*dx_h
membrane_energy = alpha*psi_m*sqrt(jI)*dx + (1. - alpha)*psi_m*sqrt(jI)*dx_h
bending_energy =  psi_b*sqrt(jI)*dx

# Hence the total elastic energy and its first and second derivatives are ::

Pi_total = bending_energy + membrane_energy + shear_energy
residual = derivative(Pi_total, z_, zt)
hessian = derivative(residual, z_, z)

# The boundary conditions prescribe a full clamping on the top boundary,
# while on the left and the right side the normal component of the
# rotation and the transverse displacement are blocked. ::

up_boundary = lambda x, on_boundary: x[1] <= 1.e-4 and on_boundary
leftright_boundary = lambda x, on_boundary: near(abs(x[0]), pi/2., 1.e-6)  and on_boundary

bc_clamped = DirichletBC(Z, project(z_, Z), up_boundary)
bc_u = DirichletBC(Z.sub(2), project(Constant(0.), Z.sub(2).collapse()), leftright_boundary)
bc_beta = DirichletBC(Z.sub(4), project(z_[4], Z.sub(4).collapse()), leftright_boundary)
bcs = [bc_clamped, bc_u, bc_beta]

# The loading is exerted by a point force applied at the midpoint of the bottom boundary.
# This is implemented using the ``PointSource`` in FEniCS and
# defining a custom``NonlinearProblem`` ::

class NonlinearProblemPointSource(NonlinearProblem):

    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs
        self.P = 0.0

    def F(self, b, x):
        assemble(self.L, tensor=b)
        point_source = PointSource(self.bcs[0].function_space().sub(2), Point(0.0, L), self.P)
        point_source.apply(b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A, x)

problem = NonlinearProblemPointSource(residual, hessian, bcs)

# We use a standard Newton solver and setup the files for the writing the
# results to disk ::

solver = NewtonSolver()
output_dir = "output/"
file_z = File(output_dir+"configuration.pvd")
file_energy = File(output_dir+"energy.pvd")

# Finally, we can solve the quasi-static problem, incrementally increasing
# the loading from :math:`0` to :math:`2000`\ N. ::

P_values = np.linspace(0.0, 0.5*2000.0, 40)
displacement = 0.*P_values
z_.assign(project(Constant((0,0,0,0,0)),Z))
for (i,P) in enumerate(P_values):
    problem.P = P
    (niter,cond) = solver.solve(problem, z_.vector())
    z_sol = project(u_+yI,V_y)
    displacement[i]=z_sol(0.0,L)[2]-yI(0.0,L)[2]
    z_sol.rename("z", "z")
    file_z << (z_sol,P)
    print("Increment %d of %s. Converged in %2d iterations. P:  %.2f, Displ: %.2f" %(i, P_values.size,niter,P, displacement[i]))
    en_function = project(psi_m+psi_b+psi_s,FunctionSpace(mesh,'P',1))
    en_function.rename("Elastic Energy","Elastic Energy")
    file_energy << (en_function,P)

# We can plot the final configuration of the shell: ::

plot_shell(project(u_+yI,V_y))
plt.savefig("output/finalconfiguration.png")

# .. image:: finalconfiguration.png
# 
# The results  for the transverse displacement at the point of application of the force
# are verificated against a standard reference from the literature,
# obtained using Abaqus ``S4R`` element and a
# structured mesh of :math:`40\times 40` elements. ::

plt.figure()
reference_Sze = np.array([
    1.e-2*np.array([0., 5.421, 16.1, 22.195, 27.657, 32.7, 37.582, 42.633,
    48.537, 56.355, 66.410, 79.810, 94.669, 113.704, 124.751, 132.653,
    138.920, 144.185, 148.770, 152.863, 156.584, 160.015, 163.211,
    166.200, 168.973, 171.505]),
    2000.*np.array([0., .05, .1, .125, .15, .175, .2, .225, .25, .275, .3,
    .325, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    ])
plt.plot(-np.array(displacement), P_values, label='fenics-shell %s divisions (AB)'%ndiv)
plt.plot(*reference_Sze, "or", label='Sze (Abaqus S4R)')
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.legend()
plt.grid()
plt.savefig("output/comparisons.png")
plt.show()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5.0, 5.0/1.648))
ax = plt.gca()
ax.plot(-np.array(displacement), P_values, 'o', color='red', label=r'$w_h$')
ax.plot(*reference_Sze, "-", color='blue', label='Sze (Abaqus S4R)')
ax.set_xlabel("Displacement (mm)")
ax.set_ylabel("Load (N)")
ax.legend()
plt.tight_layout()
plt.savefig("output/cyl_comparisons.pdf")

# .. image:: comparisons.png
# 
# 
# References
# ----------
# 
# [1] D. Arnold and F.Brezzi, Mathematics of Computation, 66(217): 1-14, 1997. https://www.ima.umn.edu/~arnold//papers/shellelt.pdf
# 
# [2] K. Sze, X. Liu, and S. Lo. Popular benchmark problems for geometric
# nonlinear analysis of shells. Finite Elements in Analysis and Design,
# 40(11):1551 – 1569, 2004.
# 
# [3] P. Betsch, A. Menzel, and E. Stein. On the parametrization of finite
# rotations in computational mechanics: A classification of concepts with
# application to smooth shells. Computer Methods in Applied Mechanics and
# Engineering, 155(3):273 – 305, 1998.
# 
# [4] P. G. Ciarlet. An introduction to differential geometry with
# applications to elasticity. Journal of Elasticity, 78-79(1-3):1–215, 2005.

u0_h, u1_h, u2_h, b0_h, b1_h = z_.split(deepcopy=True)
X0in = Expression(('r*sin(x[0])-x[0]','x[1]-x[1]','r*cos(x[0])'), r=rho, degree = 4)
initial_conf = project(X0in, V_y)
displacement = as_vector([u0_h, u1_h, u2_h])
scale_factor = 1e0 # scale factor to visualiza the deformed configuration
deformed_conf = X0in + scale_factor*displacement
deformed_conf = project(deformed_conf, V_y)
file1 = File("initial_configuration.pvd")
file1 << initial_conf
file2 = File("deformed_configuration-half.pvd")
file2 << deformed_conf