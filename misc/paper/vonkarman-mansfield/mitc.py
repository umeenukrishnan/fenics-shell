from dolfin import *
from fenics_shells import *
import numpy as np
import mshr
import matplotlib.pyplot as plt

userpar = Parameters("user")
userpar.add("ndiv", 5)
userpar.add("t", 1e-2)
parameters.add(userpar)
parameters.parse()
# Planform geometry
r0 = 1.
ndiv = parameters.user.ndiv
domain_area = np.pi*r0**2
centre = Point(0.,0.)
geom = mshr.Circle(centre, r0)
mesh = mshr.generate_mesh(geom, ndiv)
h_min = mesh.hmin()

# Thickness
t0 = parameters.user.t
ts = interpolate(Expression('t0*(1.0 - (x[0]*x[0] + x[1]*x[1])/(r0*r0))', t0=t0, r0=r0, degree=2), FunctionSpace(mesh, 'CG', 2))

# Elastic costants
E = Constant(1.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)

# Elastic stiffnesses
A = (E*ts/t0**3/(1. - nu**2))*as_tensor([[1., nu, 0.],[nu, 1., 0.],[0., 0., (1. - nu)/2]])
D = (E*ts**3/t0**3/(12.*(1. - nu**2)))*as_tensor([[1., nu, 0.],[nu, 1., 0.],[0., 0., (1. - nu)/2]])
S = E*kappa*ts/t0**3/(2*(1. + nu))

# Element
element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

# Function space
U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space
U_P = U.projected_space
u, u_t, u_ = TrialFunction(U_F), TestFunction(U_F), Function(U_F)
v_, theta_, w_, R_gamma_, p_ = split(u_)
print("ndof = %s - %s"%(U_F.dim(), U_P.dim()))

# Membrane strain and membrane energy density
e = sym(grad(v_)) + 0.5*outer(grad(w_), grad(w_))
ev = strain_to_voigt(e)
psi_N = 0.5*dot(A*ev, ev)

# Shear strain and shear energy density
psi_T = 0.5*dot(S*R_gamma_, R_gamma_)

# Bending strain and bending energy density
k_T = as_tensor(Expression((("c/imp","0.0"),("0.0","c*imp")), c=1., imp=.998,  degree=0))
k = sym(grad(theta_)) - k_T
kv = strain_to_voigt(k)
psi_M = 0.5*dot(D*kv, kv)

# Duran Liberman operator
gamma = grad(w_) - theta_
L_R = inner_e(gamma - R_gamma_, p_)

# Penalty and Lagrangian
penalty = Constant(1e-4)*(dot(u_, u_))*dx
L = (psi_M + psi_T + psi_N)*dx + L_R + penalty
F = derivative(L, u_, u_t)
J = derivative(F, u_, u)

# Boundary conditions
bcs = []

# Solver settings
u_p_ = Function(U_P)
problem = ProjectedNonlinearProblem(U_P, F, u_, u_p_, bcs=bcs, J=J)
solver = NewtonSolver()
solver.parameters['absolute_tolerance'] = 1E-6

# Critical curvature and continuations steps
R0 = r0**2/t0
c_cr = 5.16/R0
cs = np.linspace(0.0, 1.5*c_cr, 30)

# Solution
kx = []
ky = []
kxy = []
ls_load = []

defplots_dir = "output/3dplots-mitc/"
file = File(defplots_dir + "sol.pvd")

for count, i in enumerate(cs):
    print('------------------------------------------------')
    print('Now solving load increment ' + repr(count) + ' of ' + repr(len(cs)))

    k_T.c = i
    solver.solve(problem, u_p_.vector())
    v_h, theta_h, w_h, R_theta_h, p_h = u_.split()
    v0_h, v1_h = v_h.split()

    ls_load.append(i)

    # K_h = project(grad(grad(w_h)), TensorFunctionSpace(mesh, 'DG', 0))
    K_h = project(sym(grad(theta_h)), TensorFunctionSpace(mesh, 'DG', 0))
    Kxx = assemble(K_h[0,0]*dx)/domain_area
    Kyy = assemble(K_h[1,1]*dx)/domain_area
    Kxy = assemble(K_h[0,1]*dx)/domain_area

    kx.append(Kxx)
    ky.append(Kyy)
    kxy.append(Kxy)

    uvec = as_vector([v0_h, v1_h, w_h])
    uvec_pro = project(uvec, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
    file << uvec_pro

# Analytical solution
h_before = np.loadtxt("mansfield-solution/mansfield_ktpre.csv")
h_after = np.loadtxt("mansfield-solution/mansfield_ktpost.csv")
ls_Kbefore = np.loadtxt("mansfield-solution/mansfield_kpre.csv")
ls_K1after = np.loadtxt("mansfield-solution/mansfield_kxpost.csv")
ls_K2after = np.loadtxt("mansfield-solution/mansfield_kypost.csv")

# Comparison
fig = plt.figure(figsize=(5.0, 5.0/1.648))
plt.plot(ls_load, kx, "o", color='orange', label=r"$k_{1h}$")
plt.plot(ls_load, ky, "x", color='red', label=r"$k_{2h}$")
plt.plot(h_before/R0, ls_Kbefore/R0, "-", color='b', label="Analytical solution")
plt.plot(h_after/R0, ls_K1after/R0, "-", color='b')
plt.plot(h_after/R0, ls_K2after/R0, "-", color = 'b')
plt.xlabel(r"inelastic curvature $\eta$")
plt.ylabel(r"curvature $k_{1,2}$")
plt.legend()
plt.tight_layout()
plt.savefig("output/mitc-%s.png"%ndiv)

np.savetxt("output/data-mitc-%s.csv"%ndiv, np.array([ls_load,kx,ky]))
