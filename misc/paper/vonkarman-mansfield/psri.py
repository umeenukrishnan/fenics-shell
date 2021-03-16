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
lagr1 = FiniteElement("Lagrange", triangle, degree = 1)
lagr2 = FiniteElement("Lagrange", triangle, degree = 2)
bubble = FiniteElement("B", triangle, degree = 3)
en_element = lagr2 + bubble
quad_degree = 2
mixed_element = MixedElement(3*[lagr2] + 2*[lagr1 + bubble])

# Function Space
U = FunctionSpace(mesh, mixed_element)
u = Function(U)
u_trial, u_t = TrialFunction(U), TestFunction(U)
print("ndof = %s "%U.dim())

# Components
z1, z2, z3, th1, th2 = split(u) # components
z1_t, z2_t, z3_t, th1_t, th2_t = split(u_t)
z = as_vector([z1, z2]) # displacement vector
theta = as_vector([th1, th2]) # rotation vector
z_t = as_vector([z1_t, z2_t])
theta_t = as_vector([th1_t, th2_t])

# Membrane strain and membrane energy density
e = sym(grad(z)) + 0.5*outer(grad(z3), grad(z3))
ev = strain_to_voigt(e)
psi_N = 0.5*dot(A*ev, ev)

# Shear strain and shear energy density
gamma = grad(z3) - theta
psi_T = 0.5*dot(S*gamma, gamma)

# Bending strain and bending energy density
k_T = as_tensor(Expression((("c/imp","0.0"),("0.0","c*imp")), c=1., imp=.999,  degree=0))
k = sym(grad(theta)) - k_T
kv = strain_to_voigt(k)
psi_M = 0.5*dot(D*kv, kv)

# Arnold & Brezzi "selective reduced integration"
dx = dx(metadata={'quadrature_degree': 4})
J_b = psi_M*dx
J_m = psi_N*dx
J_s = psi_T*dx
# membrane, shearing energy reduced integration
dx_h = dx(metadata={'quadrature_degree': quad_degree}) # reduced integration
J_mh = psi_N*dx_h
J_sh = psi_T*dx_h

# Arnold and Brezzi Partial Selective Reduced energy
# kappa_s = 1/t0**2
kappa_s = 1.
alpha = (t0/h_min)**2
energy_PSR = J_b + alpha*J_m + alpha*J_s + (kappa_s - alpha)*J_sh + (kappa_s - alpha)*J_mh

# Penalty and Lagrangian
L = energy_PSR
F = derivative(L, u, u_t)
J = derivative(F, u, u_trial)

# Boundary conditions (to avoid rigid body motions)
zero = project(Constant((0.)),U.sub(0).collapse())
zerov = project(Constant((0.,0.,0.,0.,0.)),U)
bc1 = DirichletBC(U,zerov,"near(x[0],0.) and near(x[1],0.)",method="pointwise")
bc2 = DirichletBC(U.sub(0),zero,"near(x[0],0.) and near(x[1],1.)",method="pointwise")
bcs = [bc1,bc2]

# Solver settings
init = Function(U)
u.assign(init)
problem = NonlinearVariationalProblem(F, u, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.newton_solver.absolute_tolerance = 1E-8

# Critical curvature and continuations steps
R0 = r0**2/t0
c_cr = 5.16/R0
cs = np.linspace(0.0, 1.5*c_cr, 30)

# Solution
kx = []
ky = []
kxy = []
ls_load = []

defplots_dir = "output/3dplots-psri/"
file = File(defplots_dir + "sol.pvd")

for count,i in enumerate(cs):
    print('------------------------------------------------')
    print('Now solving load increment ' + repr(count) + ' of ' + repr(len(cs)))

    k_T.c = i
    solver.solve()
    v0_h, v1_h, w_h, th0_h, th1_h = u.split(deepcopy=True)
    theta_h = as_vector([th0_h, th1_h])

    ls_load.append(i)

    # K_h = project(grad(grad(w_h)), TensorFunctionSpace(mesh, 'DG', 0))
    K_h = project(sym(grad(theta_h)), TensorFunctionSpace(mesh, 'DG', 0))
    Kxy = assemble(K_h[0,1]*dx)/domain_area
    Kxx = assemble(K_h[0,0]*dx)/domain_area
    Kyy = assemble(K_h[1,1]*dx)/domain_area

    kx.append(Kxx)
    ky.append(Kyy)
    kxy.append(Kxy)

    uvec = as_vector([v0_h, v1_h, w_h])
    uvec_pro = project(uvec, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
    uvec_pro.rename("u","u")
    file << uvec_pro

# Analytical solution
h_before = np.loadtxt("mansfield-solution/mansfield_ktpre.csv")
h_after = np.loadtxt("mansfield-solution/mansfield_ktpost.csv")
ls_Kbefore = np.loadtxt("mansfield-solution/mansfield_kpre.csv")
ls_K1after = np.loadtxt("mansfield-solution/mansfield_kxpost.csv")
ls_K2after = np.loadtxt("mansfield-solution/mansfield_kypost.csv")

fig = plt.figure()
plot(mesh)
plt.savefig("output/mesh-%sdiv.png"%ndiv)

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
plt.savefig("output/prsi-%s.png"%ndiv)

np.savetxt("output/data-psri-%s.csv"%ndiv, np.array([ls_load,kx,ky]))
