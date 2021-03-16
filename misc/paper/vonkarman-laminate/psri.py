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

# Geometry
a_rad, b_rad = 0.3, 0.25 # [m]
ndiv = 16
domain_area = np.pi*a_rad*b_rad
centre = Point(0.,0.)
geom = mshr.Ellipse(centre, a_rad, b_rad)
mesh = mshr.generate_mesh(geom, ndiv)
h_min = mesh.hmin()
plot(mesh);plt.savefig("mesh.png")

# Thickness
h = Constant(1E-3) # [m]

# Elementary layer (see Fernandes Maurini)
E1 = 135.0E6 # [kN/m^2]
E2 = 9.5E6
G12 = 5.0E6
nu12 = 0.3
G23 = 5.0E6
alpha1 = -0.02E-6
alpha2 = 30E-6

# Stacking sequence
thetas = [np.pi/4., -np.pi/4., -np.pi/4., np.pi/4., -np.pi/4., np.pi/4., np.pi/4., -np.pi/4.]
n_layers = len(thetas)
hs = h*np.ones(n_layers)/n_layers
A, B, D = laminates.ABD(E1, E2, G12, nu12, hs, thetas)
Fs = laminates.F(G23, G23, hs, thetas)

# Element
lagr1 = FiniteElement("Lagrange", triangle, degree = 1)
lagr2 = FiniteElement("Lagrange", triangle, degree = 2)
bubble = FiniteElement("B", triangle, degree = 3)
en_element = lagr1 + bubble
quad_degree = 1
mixed_element = MixedElement(3*[lagr2] + 2*[lagr2])

# Function Space
U = FunctionSpace(mesh, mixed_element)
u = Function(U)
u_trial, u_t = TrialFunction(U), TestFunction(U)
print(U.dim())

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
psi_T = 0.5*dot(Fs*gamma, gamma)

# Inelastic curvature
k_T = as_tensor(Expression((("c00","c01"),("c01","c11")), c00=1., c01=1., c11=1.,  degree=0))

# Bending strain and bending energy density
k = sym(grad(theta))
kv = strain_to_voigt(k - k_T)
psi_M = 0.5*dot(D*kv, kv)

# Arnold & Brezzi "selective reduced integration"
J_b = psi_M*dx
J_m = psi_N*dx
J_s = psi_T*dx
# membrane, shearing energy reduced integration
dx_h = dx(metadata={'quadrature_degree': quad_degree}) # reduced integration
J_mh = psi_N*dx_h
J_sh = psi_T*dx_h

# Arnold and Brezzi Partial Selective Reduced energy
kappa_s = 1.
alpha = (h/h_min)**2
energy_PSR = J_b + J_m + alpha*J_s + (kappa_s - alpha)*J_sh

# Lagrangian sistem
penalty = Constant(1e-12)*(dot(u,u))*dx
L = energy_PSR + penalty
F = derivative(L, u, u_t)
J = derivative(F, u, u_trial)

# Initial guess
init = Function(U)
u.assign(init)

# Boundary conditions
bcs = []

# Solver settings
problem = NonlinearVariationalProblem(F, u, bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = 'snes'
solver.parameters.snes_solver.linear_solver =  'umfpack'
solver.parameters.snes_solver.maximum_iterations = 30
solver.parameters.snes_solver.absolute_tolerance = 1E-12

# from fenics_shells.analytical.vonkarman_heated import analytical_solution
from fenics_shells.analytical.vonkarman_heated import analytical_solution
c_cr, beta, R0, h_before, h_after, ls_Kbefore, ls_K1after, ls_K2after = analytical_solution(A, D, a_rad, b_rad)

# cs = np.linspace(0.0, 1.5*c_cr, 50)
cs = np.linspace(1E-6, 1E-2, 120)

# Solution
kx = []
ky = []
kxy = []
ls_load = []

for count,i in enumerate(cs):
    print('------------------------------------------------')
    print('Now solving load increment ' + repr(count) + ' of ' + repr(len(cs)))

    (NT,MT) = laminates.NM_T(E1, E2, G12, nu12, hs, thetas, 0., DeltaT_1=-i, alpha1=alpha1, alpha2=alpha2)
    k_Ti = project(-inv(D)*MT, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
    k_T.c00 = k_Ti(0.0,0.0)[0]
    k_T.c11 = k_Ti(0.0,0.0)[1]
    k_T.c01 = k_Ti(0.0,0.0)[2]

    # k_T.c = i
    solver.solve()
    v0_h, v1_h, w_h, th0_h, th1_h = u.split(deepcopy=True)
    u.assign(u)

    # ls_load.append(i*R0*k_Ti(0., 0.)[0])
    ls_load.append(i)

    K_h = project(grad(grad(w_h)), TensorFunctionSpace(mesh, 'DG', 0))
    Kxy = assemble(K_h[0,1]*dx)/domain_area
    Kxx = assemble(K_h[0,0]*dx)/domain_area
    Kyy = assemble(K_h[1,1]*dx)/domain_area
    kx.append(Kxx*R0/np.sqrt(beta))
    ky.append(Kyy*R0)
    kxy.append(Kxy*R0/(beta**(1.0/4.0)))

# Comparison
fig = plt.figure(figsize=(5.0, 5.0/1.648))
plt.plot(ls_load, kx, "o", color='orange', label=r"$k_{1h}$")
plt.plot(ls_load, ky, "x", color='red', label=r"$k_{2h}$")
# plt.plot(h_before, ls_Kbefore, "-", color='b', label="Analytical solution")
# plt.plot(h_after, ls_K1after, "-", color='b')
# plt.plot(h_after, ls_K2after, "-", color = 'b')
plt.xlabel(r"thermal gradient $\Delta T_1$")
plt.ylabel(r"curvature $k_{1,2}$")
plt.legend()
plt.tight_layout()
plt.savefig("PSRI-curvature_bifurcation.png")