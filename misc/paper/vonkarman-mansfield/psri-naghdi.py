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
sf = Constant(1.0)
A = (E*ts/t0**3/(1. - nu**2))*as_tensor([[1., nu, 0.],[nu, 1., 0.],[0., 0., sf*(1. - nu)/2]])
D = (E*ts**3/t0**3/(12.*(1. - nu**2)))*as_tensor([[1., nu, 0.],[nu, 1., 0.],[0., 0., sf*(1. - nu)/2]])
S = E*kappa*ts/t0**3/(2*(1. + nu))

# Function Space
P2 = FiniteElement("P", triangle, degree = 2)
bubble = FiniteElement("B", triangle, degree = 3)
Z = FunctionSpace(mesh, MixedElement(3*[P2 + bubble] + 2*[P2]))
z_ = Function(Z)
z, zt = TrialFunction(Z), TestFunction(Z)

# Components
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

# Strain measures
k_T = as_tensor(Expression((("c/imp","0.0"),("0.0","c*imp")), c=1., imp=.998,  degree=0))
e_elastic = lambda F: 0.5*(F.T*F - Identity(2))
k_elastic = lambda F, d: -0.5*(F.T*grad(d) + grad(d).T*F) - k_T
g_elastic = lambda F, d: F.T*d

ev = strain_to_voigt(e_elastic(F))
kv = strain_to_voigt(k_elastic(F,d))
gv = g_elastic(F,d)

# Energy densities
psi_m = 0.5*inner(A*ev, ev)
psi_b = 0.5*inner(D*kv, kv)
psi_s = 0.5*inner(S*gv, gv)

# PSRI
kappa_m = 1.
kappa_s = 1.
alpha = (t0/h_min)**2
dx = dx(metadata={'quadrature_degree': 4})
dx_h = dx(metadata={'quadrature_degree': 2}) # reduced integration
shear_energy = alpha*psi_s*dx + (kappa_s - alpha)*psi_s*dx_h
membrane_energy = alpha*psi_m*dx + (kappa_m - alpha)*psi_m*dx_h
bending_energy =  psi_b*dx

# Lagrangian
Pi_total = (bending_energy + membrane_energy + shear_energy)
residual = derivative(Pi_total, z_, zt)
hessian = derivative(residual, z_, z)

# Boundary conditions (to avoid rigid body motions)
zero = project(Constant((0.)),Z.sub(0).collapse())
zerov = project(Constant((0.,0.,0.,0.,0.)),Z)
bc1 = DirichletBC(Z,zerov,"near(x[0],0.) and near(x[1],0.)",method="pointwise")
bc2 = DirichletBC(Z.sub(0),zero,"near(x[0],0.) and near(x[1],1.)",method="pointwise")
bc3 = DirichletBC(Z.sub(1),zero,"near(x[0],1.) and near(x[1],0.)",method="pointwise")
bcs = [bc1,bc2,bc3]

# Initial guess
init = Function(Z)
z_.assign(init)

# Problem setting
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
prm['linear_solver'] = "umfpack"
prm['absolute_tolerance'] = 1E-8
prm['relaxation_parameter'] = 1.

output_dir = "output/naghdi-psri/"
file = File(output_dir + "sol.pvd")


# Critical curvature and continuations steps
R0 = r0**2/t0
c_cr = 5.16/R0
cs = np.linspace(0.0, 40.0*c_cr, 300)

# Solution
kx = []
ky = []
kxy = []
ls_load = []

for count,i in enumerate(cs):
    print('------------------------------------------------')
    print('Now solving load increment ' + repr(count) + ' of ' + repr(len(cs)))

    k_T.c = i
    solver.solve(problem, z_.vector())
    u0_h, u1_h, u2_h, b0_h, b1_h = z_.split(deepcopy=True)
    uvec = as_vector([u0_h, u1_h, u2_h])
    uvec_pro = project(uvec, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
    uvec_pro.rename("u","u")
    file << uvec_pro

    ls_load.append(i)

    K_h = project(-0.5*(F.T*grad(d) + grad(d).T*F), TensorFunctionSpace(mesh, 'DG', 0))
    Kxy = assemble(K_h[0,1]*dx)/domain_area
    Kxx = assemble(K_h[0,0]*dx)/domain_area
    Kyy = assemble(K_h[1,1]*dx)/domain_area

    kx.append(Kxx)
    ky.append(Kyy)
    kxy.append(Kxy)


# Comparison
fig = plt.figure(figsize=(5.0, 5.0/1.648))
plt.plot(ls_load, kx, "o", color='orange', label=r"$k_{1h}$")
plt.plot(ls_load, ky, "x", color='red', label=r"$k_{2h}$")
plt.xlabel(r"inelastic curvature $\eta$")
plt.ylabel(r"curvature $k_{1,2}$")
plt.legend()
plt.tight_layout()
plt.savefig("output/psri-naghdi.png")
np.savetxt("output/data-psri-naghdi-%s.csv"%ndiv, np.array([ls_load,kx,ky]))