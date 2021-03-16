from dolfin import *

"""
Solution of the nonlinear generalized Naghdi shell, as derived from the
3D elasticity in Mardare's paper:

https://www.ljll.math.upmc.fr/~mardare/recherche/pdf/shellmodels.pdf

Here we do not allow thickness-stretching (5 parameters model):
parametrization of the rotations via spherical angles.
"""
parameters.form_compiler.quadrature_degree =  2
info(parameters, True)
# Create mesh
mesh = UnitSquareMesh(20, 20)

# Define function spaces
U = VectorFunctionSpace(mesh, "CG", 2, dim = 3)
A = VectorFunctionSpace(mesh, "CG", 2, dim = 2)
UA = MixedFunctionSpace([U,A])

# Define test and trial functions
(u, a) = TrialFunctions(UA)
(u_t, a_t) = TestFunctions(UA)

# Define initial values (?)
s_f = Function(UA)
s_f = interpolate(Expression(('0', '0.', '0.','0.', '0.')), UA)
(u_f, a_f) = s_f.split()

# Define Dirichlet boundary
def left(x,on_boundary):
    return on_boundary and (x[0] == 0. )

def up(x,on_boundary):
    return on_boundary and (x[1] == 1.)

def right(x,on_boundary):
    return on_boundary and (x[0] == 1. )
    
def bottom(x,on_boundary):
    return on_boundary and (x[1] == 0. )    

bc_ul = DirichletBC(UA.sub(0), Constant((0.0,0.0,0.0)), left)
bc_uu = DirichletBC(UA.sub(0), Constant((0.0,0.0,0.0)), up)
bc_ur = DirichletBC(UA.sub(0), Constant((0.0,0.0,0.0)), right)
bc_ub = DirichletBC(UA.sub(0), Constant((0.0,0.0,0.0)), bottom)
bc_al = DirichletBC(UA.sub(1), Constant((0.0,0.0)), left)
bc_au = DirichletBC(UA.sub(1), Constant((0.0,0.0)), up)
bc_ar = DirichletBC(UA.sub(1), Constant((0.0,0.0)), right)
bc_ab = DirichletBC(UA.sub(1), Constant((0.0,0.0)), bottom)
bc = [bc_ul, bc_uu, bc_al]

#~ Director vector
d = lambda a: as_vector([sin(a[1])*cos(a[0]), -sin(a[0]), cos(a[1])*cos(a[0])])
# Deformation gradients
F = lambda u: as_tensor([[1.0, 0.0],[0.0, 1.0],[Constant(0), Constant(0)]]) + grad(u)
# Stretching tensor
G = lambda F0: 0.5*(F0.T*F0 - Identity(2))
# Curvature tensor
R = lambda F0, d:  0.5*(F0.T*grad(d) + grad(d).T*F0)
# P-tensor
P = lambda d: 0.5*(grad(d).T*grad(d))
# Shear strain vector
g = lambda F0, d: F0.T*d
# Thickness strain vector
e = lambda d: grad(d).T*d

# Elasticity parameters
mu, lb = Constant(1.E1), Constant(1.E1) # Lame parameters
eps = Constant(0.1) # (Half)-Thickness parameter

# Dual stress tensor (constitutive law)
S = lambda X: 2*mu*X + ((2*mu*lb)/(2*mu + lb))*tr(X)*Identity(2)

# Body force
f = Expression(("0.*t","0.0000*t","t"), t =  1)

#~ Membrane energy density
psi_G = eps*inner(S(G(F(u_f))), G(F(u_f)))
#~ Bending energy density
psi_R = (eps**3/3.0)*inner(S(R(F(u_f), d(a_f))), R(F(u_f), d(a_f)))
#~ Shear energy density
psi_g = eps*2.0*mu*inner(g(F(u_f), d(a_f)), g(F(u_f), d(a_f))) #+ (inner(d_f ,d_f) - 1)
#~ Coupled Membrane-P energy density
psi_GP = (eps**3/3.0)*(inner(S(G(F(u_f))), P(d(a_f))) + inner(S(P(d(a_f))), G(F(u_f)))) 
#~ P energy density
psi_P = (eps**5/5.0)*inner(S(P(d(a_f))), P(d(a_f)))
# Thickness-stretching energy density
psi_e = (eps**3/3.0)*2*mu*inner(e(d(a_f)), e(d(a_f)))
# Stored strain energy density
psi = 0.5*(psi_G + psi_R + psi_g) + (psi_e + psi_GP + psi_P )

# Total potential energy
Pi = psi*dx - eps**3*inner(f, u_f)*dx


# Test
print "-------------------------------------------"
psi_Gv = assemble(psi_G*dx)
print "Membrane energy : " , psi_Gv
psi_Rv = assemble(psi_R*dx)
print "Bending energy : " , psi_Rv
psi_gv = assemble(psi_g*dx)
print "Shear energy : " , psi_gv
psi_GPv = assemble(psi_GP*dx)
print "Membrane-P energy: " , psi_Gv
psi_Pv = assemble(psi_P*dx)
print "P energy : " , psi_Pv
psi_ev = assemble(psi_e*dx)
print "Thickness-stretching energy : " , psi_ev

# First variation of Pi
F = derivative(Pi, (u_f, a_f), (u_t, a_t))

# Compute Jacobian of F
dF = derivative(F, (u_f, a_f), (u, a))

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, s_f, bcs=bc, J=dF)
solver = NonlinearVariationalSolver(problem)


prm = solver.parameters
prm['nonlinear_solver'] = "snes"
prm['snes_solver']['relative_tolerance'] = 1E-4
prm['snes_solver']['absolute_tolerance'] = 1E-7
prm['snes_solver']['error_on_nonconvergence'] = False
prm['snes_solver']['maximum_iterations'] = 20
#~ prm['snes_solver']['linear_solver'] = "superlu_dist"
prm['snes_solver']['linear_solver'] = "lu"
info(prm, True)
import numpy as np
loadings = np.linspace(0,30,10)
for t in loadings:
	f.t = t 
	solver.solve()
	print "-------------------------------------------"
	psi_Gv = assemble(psi_G*dx)
	print "Membrane energy : " , psi_Gv
	psi_Rv = assemble(psi_R*dx)
	print "Bending energy : " , psi_Rv
	psi_gv = assemble(psi_g*dx)
	print "Shear energy : " , psi_gv
	psi_GPv = assemble(psi_GP*dx)
	print "Membrane-P energy: " , psi_Gv
	psi_Pv = assemble(psi_P*dx)
	print "P energy : " , psi_Pv
	psi_ev = assemble(psi_e*dx)
	print "Thickness-stretching energy : " , psi_ev
u_sol, a_sol = s_f.split(deepcopy=True) # extract components
plot(u_sol, title="displacement", interactive = True, mode = "displacement")
plot(d(a_sol), title="director", interactive = True)

