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

"""This demo program solves the Reissner-Mindlin-von-Karman equations on a circular plate with
lenticular cross section. The plate is free on the boundary. Analytical
solution can be found in the paper:

E. H. Mansfield, "Bending, buckling and curling of a heated elliptical plate."
Proceedings of the Royal Society of London A: Mathematical, Physical and
Engineering Sciences.  Vol. 288. No. 1414. The Royal Society, 1965.

We use the Duran-Liberman projection operator expressed in pure UFL where extra
Lagrange multipliers exist on the edge of each element to enforce the
compatibility condition.
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

# Define the mesh
radius = 1.0
n_div = 40
centre = Point(0.,0.)
domain_area = np.pi*radius**2
geom = mshr.Circle(centre, radius)
mesh = mshr.generate_mesh(geom, n_div)
h_max = mesh.hmax()
mesh.init()

# In-plane displacements, rotations, out-of-plane displacements
# shear strains and Lagrange multiplier field.
element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space
U_P = U.projected_space

# Define the material parameters
young, poisson, thickness = 1.0, 0.3, 1E-2
kappa = Constant(5.0/6.0)
E = Constant(young)
nu = Constant(poisson)
t = Constant(thickness)

u, u_t, u_ = TrialFunction(U_F), TestFunction(U_F), Function(U_F)
v_, theta_, w_, R_gamma_, p_ = split(u_)

# Then, we define the shear energy in terms of the shear strain space. Note
# that we must attach out own measure dx. This is so you can easily define
# multi-material problems if you want to.
psi_s = psi_T(R_gamma_, E=E, nu=nu, kappa=kappa, t=t)
L_s = psi_s*dx

# Defines the lenticular thinning of the plate
th_f = Expression('(1.0 - (x[0]*x[0])/(R*R) - (x[1]*x[1])/(R*R))', R=radius)

# Target inelastic curvature
k_T = as_tensor(Expression((("1.0*c","0.0*c"),("0.0*c","0.97*c")), c=1.0))
k_ef = k(theta_) - k_T

# Define the bending energy in terms of the standard rotation space.
psi_b = psi_M(k_ef, E=E, nu=nu, t=t)
L_b = psi_b*dx

# Define the (von Karman) membrane energy in terms of the standard displacement
# spaces.
e = von_karman_e(v_, grad(w_))
psi_m = psi_N(e, E=E, nu=nu, t=t)
L_m = psi_m*dx

# Fix the value in the centre to eliminate the nullspace
def center(x, on_boundary):
    return x[0]**2 + x[1]**2 < DOLFIN_EPS

bc_v = DirichletBC(U.sub(0), Constant((0.0,0.0)), center, method="pointwise")
bc_R = DirichletBC(U.sub(1), Constant((0.0,0.0)), center, method="pointwise")
bc_w = DirichletBC(U.sub(2), Constant(0.0), center, method="pointwise")
bcs = [bc_v, bc_R, bc_w]

# Define external work
f = Constant(0.0)
L_e = f*w_*dx

# Compute the Residual and Jacobian.
# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.
L_R = inner_e(gamma(theta_, w_) - R_gamma_, p_)

L = L_m + L_s + L_b + L_R - L_e
F = derivative(L, u_, u_t)
J = derivative(F, u_, u)

# Set the problem and the solver parameters
u_p_ = Function(U_P)
problem = ProjectedNonlinearProblem(U_P, F, u_, u_p_, bcs=bcs, J=J)
solver = NewtonSolver()
solver.absolute_tolerance = 1E-20
solver.relative_tolerance = 1E-5

# Analytical critical inelastic curvature (see E. H. Mansfield, 1962; Seffen, McMahon, 2006)
c_cr = (thickness/radius**2)*(8./(1.0 + poisson)**(1.5))
loadings = np.linspace(0.0, 1.75*c_cr, 40)

# Solution
kx = []
ky = []
kxy = []
ls_load = []

output_dir = "output/vonkarman-mitc-mansfield/"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

D = VectorFunctionSpace(mesh, 'CG', 2, dim=3)
fid = File(output_dir + "solution.pvd")
for count, i in enumerate(loadings):
	k_T.c = i
	solver.solve(problem, u_p_.vector())
	v_h, theta_h, w_h, R_theta_h, p_h = u_.split()
	K_h = project(k(theta_h), TensorFunctionSpace(mesh, 'DG', 0))
	Kxx = assemble(K_h[0,0]*dx)/domain_area
	# Kyy = assemble(K_h[1,1]*dx)/domain_area
	Kyy = 2.*w_h(0., 0.9999)
	Kxy = assemble(K_h[0,1]*dx)/domain_area
	ls_load.append(i)
	kx.append(Kxx)
	ky.append(Kyy)
	kxy.append(Kxy)
	vector = as_vector([v_h[0], v_h[1], w_h])
	disp = project(vector, D)
	disp.rename("disp", "disp") #see the QA reported below.
	fid << disp, i

np.savetxt(output_dir + "kx.csv", kx)
np.savetxt(output_dir + "ky.csv", ky)
np.savetxt(output_dir + "kT.csv", loadings)