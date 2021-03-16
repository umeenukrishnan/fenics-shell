# Copyright (C) 2017 Jack S. Hale
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

"""This demo program solves the out-of-plane Reissner-Mindlin equations on an
L-shaped plate clamped along the two edges that form the re-entrant corner,
whilst free on the other edges. This problem is described in Section 5.1.1 of
da Veiga et al.  http://dx.doi.org/10.1007/s10543-000-0000-x

This demo shows how to reconstruct the reduced rotations from the primal
rotation variable using the reduction operator.  This reduced rotation is then
used to calculate an aposteriori error estimate from da Veiga et al.
http://dx.doi.org/10.1137/11085640X . This error estimator, without the
post-processing scheme also described in da Veiga et al. is very similar to the
one outlined Carstensen and Hu. The main difference is that de Veiga et al.
also develop indicators that exist on certain subsets of the boundary to better
resolve the boundary layer effects inherent in the Reissner-Mindlin problem, see:
http://dx.doi.org/10.1090/S0025-5718-07-02028-5 

We use the Duran-Liberman projection operator to alleviate the problem of shear
locking in the Kirchhoff limit.

This demo uses some nice ideas from Marie Rognes' script for the Poisson
problem at: http://nbviewer.ipython.org/gist/meg-simula/299a24c401f38678c6ec
"""

import os
import numpy as np
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required to run this demo.")

from dolfin import *
from ufl import rot
from fenics_shells import *

try:
    from mshr import *
except ImportError:
    raise ImportError("mshr is required to run this demo.")

import matplotlib
import matplotlib.pyplot as plt

output_dir = os.path.join(os.getcwd(), "output")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Material parameters 
E = Constant(10920.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)

# Now boundary conditions. These should be defined on the primal space.
def vertical_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.5) and (not near(x[1], 1.0))


def horizontal_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0.5) and (not near(x[0], 1.0))


def clamped_boundary(x, on_boundary):
    return on_boundary and (vertical_boundary(x, on_boundary) or horizontal_boundary(x, on_boundary))


def free_boundary(x, on_boundary):
    return on_boundary and (not clamped_boundary(x, on_boundary))


def main():
    # Thick plate
    results_thick, meshes_thick = solve(Constant(1E-1))
    print(results_thick)
    results_thick.to_json(os.path.join(output_dir, "results_adaptive_thick.json"))

    results_uniform_thick, _ = solve(Constant(1E-1), uniform_refinement=True, max_refinements=8)
    print(results_uniform_thick)
    results_uniform_thick.to_json(os.path.join(output_dir, "results_uniform_thick.json"))

    # Thin plate
    results_thin, meshes_thin = solve(Constant(1E-4))
    print(results_thin)
    results_thin.to_json(os.path.join(output_dir, "results_adaptive_thin.json"))

    results_uniform_thin, _ = solve(Constant(1E-4), uniform_refinement=True, max_refinements=8)
    print(results_uniform_thin)
    results_uniform_thin.to_json(os.path.join(output_dir, "results_uniform_thin.json"))
    
    matplotlib.rcParams['lines.linewidth'] = .5
    
    fig1 = plt.figure()
    plot(meshes_thick[3], backend="matplotlib")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesh_thick_3.pdf"))
    
    fig2 = plt.figure()
    plot(meshes_thick[6], backend="matplotlib")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesh_thick_6.pdf"))
    
    fig1 = plt.figure()
    plot(meshes_thin[3], backend="matplotlib")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesh_thin_3.pdf"))
    
    fig2 = plt.figure()
    plot(meshes_thin[6], backend="matplotlib")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesh_thin_6.pdf"))


def solve(t, uniform_refinement=False, max_refinements=16):
    relative_tolerance = 1E-4
    alpha = 0.2

    # Create an l-shaped mesh
    big_box = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
    small_box = Rectangle(Point(0.0, 0.0), Point(0.5, 0.5))
    mesh = generate_mesh(big_box - small_box, 1)
    mesh = refine(mesh)
    
    results = []
    meshes = []
    for _ in range(0, max_refinements):
        result = {}

        (u_fh, f) = reissner_mindlin_solver(mesh, t)
        etas = estimate(u_fh, f, t)
        
        result['dofs'] = u_fh.function_space().dim()
        result['eta_squared'] = etas.vector().sum()
        result['h_max'] = u_fh.function_space().mesh().hmax()
        result['h_min'] = u_fh.function_space().mesh().hmin()
        result['cells'] = u_fh.function_space().mesh().num_cells()
        result['eta'] = np.sqrt(result['eta_squared'])
        results.append(result)        
        
        relative_error = result['eta_squared']/results[0]['eta_squared'] 

        if np.sqrt(relative_error) < np.sqrt(relative_tolerance):
            break
        
        if uniform_refinement is True:
            markers = MeshFunction("bool", mesh, mesh.geometry().dim(), True)
        else:
            markers = mark(alpha, etas)
        # See: http://fenicsproject.org/qa/4188/distributed-expressions-pointwise-evaluations-allow_expression
        parameters['ghost_mode'] = 'none'
        meshes.append(Mesh(mesh))
        mesh = refine(mesh, markers)
        parameters['ghost_mode'] = 'shared_facet'

    df = pd.DataFrame(results)
    return (df, meshes)


def reissner_mindlin_solver(mesh, t):
    # This solver is almost identical to the one contained in the file
    # demo_reissner-mindlin-clamped.py, but we use the DuranLibermanSpace
    # helper function to create the ProjectedFunctionSpace automatically.
    U = DuranLibermanSpace(mesh)
 
    
    U_F = U.full_space
    U_P = U.projected_space
    
    u_ = Function(U_F)
    theta_, w_, R_gamma_, p_ = split(u_)
    u = TrialFunction(U_F)
    u_t = TestFunction(U_F)

    psi_b = psi_M(k(theta_), E=E, nu=nu, t=Constant(1.0))
    L_b = psi_b*dx
    F_b = derivative(L_b, u_, u_t)
    a_b = derivative(F_b, u_, u)
   
    psi_s = psi_T(R_gamma_, E=E, nu=nu, t=t**-2, kappa=kappa)
    L_s = psi_s*dx
    F_s = derivative(L_s, u_, u_t)
    a_s = derivative(F_s, u_, u)

    L_R = inner_e(gamma(theta_, w_) - R_gamma_, p_)
    F_R = derivative(L_R, u_, u_t)
    a_R = derivative(F_R, u_, u)
    
    a = a_b + a_s + a_R

    # Uniform transverse loading
    f = Constant(1.0) 
    L = derivative(f*w_*dx, u_, u_t) 
 

    # Clamped on the re-entrant corner. 
    bcs = [DirichletBC(U, Constant((0.0, 0.0, 0.0)), clamped_boundary)]
    
    # Now we use projected_assemble to assemble a_b in the normal way, and apply
    # the action of our projection (c == d) on a_s. This returns a matrix defined on
    # the primal space (theta, w) only.
    A, b = assemble(U_P, a, L)

    for bc in bcs:
        bc.apply(A, b)
    
    u_h = Function(U_P)
    solver = PETScLUSolver("mumps")
    solver.solve(A, u_h.vector(), b)

    theta_h, w_h = u_h.split()
    
    # Reconstruct the full solution. 
    reconstruct_full_space(u_, u_h, a, L)

    # Return the solution in the full space U_F and the loading
    return (u_, f)


def estimate(u_fh, f, t):
    theta_h, w_h, R_gamma_h, p_h = u_fh.split()
    mesh = u_fh.function_space().mesh()

    # Shear stress
    R_gamma_ = variable(R_gamma_h)
    psi_s = psi_T(R_gamma_, E=E, nu=nu, kappa=kappa, t=t**-2)
    T_h = project(diff(psi_s, R_gamma_), R_gamma_h.function_space().collapse())
    
    # Bending moments
    k_ = variable(k(theta_h))
    psi_b = psi_M(k_, E=E, nu=nu, t=Constant(1.0))
    M_h = project(diff(psi_b, k_), TensorFunctionSpace(mesh, "CG", 1)) 

    # Reduced rotations
    R_theta_h = project(grad(w_h) - R_gamma_h, R_gamma_h.function_space().collapse()) 
    
    # mesh quantities
    h_K = CellDiameter(mesh)
    h_E = FacetArea(mesh)
    n_E = FacetNormal(mesh)

    # Assemble indicators eta on DG0 space.
    V_E = FunctionSpace(mesh, "DG", 0)
    eta_t = TestFunction(V_E)
    eta = Function(V_E)

    # Beginning with terms on element interiors
    eta_1 = inner(h_K**2*(t**2 + h_K**2)*inner(f + div(T_h), f + div(T_h)), eta_t)*dx
    eta_2 = inner(h_K**2*inner(div(M_h) + T_h, div(M_h) + T_h), eta_t)*dx
    
    # terms on element edges
    eta_3 = inner(avg(h_E)*inner(jump(M_h, n_E), jump(M_h, n_E)), avg(eta_t))*dS 
    eta_4 = inner((avg(h_E*(t**2 + h_E**2))*inner(jump(T_h, n_E), jump(T_h, n_E))), avg(eta_t))*dS
    
    # inconsistency term
    eta_5 = inner(inner(rot(theta_h - R_theta_h), rot(theta_h - R_theta_h)), eta_t)*dx + \
            inner(inner(theta_h - R_theta_h, theta_h - R_theta_h), eta_t)*dx
    
    
    # free boundary error indicator
    free = AutoSubDomain(free_boundary)
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    free.mark(boundaries, 1)
    ds = Measure("ds", subdomain_data=boundaries)
    
    eta_6 = inner(h_E*inner(M_h*n_E, M_h*n_E), eta_t)*ds(1) \
        + inner(h_E*(h_E**2 + t**2)*inner(inner(T_h, n_E), inner(T_h, n_E)), eta_t)*ds(1)
    
    # all terms together
    assemble(eta_1 + eta_2 + eta_3 + eta_4 + eta_5 + eta_6, tensor=eta.vector())
    
    return eta


def mark(alpha, indicators):
    # TODO: Make this work in parallel!
    etas = indicators.vector().get_local()
    indices = etas.argsort()[::-1]
    sorted = etas[indices]

    total = sum(sorted)
    fraction = alpha*total

    mesh = indicators.function_space().mesh()
    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)

    v = 0.0
    for i in indices:
        if v >= fraction:
            break
        markers[int(i)] = True
        v += sorted[i]

    return markers


def test_quick():
    df, _ = solve(1E-4, max_refinements=12)
    # Check convergence rate around O(d^{-1/2})
    print(df)
    convergence_rate = np.polyfit(np.log(df['dofs']), np.log(df['eta']), 1)[0]
    assert -convergence_rate >= 0.5


if __name__ == "__main__":
    main()
