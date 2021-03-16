# Copyright (C) 2015 Jack S. Hale
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
import pytest
import numpy as np

from dolfin import *
from fenics_shells import *

def _analytical_lovadina(nx, t=1.0):
    mesh = UnitSquareMesh(nx, nx)
    
    element_W = FiniteElement("Lagrange", triangle, 2)
    W = FunctionSpace(mesh, element_W)

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = [DirichletBC(W, Constant(0.0), all_boundary)]
    bcs_theta = [DirichletBC(W, Constant(0.0), all_boundary)]

    w = TrialFunction(W)
    w_t = TestFunction(W)
    w_ = Function(W)

    E = Constant(10920.0)
    nu = Constant(0.3)
    alpha = cdg_stabilization(E, t)

    theta_ = grad(w_)
    k_ = k(theta_)

    # Bending moment
    D = (E*t**3)/(12.0*(1.0 - nu**2)) # bending stiffness
    M_ = D*((1.0 - nu)*k_ + nu*tr(k_)*Identity(2)) # bending stress tensor
    # Bending energy density
    psi_b = 1./2.*inner(M_, k_)
    
    # Discontinuos contribution to the elastic energy density
    # Stabilization parameter
    alpha = cdg_stabilization(E, t)
    Pi_cdg = cdg_energy(theta_, M_, alpha, mesh, bcs_theta=bcs_theta, dS=dS)

    from fenics_shells.analytical.lovadina_clamped import Loading, Displacement

    f = Loading(degree=3)
    f.t = t
    f.E = E
    f.nu = nu
    
    W_ext = f*w_*dx
    Pi = psi_b*dx + Pi_cdg - W_ext

    F = derivative(Pi, w_, w_t)
    J = derivative(F, w_, w)

    A, b = assemble_system(J, -F, bcs=bcs)
    solver = PETScLUSolver()
    solver.solve(A, w_.vector(), b)

    w_e = Displacement(degree=6)
    w_e.t = 0.0
    w_e.nu = nu
    
    result = {}
    result['hmax'] = mesh.hmax()
    result['hmin'] = mesh.hmin()
    result['w_l2'] = errornorm(w_e, w_, norm_type='l2')/norm(w_, norm_type='l2')
    result['w_h1'] = errornorm(w_e, w_, norm_type='h1')/norm(w_, norm_type='h1')

    return result

def _runner(norms, expected_convergence_rates):
    nxs = [8, 16, 32, 64]
    assert(len(nxs) > 1)

    results = []
    t = 1.0
    for nx in nxs:
        result = _analytical_lovadina(nx, t)
        results.append(result)

    for norm, expected_convergence_rate in zip(norms, expected_convergence_rates):
        hs = np.array([x['hmax'] for x in results])
        errors = np.array([x[norm] for x in results])

        actual_convergence_rate = np.polyfit(np.log(hs), np.log(errors), 1)[0]
        err_msg = "Convergence rate in norm %s = %.3f, expected %.3f" % \
                  (norm, actual_convergence_rate, expected_convergence_rate)
        assert actual_convergence_rate >= expected_convergence_rate, err_msg
        print(err_msg)

    print(results)

def test_cdg_kirchhoff_love():
    # TODO: Implement energy (H^2-equivalent) norm
    norms = ['w_l2', 'w_h1']
    expected_convergence_rates = [1.9, 0.9]

    _runner(norms, expected_convergence_rates)
