# -*- coding: utf-8 -*-

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

import os 
import pytest

import numpy as np
import pandas as pd

import dolfin
from dolfin import *
from fenics_shells import *

import matplotlib
import matplotlib.pyplot as plt
from mpltools import annotation

def _analytical_lovadina(nx, function_space_type, t):
    """For given number of divisions, and an element, solve the Lovadina
    clamped plate problem for a given thickness parameter t.
    """
    mesh = UnitSquareMesh(nx, nx)
    U = function_space_type(mesh)

    U_F = U.full_space
    U_P = U.projected_space

    def all_boundary(x, on_boundary):
        return on_boundary

    bc_R = DirichletBC(U_P.sub(0), Constant((0.0, 0.0)), all_boundary)
    bc_V_3 = DirichletBC(U_P.sub(1), Constant((0.0)), all_boundary)
    bcs = [bc_R, bc_V_3]
   
    E = 10920.0
    kappa = 5.0/6.0
    nu = 0.3

    from fenics_shells.analytical.lovadina_clamped import Loading, Rotation, Displacement
    theta_e = Rotation(degree=3)
    w_e = Displacement(degree=6)
    f = Loading(degree=3)
    w_e.t = t
    f.t = t
    w_e.nu = nu
    f.nu = nu
    w_e.E = E
    f.E = E

    u = TrialFunction(U_F)
    u_t = TestFunction(U_F)
    u_ = Function(U_F)
    theta_, w_, R_gamma_, p_ = split(u_)

    E = Constant(E)
    nu = Constant(nu)
    t = Constant(t) 
    kappa = Constant(kappa)
    
    # Elastic energy density
    psi = psi_M(k(theta_), E=E, nu=nu, t=t) \
        + psi_T(R_gamma_, E=E, nu=nu, t=t, kappa=kappa) \
    # Elastic energy
    L_el = psi*dx

    # External work
    W_ext = inner(f, w_)*dx

    # Reduction operator
    L_R = inner_e(gamma(theta_, w_) - R_gamma_, p_)

    L = L_el + L_R - W_ext
    F = derivative(L, u_, u_t)
    J = derivative(F, u_, u)
 
    A, b = assemble(U_P, J, rhs(F))
    for bc in bcs:
        bc.apply(A, b)

    u_p_ = Function(U_P)
    
    solver = PETScLUSolver("mumps")
    solver.solve(A, u_p_.vector(), b)

    theta_h, w_h = u_p_.split()
    
    result = {}
    result['t'] = t.values()[0]
    result['nx'] = nx
    result['hmax'] = mesh.hmax()
    result['hmin'] = mesh.hmin()
    result['theta_l2'] = errornorm(theta_e, theta_h, norm_type='l2')/norm(theta_h, norm_type='l2')
    result['theta_h1'] = errornorm(theta_e, theta_h, norm_type='h1')/norm(theta_h, norm_type='h1')
    result['w_l2'] = errornorm(w_e, w_h, norm_type='l2')/norm(w_h, norm_type='l2')
    result['w_h1'] = errornorm(w_e, w_h, norm_type='h1')/norm(w_h, norm_type='h1')
    result['hmax'] = mesh.hmax()
    result['dofs'] = U.dim()
    
    return result, u_p_


def _runner(element, norms, expected_convergence_rates):
    """Given an element and norms, compare the computed convergence
    rate and the expected convergence rate and assert that the former
    is greater than the latter."""
   
    nxs = [16, 32, 64, 128]
    ts = [1E0, 1E-2, 1E-4] 
    
    results = []
    for t in ts:
        for nx in nxs:
            result, _ = _analytical_lovadina(nx, element, t)
            results.append(result)
    
    df = pd.DataFrame(results)
    print(df)

    # Check convergence rate
    ts = df['t'].unique()
    for norm, expected_convergence_rate in zip(norms, expected_convergence_rates):
        for t in ts:
            df_t = df[df['t'] == t]
            
            hs = df_t['hmax']
            errors = df_t[norm]

            actual_convergence_rate = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            message = "t = {:.1E} convergence rate in norm {} = {:.3f}, expected >= {:.3f}".format(t, norm, actual_convergence_rate, expected_convergence_rate)
            print(message)
            assert actual_convergence_rate >= expected_convergence_rate, message

    _plot_convergence(df)

    # Final run to get nice plot of result
    _, u_p = _analytical_lovadina(24, element, 1E-4)
    _plot_result(u_p)


def _plot_result(u_p):
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    fig = plt.figure(figsize=(4.0, 4.0))

    theta_h, w_h = u_p.split()
    ax = plot(w_h, mode='contourf')
    fig.colorbar(ax, format="%.0e")

    mesh = theta_h.function_space().mesh()
    theta = theta_h.compute_vertex_values(mesh)
     
    nv = mesh.num_vertices()
    gdim = mesh.geometry().dim()
    if len(theta) != gdim*nv:
        raise AttributeError('Vector length must match geometric dimension.')
    X = mesh.coordinates()
    X = [X[:, i] for i in range(gdim)]
    U = [theta[i*nv:(i + 1)*nv] for i in range(gdim)]
    ax = plt.gca()
    args = X + U
    ax.quiver(*args)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chinosi-solution.pdf"))


def _plot_convergence(df):
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    width = 5.0 
    height = width / 1.648
    size = (width, height)
    
    # All 4 norms for thin plate t = 1E-4
    norms = ['w_l2', 'w_h1', 'theta_l2', 'theta_h1']
    norms_pretty = [r'$w \; L^2$', r'$w \; H^1$', r'$\theta \; L^2$', r'$\theta \; H^1$']

    df_t = df[df['t'] == 1E-4]
    fig = plt.figure(figsize=size)
    ax = plt.gca()
    for norm, norm_pretty in zip(norms, norms_pretty):
        plt.loglog(df_t['hmax'], df_t[norm], 'o-', label=norm_pretty)
    ax.legend(loc=4)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'error')
    ax.set_xlim([10**-2, 10**-1])
    ax.minorticks_off()
    annotation.slope_marker((3E-2, 2E-3), (2, 1), ax=ax)
    annotation.slope_marker((3E-2, 3E-2), (1, 1), ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "duran-liberman-convergence-thin.pdf"))

    # Vary t, convergence in w_h1
    fig = plt.figure(figsize=size)
    ax = plt.gca()
    ts = df['t'].unique()
    for t in np.sort(ts)[::-1]:
        df_t = df[df['t'] == t]
        if t == 1E-4:
            ax.loglog(df_t['hmax'], df_t['w_h1'], 'o--', label="$t = ${:.0E}".format(t))
        else:
            ax.loglog(df_t['hmax'], df_t['w_h1'], 'o-', label="$t = ${:.0E}".format(t))
    ax.legend(loc=4)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$H^1(\omega)$ error')
    ax.set_xlim([10**-2, 10**-1])
    ax.minorticks_off()
    annotation.slope_marker((5E-2, 6E-2), (1, 1), ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "duran-liberman-w-locking-free.pdf"))

    # Vary t, convergence in theta_h1
    fig = plt.figure(figsize=size)
    ax = plt.gca()
    ts = df['t'].unique()
    for t in np.sort(ts)[::-1]:
        df_t = df[df['t'] == t]
        if t == 1E-4:
            ax.loglog(df_t['hmax'], df_t['theta_h1'], 'o--', label="$t = ${:.0E}".format(t))
        else:
            ax.loglog(df_t['hmax'], df_t['theta_h1'], 'o-', label="$t = ${:.0E}".format(t))
    ax.legend(loc=4)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$H^1(\omega, \mathbb{R}^2)$ error')
    ax.set_xlim([10**-2, 10**-1])
    ax.minorticks_off()
    annotation.slope_marker((5E-2, 1.5E-1), (1, 1), ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "duran-liberman-theta-locking-free.pdf"))


def test_duran_liberman():
    """Run test for DuranLiberman element type"""
    norms = ['theta_l2', 'theta_h1', 'w_l2', 'w_h1']
    #expected_convergence_rates = [2.0, 1.0, 2.0, 1.0]
    expected_convergence_rates = [1.8, 0.8, 1.8, 0.8]

    _runner(DuranLibermanSpace, norms, expected_convergence_rates)       

@pytest.mark.skipif(dolfin.__version__ != "1.5.0", reason="Calculation of dual \
                    DOFs of enriched elements disabled in DOLFIN >1.5.0")  
def test_mitc7():
    """Run test for MITC7 element type"""
    norms = ['theta_l2', 'theta_h1', 'w_l2', 'w_h1']
    #expected_convergence_rates = [3.0, 2.0, 3.0, 2.0]
    expected_convergence_rates = [2.8, 1.8, 2.8, 1.8]

    _runner(MITC7, norms, expected_convergence_rates)


if __name__ == "__main__":
    test_duran_liberman()
