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

import numpy as np 

from dolfin import *
from fenics_shells import * 

def test_assembler_reissner_mindlin(space=DuranLibermanSpace):
    """This test checks that the linear algebra objects produced by the
    built-in FEniCS assembler and the fenics-shells projected assembler induce the
    same (natural, energy) norm for the Reissner-Mindlin problem.
    """
    # First of all we setup everything on the full space
    # as usual.    
    mesh = UnitSquareMesh(32, 32, "crossed")

    E = Constant(10920.0)
    nu = Constant(0.3)
    t = Constant(1E-2)
    kappa = Constant(5.0/6.0)

    U = space(mesh)
    U_F = U.full_space
    U_P = U.projected_space

    u_ = Function(U_F)
    theta_, w_, R_gamma_, p_ = split(u_) 
    u = TrialFunction(U_F)
    u_t = TestFunction(U_F)
    
    # Elastic energy density
    psi = psi_M(k(theta_), E=E, nu=nu, t=t) \
        + psi_T(R_gamma_, E=E, nu=nu, t=t, kappa=kappa) \
    # Elastic energy
    L_el = psi*dx

    # External work
    W_ext = inner(Constant(1.0), w_)*dx

    # Reduction operator
    L_R = inner_e(gamma(theta_, w_) - R_gamma_, p_)    

    L = L_el + L_R - W_ext
    F = derivative(L, u_, u_t)
    J = derivative(F, u_, u) 

    # Operators on full space
    A_f = assemble(J)
    F_f = assemble(F)

    # Operators on projected space
    A_p, F_p = assemble(U_P, J, F)
   
    # Initialise Functions on U_F and U_P  with 1.0 everywhere. These are used
    # to calculate the inner product induced by the above forms.    
    u_p = interpolate(Constant((1.0, 1.0, 1.0)), U_P)
    u_f = Function(U_F)
    # Generate compatible full solution
    reconstruct_full_space(u_f, u_p, J, F)

    # Tests
    from numpy.testing import assert_approx_equal
    # inner product induced by A_f and A_p
    assert_approx_equal(u_f.vector().inner(A_f*u_f.vector()),
                        u_p.vector().inner(A_p*u_p.vector()))
    # inner product induced by F_p and F_f
    assert_approx_equal(F_f.inner(u_f.vector()),
                        F_p.inner(u_p.vector()))
