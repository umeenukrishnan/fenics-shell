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

import dolfin as df
from dolfin import *

def cdg_energy(theta, M, stabilization, mesh, bcs_theta=None, dS=df.dS):
    r"""Return the continuous/discontinuous terms for a fourth-order plate model.

    .. math::
        \pi_{cdg} = - \partial_n w  \cdot M_{n}(w) + \frac{1}{2} \frac{\alpha}{|e|} |\partial_n w |^2

    Args:
        theta: Rotations, UFL or DOLFIN Function of rank (2,) (vector).
        M: UFL form of bending moment tensor of rank (2,2) (tensor).
        stabilization: a constant or ulf expression providing the stabilization parameter
                       of the continuous/discontinuous formulation.
                       This should be an eximation of the norm of the bending stiffness

        mesh: DOLFIN mesh.
        bcs_theta (Optional): list of dolfin.DirichletBC for the rotations
            theta. Defaults to None.
        dS: (Optional). Measure on interior facets. Defaults to dolfin.dS.

    Returns:
        a dolfin.Form associated with the continuous/discontinuous formulation.

    The Kirchhoff-Love plate model is a fourth-order PDE, giving rise to a
    weak form with solution in Sobolev space :math:`H^2(\Omega)`. Because FEniCS
    does not currently include support for :math:`H^2(\Omega)` conforming elements
    we implement a hybrid continuous/discontinuous approach, allowing the use of
    Lagrangian elements with reduced regularity requirements.

    Description can be found in the paper:
        G. Engel, K. Garikipati, T. J. R. Hughes, M. G. Larson, L. Mazzei and
        R. L. Taylor, "Continuous/discontinuous finite element approximations of
        fourth-order elliptic problems in structural and continuum mechanics with
        applications to thin beams and plates, and strain gradient elasticity" Comput.
        Method. Appl. M., vol. 191, no. 34, pp. 3669-3750, 2002.
    """
    # Various bits of DG machinery
    h = CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(mesh)

    theta_n_jump = jump(theta, n)
    theta_n = inner(theta, n)
    M_n = inner(M, outer(n, n))
    # Stabilisation term on interior facets of mesh
    Pi_CDG = -inner(theta_n_jump, avg(M_n))*dS + \
                  (1.0/2.0)*stabilization('+')/h_avg*inner(theta_n_jump, theta_n_jump)*dS

    # Weak enforcement of Dirichlet boundary conditions on rotations
    if bcs_theta is not None:
        # Turn sub_domains attached to boundary conditions
        # into marked facets on the boundary.
        mf = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
        mf.set_all(0)
        for (i, bc) in enumerate(bcs_theta):
            print(dir(bc))
            rotation_boundary = bc.user_sub_domain()
            rotation_boundary.mark(mf, i + 1)

        ds = Measure("ds")(subdomain_data=mf)
        for (i, bc) in enumerate(bcs_theta):
            try:
                imposed_rotation = interpolate(bc.value(), FunctionSpace(bc.function_space().collapse())) 
            except RuntimeError:
                # Not very elegant, how can we query the FunctionSpace to know
                # if it needs collapsing?
                imposed_rotation = interpolate(bc.value(), FunctionSpace(bc.function_space()))
            # Weakly impose Dirichlet condition
            theta_n_eff = theta_n - imposed_rotation
            psi_CDG_ds = -inner(theta_n_eff, M_n) + \
                          (1.0/2.0)*stabilization/h*inner(theta_n_eff, theta_n_eff)
            Pi_CDG += psi_CDG_ds*ds(i + 1)

    return Pi_CDG


def cdg_stabilization(E, t):
    r"""Returns the stabilization parameter as the norm of the bending
    stiffness matrix.

    Args:
        E: Young's modulus, Constant or Expression.

        t: Thickness, Constant or Expression.

    Returns:
        a dolfin.Coefficient providing the stabilization parameter
        of the continuous/discontinuous formulation.
    """
    return E*t**3 # FIXME: shouldn't be divided by 12?

