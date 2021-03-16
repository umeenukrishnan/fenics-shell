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

from dolfin import * 
from ..functions import ProjectedFunctionSpace

def DuranLibermanSpace(mesh):
    """
    A helper function which returns a FiniteElement for the simulation of the
    out-of-plane Reissner-Mindlin problem without shear-locking based on the
    ideas of Duran and Liberman's paper:

    R. Durán and E. Liberman, "On mixed finite element methods for the
    Reissner-Mindlin plate model" Math. Comp., vol. 58, no. 198, pp. 561–573,
    1992.

    Args:
        mesh (dolfin.Mesh): a mesh of geometric dimension 2.

    Returns:
        fenics_shells.ProjectedFunctionSpace.
    """
    if mesh.geometry().dim() != 2:
        dolfin.cpp.dolfin_error("function_spaces.py", "generate DuranLiberman" + \
                                "function space", "mesh must have geometric dimension 2")
    
    element = MixedElement([VectorElement("Lagrange", triangle, 2),
                            FiniteElement("Lagrange", triangle, 1),
                            FiniteElement("N1curl", triangle, 1),
                            FiniteElement("N1curl", triangle, 1)])
        
    U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
    return U


def MITC7Space(mesh, space_type="multipliers"):
    """
    Warning: Currently not working due to regressions in FFC/FIAT.

    A helper function which returns a FunctionSpace for the simulation of the
    out-of-plane Reissner-Mindlin problem without shear-locking based on the
    ideas of Brezzi, Bathe and Fortin's paper:
    
    "Mixed-interpolated elements for Reissner–Mindlin plates" Int.  J. Numer.
    Meth. Engng., vol. 28, no. 8, pp. 1787–1801, Aug. 1989.
    http://dx.doi.org/10.1002/nme.1620280806

    In the case that space_type is "multipliers", a dolfin.FunctionSpace will
    be returned with an additional Lagrange multiplier field to tie the shear
    strains computer with the primal variables (transverse displacement and
    rotations) to the independent shear strain variable.

    In the case that space_type is "primal", a dolfin.FunctionSpace will
    be returned with just the primal (transverse displacement and rotation)
    variables. Of course, any Reissner-Mindlin problem constructed with this
    approach will be prone to shear-locking. 

    Args:
        mesh (dolfin.Mesh): a mesh of geometric dimension 2.
        space_type (Optional[str]): Can be "primal", or
        "multipliers". Default is "multipliers".

    Returns:
        dolfin.FunctionSpace.
    """
    dolfin.cpp.dolfin_error("function_spaces.py", "generate MITC7 function space",
                            "A regressions in DOLFIN 1.6.0 and 1.7.0dev mean that MITC7 doesn't " + \
                            "currently work. The regression is that we cannot evaluate " + \
                            "dofs on an enriched element. The `multipliers` " + \
                            "version of MITC7 did work with FEniCS 1.5.0")

    if mesh.geometry().dim() != 2:
        dolfin.cpp.dolfin_error("function_spaces.py", "generate MITC7 function space",
                                "mesh must have geometric dimension 2")

    element = [EnrichedElement(VectorElement("Lagrange", triangle, 2),
                               VectorElement("Bubble", triangle, 3)),
               FiniteElement("Lagrange", triangle, 2)]
    if space_type == "multipliers":
        element += [FiniteElement("N1curl", triangle, 2),
                    RestrictedElement(FiniteElement("N1curl", triangle, 2), "edge"),
                    RestrictedElement(FiniteElement("N1curl", triangle, 2), "interior")]
        element = MixedElement(element)
        U = FunctionSpace(mesh, element)

        return U
    elif space_type == "primal":
        element = MixedElement(element)
        element = MixedElement(element)
        U = FunctionSpace(mesh, element)

        return U
    elif space_type == "projected":
        dolfin.cpp.dolfin_error("function_spaces.py", "generate MITC7 function space",
                                "space_type projected not yet supported for MITC7")
    else:
        dolfin.cpp.dolfin_error("function_spaces.py", "generate DuranLiberman function space",
                                "space_type %s not recognised" % space_type)
