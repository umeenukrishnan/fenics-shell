# Copyright (C) 2015 Corrado Maurini
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

def ABD_reference_reddy():
    A = array([ [ 1.56136024e+07, 3.63107034e+05, 2.87142598e-11],
                [ 3.63107034e+05, 1.56136024e+07, 8.38407580e-10],
                [ 2.87142598e-11, 8.38407580e-10, 1.01360000e+06]]),
    B = array([ [ 1.28158628e+03, 0.00000000e+00, -2.59864051e-15],
                [ 0.00000000e+00, -1.28158628e+03, -7.58758860e-14],
                [ -2.59864051e-15, -7.58758860e-14, 0.00000000e+00]]),
    D = array([ [ 1.70505743e-01, 3.96524984e-03, 3.13569288e-19],
                [ 3.96524984e-03, 1.70505743e-01, 9.15569024e-18],
                [ 3.13569288e-19, 9.15569024e-18, 1.10688499e-02]])
    return (A, B, D)


def test_NM_T():
    # Create a mesh and define the function space
    mesh = UnitSquareMesh(25, 25)
    E1 = 26.25
    E2 = 1.5
    E3 = 3.0
    G12 = 1.04
    nu12 = 0.28
    alpha1 = 2.
    alpha2 = 15.0
    hs = 0.005*np.array([4., 2., 1., 2.]) # total thickness
    thetas = np.pi/180.*np.array([90, 45, 0, -45.])# orientation of the layers

    V_stresses = VectorFunctionSpace(mesh, "DG", 0, dim = 3)
    NT, MT = laminates.NM_T(E1, E2, G12, nu12, hs, thetas, -300., 0., alpha1, alpha2)
    NT_p = project(NT, V_stresses)
    MT_p = project(MT, V_stresses)

    # reference results (checked against a published result)
    NT_r = np.array([-476.79604629, -637.08413693, 0.])
    MT_r = np.array([-1.60288091, 1.60288091, 0.80144045])

    error = np.linalg.norm(NT_p(.21,.12)-NT_r) + np.linalg.norm(MT_p(.21,.12)-MT_r)
    tol = 1.e-6
    err_msg = "Error on N_T and M_T is larger that expeceted, error = %.3f, expected %.3f" % (error, tol)

    assert error < tol, err_msg

def test_ABD():
    mesh = UnitSquareMesh(25, 25)
    E1 = 26.25
    E2 = 1.5
    E3 = 3.0
    G12 = 1.04
    nu12 = 0.28
    alpha1 = 2.
    alpha2 = 15.0
    hs = 0.005*np.array([4., 2., 1., 2.]) # total thickness
    thetas = np.pi/180.*np.array([90, 45, 0, -45.])# orientation of the layers

    # Calculate the laminates matrices (ufl matrices)
    A, B, D = laminates.ABD(E1, E2, G12, nu12, hs, thetas)

    V_stiffness = TensorFunctionSpace(mesh, 'DG', 0, shape = (3,3))
    A_p = project(A, TensorFunctionSpace(mesh, 'DG', 0, shape = (3,3)))
    B_p = project(B, TensorFunctionSpace(mesh, 'DG', 0, shape = (3,3)))
    D_p = project(D, TensorFunctionSpace(mesh, 'DG', 0, shape = (3,3)))

    # reference results (checked against a published result)
    A_r = np.array([0.32636895, 0.13334055, 0., 0.13334055, 0.69928963, 0., 0., 0., 0.1611555])
    B_r = 1.e-3*np.array([2.58565188, 1.14355497, -0.93230171, 1.14355497, -4.87276181, -0.93230171, -0.93230171, -0.93230171, 1.14355497])
    D_r = 1.e-6*np.array([46.22822424, 22.02473652, -18.64603423, 22.02473652, 127.80462402, -18.64603423, -18.64603423, -18.64603423, 26.71850882])

    # Check
    error = np.linalg.norm(A_p(.21,.12)-A_r) + np.linalg.norm(B_p(.21,.12)-B_r) + np.linalg.norm(D_p(.21,.12)-D_r)
    tol = 1.e-6
    err_msg = "Error on ABD is larger that expeceted, error = %.3f, expected %.3f" % (error, tol)

    assert error < tol, err_msg
