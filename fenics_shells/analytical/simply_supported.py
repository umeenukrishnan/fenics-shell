#!/usr/bin/python
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

"""
Analytical solution for simply-supported Reissner-Mindlin square plate under a
uniform transverse load.
"""

import numpy as np

from dolfin import UserExpression

def _exact_solution_constants(E, nu, t, p, iterations=200):
    ms = np.arange(1, iterations, 2)
    ns = np.copy(ms)

    m_square = np.tile(ms, (ms.shape[0], 1))
    n_square = m_square.T
    mn = np.outer(ms, ns)

    D = E*t**3/(12.0*(1 - nu**2))
    G = E/(2.0*(1.0 + nu))
    k = 5.0/6.0

    A11 = E*t/(1.0 - nu**2)
    A12 = nu*E*t*(1.0 - nu**2)
    A22 = A11
    A44 = G*t
    A55 = G*t
    A66 = G*t

    D11 = D
    D12 = nu*D
    D22 = D
    D66 = G*t**3/12.0

    Q11 = E/(1.0 - nu**2)
    Q12 = nu*Q11
    Q22 = Q11
    Q66 = G
    Q44 = G
    Q55 = G

    A = np.pi*m_square
    B = np.pi*n_square

    s11 = k*(A55*A**2 + A44*B**2)
    s12 = k*(A55*A)
    s13 = k*(A44*B)

    s22 = (D11*A**2 + D66*B**2 + k*A55)
    s23 = (D12 + D66)*A*B
    s33 = D66*A**2 + D22*B**2 + k*A44

    b0 = s22*s33 - s23*s23
    b1 = s23*s13 - s12*s33
    b2 = s12*s23 - s22*s13
    bmn = s11*b0 + s12*b1 + s13*b2

    Q = p*16.0/(mn*np.pi**2)

    W = b0*Q/bmn
    PhiX = b1*Q/bmn
    PhiY = b2*Q/bmn

    return (A, B, W, PhiX, PhiY)


class Displacement(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _precalculate_constants(self):
        self.A, self.B, self.W, self.PhiX, self.PhiY = _exact_solution_constants(self.E, self.nu, self.t, self.p)
    
    def eval(self, value, x):
        if not hasattr(self, 'A'):
            self._precalculate_constants()
        
        value[0] = np.sum(self.W*np.sin(x[0]*self.A)*np.sin(x[1]*self.B))


class Rotation(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _precalculate_constants(self):
        self.A, self.B, self.W, self.PhiX, self.PhiY = _exact_solution_constants(self.E, self.nu, self.t, self.p)

    def eval(self, value, x):
        if not hasattr(self, 'A'):
            self._precalculate_constants()

        value[0] = -np.sum(self.PhiX*np.cos(x[0]*self.A)*np.sin(x[1]*self.B))
        value[1] = -np.sum(self.PhiY*np.sin(x[0]*self.A)*np.cos(x[1]*self.B))

    def value_shape(self):
        return (2,)


