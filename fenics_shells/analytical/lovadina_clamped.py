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
Analytical solution for clamped Reissner-Mindlin plate
problem from Lovadina et al.
"""

from dolfin import UserExpression

class Loading(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[0] = ((self.E*self.t**3)/(12.0*(1.0 - self.nu**2))*((12.0*x[1]*(x[1] - 1.0)*
                    (5.0*x[0]**2 - 5.0*x[0] + 1.0)*
                    (2.0*x[1]**2*(x[1]-1.0)**2 +
                    x[0]*(x[0] - 1.0)*
                    (5.0*x[1]**2 - 5.0*x[1] + 1.0))) +
                    (12.0*x[0]*(x[0] - 1.0)*
                    (5.0*x[1]**2 - 5.0*x[1] + 1.0)*
                    (2.0*x[0]**2*(x[0]-1.0)**2 +
                    x[1]*(x[1] - 1.0)*
                    (5.0*x[0]**2 - 5.0*x[0] + 1.0)))))

class Displacement(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def eval(self, value, x):
        value[0] = (1.0/3.0)*x[0]**3*(x[0]-1.0)**3*x[1]**3*(x[1]-1.0)**3 - \
                   (2.0*self.t**2/(5.0*(1.0 - self.nu)))* \
                   ((x[1]**3*(x[1] - 1.0)**3*x[0]*(x[0]-1.0)*(5.0*x[0]**2 - 5.0*x[0] + 1.0)) + \
                   (x[0]**3*(x[0] - 1.0)**3*x[1]*(x[1]-1.0)*(5.0*x[1]**2 - 5.0*x[1] + 1.0)))


class Rotation(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def eval(self, value, x):
        value[0] = x[1]**3*(x[1] - 1.0)**3*x[0]**2*(x[0] - 1.0)**2*(2.0*x[0] - 1.0)
        value[1] = x[0]**3*(x[0] - 1.0)**3*x[1]**2*(x[1] - 1.0)**2*(2.0*x[1] - 1.0)

    def value_shape(self):
        return (2,)
