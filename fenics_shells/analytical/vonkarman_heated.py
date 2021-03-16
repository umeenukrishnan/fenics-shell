#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2018 Matteo Brunetti
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
Analytical solution for elliptic orthotropic von Karman plate with lenticular 
thickness subject to a uniform field of inelastic curvatures.
"""

import numpy as np

def analytical_solution(Ai, Di, a_rad, b_rad):

    beta = Ai(0,0)[4]/Ai(0,0)[0]
    nu = Ai(0,0)[1]/Ai(0,0)[0]
    gamma = Ai(0,0)[8]/Ai(0,0)[0]
    rho = gamma/(1.0 - (nu**2/beta))
    mu = nu/np.sqrt(beta) 
    eta = gamma/np.sqrt(beta)

    # - analytical dimensionless critical values (see A. Fernandes et al., 2010)
    hQ = np.sqrt(2*eta)*(1.0 + 2.0*eta + mu)/(1.0 + mu) # since kTx = kTy = h, hQ as in the reference paper
    hP_pos = 2*np.sqrt(1.0 - mu)/(1.0 + mu) # since kTx = kTy = h, hP+ as in the reference paper
    h_cr = np.minimum(hQ, hP_pos)

    # - characteristic radius for lenticular cross-section
    ratio = b_rad/a_rad
    psiq = (1.0/(48.0*np.pi**2))*(1.0 - (nu**2/beta))*ratio**2/(5.0 + 2.0*((1.0/rho) - (nu/beta))*ratio**2 + (5.0/beta)*ratio**4)
    R0 = 2.0*np.sqrt(psiq*Ai(0,0)[0]/Di(0,0)[0])*np.pi*a_rad*b_rad

    # - critical curvature
    c_cr = h_cr/R0

    h_before = np.linspace(0.0, 0.9999*h_cr, 100)
    h_after = np.linspace(1.00001*h_cr, 1.5*h_cr, 100)
    lsh = [i for i in h_before] + [j for j in h_after]
    ls_f = [(1.0 + mu)*(np.sqrt(3.0)*np.sqrt(27*j**2 + 4.0*(1.0 + mu)) + 9.0*j) for j in h_before]
    ls_Kbefore = [(2.0**(1.0/3.0)*i**(2.0/3.0) - 2.0*3.0**(1.0/3.0)*(1.0 + mu))/(6.0**(2.0/3.0)*i**(1.0/3.0)) for i in ls_f]
    ls_K1after = [j/2.0*(1.0 + mu) + np.sqrt(j*j/4.0*(1.0 + mu)*(1.0 + mu) - (1.0 - mu)) for j in h_after]
    ls_K2after = [j/2.0*(1.0 + mu) - np.sqrt(j*j/4.0*(1.0 + mu)*(1.0 + mu) - (1.0 - mu)) for j in h_after]

    return (c_cr, beta, R0, h_before, h_after, ls_Kbefore, ls_K1after, ls_K2after)

