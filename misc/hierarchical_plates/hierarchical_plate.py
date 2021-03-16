#!/usr/bin/python
# -*- coding: utf-8 -*-

import dolfin as df
import ufl
import numpy as np

# TODO: Is it possible to replace these definitions with sympy or FIAT versions?

# Manually tabulated Legendre polynomials
# These are used as the director functions in the terminology of Dauge et al.
# Below x can actually be read as x_3 the
# coordinate through the thickness of the plate
legendre_polynomials = [lambda x: 1,
                        lambda x: x,
                        lambda x: 0.5*(3*x**2 - 1),
                        lambda x: 0.5*(5*x**3 - 3*x),
                        lambda x: 0.125*(35*x**4 - 30*x**2 + 3),
                        lambda x: 0.125*(63*x**5 - 70*x**3 + 15*x)]

# and manually computed derivatives of the above
derivative_legendre_polynomials = [lambda x: 0,
                                   lambda x: 1,
                                   lambda x: 3*x,
                                   lambda x: 0.5*(15*x - 3),
                                   lambda x: 0.5*(35*x**3 - 15*x),
                                   lambda x: 0.125*(315*x**4 - 210*x**2 + 15)]

# Contains weights and points for Gauss Legendre rules
from FIAT.quadrature import compute_gauss_jacobi_rule
gauss_legendre_quadrature = [compute_gauss_jacobi_rule(0, 0, n) for n in range(1,5)]

mesh = df.UnitSquareMesh(10, 10)

# As defined in eq. 32 of Dauge et al.
# TODO: Check that what I have done makes sense for q_i = 0
qs = [2, 2, 1]
degrees = [1, 1, 1]

# V.sub(0) and V.sub(1) correspond to u_T and V.sub(2) is u_3
# TODO: It is quite irritating that we cannot define a VectorFunctionSpace with dim=1
# as it requires continual if-else statements in the list comprehensions. Perhaps
# we can wrap VectorFunctionSpace/FunctionSpace and have it return the correct form.
V = df.MixedFunctionSpace([df.VectorFunctionSpace(mesh, "CG", degree, dim=q + 1) if q >= 1 \
                           else df.FunctionSpace(mesh, "CG", degree) for q, degree in zip(qs, degrees)])

zs = df.TrialFunctions(V)
ys = df.TestFunctions(V)

# Build vector expression for hierarchical expansions of displacements
def v_h(ys, qs, x_3):
    v_h = [None]*3
    for i, (q, y) in enumerate(zip(qs, ys)):
        terms = [y[n]*legendre_polynomials[n](x_3) if q >= 1
                 else y*legendre_polynomials[0](x_3) for n in range(0, q + 1)]
        v_h[i] = sum(terms)

    return df.as_vector(v_h)

# Build tensor expression for hierarchical expansion of derivative of displacements
def nabla_v_h(v_h, ys, qs, x_3):
    # UFL is obviously not aware of the 'meaning' attached to the direction x_3
    # so we must peform this differentiation automatically with respect to x_T
    # and then manually with respect to x_3 before generating the 
    # expression for the Jacobian matrix.
    
    # TODO: It would be nice if instead of taking ansatz ys we take v_h instead
    # but not sure if this is possible.
    # TODO: How do I assert that v_h and ys are related for error checking?
    nabla_v_h = np.empty((3, 3), dtype=object)
    
    # We construct the Jacobian row by row.
    # Here we are constructing the derivatives of u_i with respect to x_T.
    nabla_v_h[0, 0:2] = df.grad(v_h[0])
    nabla_v_h[1, 0:2] = df.grad(v_h[1])
    nabla_v_h[2, 0:2] = df.grad(v_h[2])
    # Finished constructing the first two columns of nabla_v_h.

    # Then we construct the derivatives of u_i with respect to x_3
    # which must be done manually as UFL has no knowledge of our third dimension!
    
    #print df.as_tensor(nabla_v_h)
    
# Once we have defined these two expressions it should be possible to use them
# in a full 3D elasticity setting! (maybe hyperelasticity too?).
u_h = v_h(zs, qs, 0.3)
nabla_u_h = nabla_v_h(u_h, ys, qs, 0.3)
