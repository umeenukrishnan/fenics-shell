..    # vim: set fileencoding=utf8 :

=======================================
Simply supported Reissner-Mindlin plate
=======================================

This demo is implemented in the single Python file :download:`demo_reissner-mindlin-simply-supported.py`.

This demo program solves the out-of-plane Reissner-Mindlin equations on the
unit square with uniform transverse loading with simply supported boundary
conditions.

The first part of the demo is similar to the demo :ref:`ReissnerClamped`. ::

    from dolfin import *
    from fenics_shells import *
    
    mesh = UnitSquareMesh(64, 64)
    
    element = MixedElement([VectorElement("Lagrange", triangle, 2),
                            FiniteElement("Lagrange", triangle, 1),
                            FiniteElement("N1curl", triangle, 1),
                            FiniteElement("N1curl", triangle, 1)])

    U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
    U_F = U.full_space

    u_ = Function(U_F)
    theta_, w_, R_gamma_, p_ = split(u_)
    u = TrialFunction(U_F)
    u_t = TestFunction(U_F)

    E = Constant(10920.0)
    nu = Constant(0.3)
    kappa = Constant(5.0/6.0)
    t = Constant(0.0001)

    k = sym(grad(theta_))

    D = (E*t**3)/(24.0*(1.0 - nu**2))
    psi_M = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2) 

    psi_T = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

    f = Constant(1.0)
    W_ext = inner(f*t**3, w_)*dx

    gamma = grad(w_) - theta_

We instead use the :py:mod:`fenics_shells` provided :py:func:`inner_e` function::

    L_R = inner_e(gamma - R_gamma_, p_)
    L = psi_M*dx + psi_T*dx + L_R - W_ext

    F = derivative(L, u_, u_t)
    J = derivative(F, u_, u)

    A, b = assemble(U, J, -F)

and apply simply-supported boundary conditions::

    def all_boundary(x, on_boundary):
        return on_boundary

    def left(x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    def right(x, on_boundary):
        return on_boundary and near(x[0], 1.0)

    def bottom(x, on_boundary):
        return on_boundary and near(x[1], 0.0)

    def top(x, on_boundary):
        return on_boundary and near(x[1], 1.0)

    # Simply supported boundary conditions.
    bcs = [DirichletBC(U.sub(1), Constant(0.0), all_boundary),
           DirichletBC(U.sub(0).sub(0), Constant(0.0), top),
           DirichletBC(U.sub(0).sub(0), Constant(0.0), bottom),
           DirichletBC(U.sub(0).sub(1), Constant(0.0), left),
           DirichletBC(U.sub(0).sub(1), Constant(0.0), right)]

    for bc in bcs:
        bc.apply(A, b)

and solve the linear system of equations before writing out the results to
files in ``output/``::

    u_p_ = Function(U)
    solver = PETScLUSolver("mumps")
    solver.solve(A, u_p_.vector(), b)
    reconstruct_full_space(u_, u_p_, J, -F)

    save_dir = "output/"
    theta, w, R_gamma, p = u_.split()
    fields = {"theta": theta, "w": w, "R_gamma": R_gamma, "p": p}
    for name, field in fields.items():
        field.rename(name, name)
        field_file = XDMFFile("%s/%s.xdmf" % (save_dir, name))
        field_file.write(field)

We check the result against an analytical solution calculated using a 
series expansion::

    from fenics_shells.analytical.simply_supported import Displacement
    
    w_e = Displacement(degree=3)
    w_e.t = t.values()
    w_e.E = E.values()
    w_e.p = f.values()*t.values()**3
    w_e.nu = nu.values()
    
    print("Numerical out-of-plane displacement at centre: %.4e" % w((0.5, 0.5)))
    print("Analytical out-of-plane displacement at centre: %.4e" % w_e((0.5, 0.5)))

Unit testing
============

::

    def test_close():
        import numpy as np
        assert(np.isclose(w((0.5, 0.5)), w_e((0.5, 0.5)), atol=1E-3, rtol=1E-3))
