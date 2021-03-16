..    # vim: set fileencoding=utf8 :

.. _RollUpCantilever:

======================================
A non-linear Naghdi roll-up cantilever
======================================

This demo is implemented in the single Python file :download:`demo_nonlinear-naghdi-cantilever.py`.

This demo program solves the non-linear Naghdi shell equations on a rectangular
plate with a constant bending moment applied. The plate rolls up completely on
itself. The numerical locking issue is cured using a Durán-Liberman approach. 

To follow this demo you should know how to:

- Define a :py:class:`MixedElement` and a :py:class:`FunctionSpace` from it.
- Define the Durán-Liberman (MITC) reduction operator using UFL for a linear
  problem, e.g. Reissner-Mindlin. This procedure extends simply to the non-linear
  problem we consider here.
- Write variational forms using the Unified Form Language.
- Automatically derive Jabobian and residuals using :py:func:`derivative`.
- Apply Dirichlet boundary conditions using :py:class:`DirichletBC` and :py:func:`apply`.
- Apply Neumann boundary conditions by marking a :py:class:`FacetFunction` and
  create a new :py:class:`Measure` object.
- Solve non-linear problems using :py:class:`ProjectedNonlinearProblem`.
- Output data to XDMF files with :py:class:`XDMFFile`.

This demo then illustrates how to:

- Define and solve a non-linear Naghdi shell problem with a *flat* reference
  configuration.
 
We begin by setting up our Python environment with the required modules::

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    from dolfin import *
    from ufl import RestrictedElement
    from fenics_shells import *

We set the default quadrature degree. UFL's built in quadrature degree
detection often overestimates the required degree for these complicated
forms::

    parameters["form_compiler"]["quadrature_degree"] = 2

Our reference middle surface is a rectangle :math:`\omega = [0, 12] \times [-0.5,
0.5]`::

    length = 12.0
    width = 1.0 
    P1, P2 = Point(0.0, -width/2.0), Point(length, width/2.0)
    mesh = RectangleMesh(P1, P2, 48, 4, "crossed")

We then define our :py:class:`MixedElement` which will discretise the in-plane
displacements :math:`v \in [\mathrm{CG}_1]^2`, rotations :math:`\beta \in
[\mathrm{CG}_2]^2`, out-of-plane displacements :math:`w \in \mathrm{CG}_1`, the
shear strains. Two further auxilliary fields are also considered, the reduced
shear strain :math:`\gamma_R`, and a Lagrange multiplier field :math:`p` which
ties together the Naghdi shear strain calculated from the primal variables and the
reduced shear strain :math:`\gamma_R`. Both :math:`p` and :math:`\gamma_R` are
are discretised in the space :math:`\mathrm{NED}_1`, the vector-valued Nédélec
elements of the first kind. The final element definition is then::

    element = MixedElement([VectorElement("Lagrange", triangle, 1),
                            VectorElement("Lagrange", triangle, 2),
                            FiniteElement("Lagrange", triangle, 1),
                            FiniteElement("N1curl", triangle, 1),
                            RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

We then pass our ``element`` through to the :py:class:`ProjectedFunctionSpace`
constructor.  As in the other documented demos, we can project out :math:`p`
and :math:`\gamma_R` fields at assembly time. We specify this by passing the
argument ``num_projected_subspaces=2``::

    U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
    U_F = U.full_space
    U_P = U.projected_space

We assume constant material parameters; Young's modulus :math:`E`, Poisson's
ratio :math:`\nu`, and thickness :math:`t`::

    E, nu = Constant(1.2E6), Constant(0.0) 
    mu = E/(2.0*(1.0 + nu))
    lmbda = 2.0*mu*nu/(1.0 - 2.0*nu) 
    t = Constant(1E-1)

Using only the `full` function space object ``U_F`` we setup our variational
problem by defining the Lagrangian of our problem. We begin by creating a
:py:class:`Function` and splitting it into each individual component function::
    
    u, u_t, u_ = TrialFunction(U_F), TestFunction(U_F), Function(U_F)
    v_, beta_, w_, Rgamma_, p_ = split(u_)

For the Naghdi problem it is convienient to recombine the in-plane
displacements :math:`v` and out-of-plane displacements :math:`w` into a single
vector field :math:`z`:: 

    z_ = as_vector([v_[0], v_[1], w_])

We can now define our non-linear Naghdi strain measures. Assuming the normal
fibres of the shell are unstrechable, we can parameterise the director vector
field :math:`d: \omega \to \mathbb{R}^3` using the two independent rotations
:math:`\beta`:: 

    d = as_vector([sin(beta_[1])*cos(beta_[0]), -sin(beta_[0]), cos(beta_[1])*cos(beta_[0])])
   
The deformation gradient :math:`F` can be defined as::
 
    F = grad(z_) + as_tensor([[1.0, 0.0],
                             [0.0, 1.0],
                             [Constant(0.0), Constant(0.0)]])

From which we can define the stretching (membrane) strain :math:`e`::

    e = 0.5*(F.T*F - Identity(2))

The curvature (bending) strain :math:`k`::

    k = 0.5*(F.T*grad(d) + grad(d).T*F)

and the shear strain :math:`\gamma`::

    gamma = F.T*d

We then define the constitutive law in terms of a general dual strain
measure tensor :math:`X`:: 

    S = lambda X: 2.0*mu*X + ((2.0*mu*lmbda)/(2.0*mu + lmbda))*tr(X)*Identity(2) 

From which we can define the membrane energy density::

    psi_N = 0.5*t*inner(S(e), e)

the bending energy density::

    psi_K = 0.5*(t**3/12.0)*inner(S(k), k)

and the shear energy density::

    psi_T = 0.5*t*mu*inner(Rgamma_, Rgamma_)

and the total energy density from all three contributions::

    psi = psi_N + psi_K + psi_T

We define the Durán-Liberman reduction operator by tying the shear strain
calculated with the displacement variables :math:`\gamma = F^T d` to the
reduced shear strain :math:`\gamma_R` using the Lagrange multiplier field
:math:`p`::

    L_R = inner_e(gamma - Rgamma_, p_)

We then turn to defining the boundary conditions and external loading.  On the
left edge of the domain we apply clamped boundary conditions which corresponds
to constraining all generalised displacement fields to zero::

    left = lambda x, on_boundary: x[0] <= DOLFIN_EPS and on_boundary
    bc_v = DirichletBC(U.sub(0), Constant((0.0, 0.0)), left)
    bc_a = DirichletBC(U.sub(1), Constant((0.0, 0.0)), left)
    bc_w = DirichletBC(U.sub(2), Constant(0.0), left)
    bcs = [bc_v, bc_a, bc_w]

On the right edge of the domain we apply a traction::

    # Define subdomain for boundary condition on tractions
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 12.0) <= DOLFIN_EPS and on_boundary        

    right_tractions = Right()
    exterior_facet_domains = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    exterior_facet_domains.set_all(0)
    right_tractions.mark(exterior_facet_domains, 1)
    ds = Measure("ds")(subdomain_data=exterior_facet_domains)

    M_right = Expression(('M'), M=0.0, degree=0)

    W_ext = M_right*beta_[1]*ds(1)

We can now define our Lagrangian for the complete system::

    L = psi*dx + L_R - W_ext
    F = derivative(L, u_, u_t) 
    J = derivative(F, u_, u)

Before setting up the non-linear problem with the special `ProjectedFunctionSpace`
functionality::

    u_p_ = Function(U_P)
    problem = ProjectedNonlinearProblem(U_P, F, u_, u_p_, bcs=bcs, J=J)
    solver = NewtonSolver()

and solving::

    solver.parameters['error_on_nonconvergence'] = False
    solver.parameters['maximum_iterations'] = 20
    solver.parameters['linear_solver'] = "mumps"
    solver.parameters['absolute_tolerance'] = 1E-20
    solver.parameters['relative_tolerance'] = 1E-6

    output_dir = "output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

We apply the moment with 20 continuation steps::

    M_max = 2.0*np.pi*E.values()[0]*t.values()[0]**3/(12.0*length)
    Ms = np.linspace(0.0, M_max, 20)
    
    w_hs = []
    v_hs = []
    
    for i, M in enumerate(Ms):
        M_right.M = M
        solver.solve(problem, u_p_.vector())
        
        v_h, theta_h, w_h, Rgamma_h, p_h = u_.split(deepcopy=True)
        z_h = project(z_, VectorFunctionSpace(mesh, "CG", 1, dim=3))
        z_h.rename('z', 'z')

        XDMFFile(output_dir + "z_{}.xdmf".format(str(i).zfill(3))).write(z_h)

        w_hs.append(w_h(length, 0.0))
        v_hs.append(v_h(length, 0.0)[0])

This problem has a simple closed-form analytical solution which we plot against
for comparison::

    w_hs = np.array(w_hs)
    v_hs = np.array(v_hs)

    Ms_analytical = np.linspace(1E-3, 1.0, 100)
    vs = 12.0*(np.sin(2.0*np.pi*Ms_analytical)/(2.0*np.pi*Ms_analytical) - 1.0)
    ws = -12.0*(1.0 - np.cos(2.0*np.pi*Ms_analytical))/(2.0*np.pi*Ms_analytical)

    fig = plt.figure(figsize=(5.0, 5.0/1.648))
    plt.plot(Ms_analytical, vs/length, "-", label="$v/L$")
    plt.plot(Ms/M_max, v_hs/length, "x", label="$v_h/L$")
    plt.plot(Ms_analytical, ws/length, "--", label="$w/L$")
    plt.plot(Ms/M_max, w_hs/length, "o", label="$w_h/L$")
    plt.xlabel("$M/M_{\mathrm{max}}$")
    plt.ylabel("normalised displacement")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/cantilever-displacement-plot.pdf")
    plt.savefig("output/cantilever-displacement-plot.png")

Unit testing
============

::

    def test_close():
        assert(np.isclose(w_h(length, 0.0)/length, 0.0, atol=1E-3, rtol=1E-3))
        assert(np.isclose(v_h(length, 0.0)[0]/length, -1.0, atol=1E-3, rtol=1E-3))
