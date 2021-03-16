..    # vim: set fileencoding=utf8 :

.. _KirchhoffClamped:

============================
Clamped Kirchhoff-Love plate
============================

This demo is implemented in a single Python file
:download:`demo_kirchhoff-love-clamped.py`.

This demo program solves the out-of-plane Kirchhoff-Love equations on the unit
square with uniform transverse loading and fully clamped boundary conditions.

We use the Continuous/Discontinuous Galerkin (CDG) formulation to allow the use of
:math:`H^1`-conforming elements for this fourth-order, or :math:`H^2`-type
problem. This demo is very similar to the `Biharmonic equation
<https://fenics-dolfin.readthedocs.io/en/latest/demos/biharmonic/python/demo_biharmonic.py.html>`_
demo in the main DOLFIN repository, and as such we recommend reading that page
first. The main differences are:

- we express the stabilisation term in terms of a Lagrangian functional rather
  than as a bilinear form,
- the Lagrangian function in turn is expressed in terms of the bending moment
  and rotations rather than the primal field variable,
- and we show how to place Dirichlet boundary conditions on the first-derivative of
  the solution (the rotations) using a weak approach.

We begin as usual by importing the required modules::

    from dolfin import *
    from fenics_shells import *

and creating a mesh::

    mesh = UnitSquareMesh(64, 64)

We use a second-order scalar-valued element for the transverse
displacement field :math:`w \in CG_2`::

    element_W = FiniteElement("Lagrange", triangle,  2)
    W = FunctionSpace(mesh, element_W)

and then define :py:class:`Function`, :py:class:`TrialFunction` and
:py:class:`TestFunction` objects to express the variational form::

    w_ = Function(W)
    w = TrialFunction(W)
    w_t = TestFunction(W)

We take constant material properties throughout the domain::

    E = Constant(10920.0)
    nu = Constant(0.3)
    t = Constant(1.0)

The Kirchhoff-Love model, unlike the Reissner-Mindlin model, is a
`rotation-free` model: the rotations :math:`\theta` do not appear explicitly as
degrees of freedom. Instead, the rotations of the Kirchhoff-Love model are
calculated from the transverse displacement field as:

.. math::
    \theta = \nabla w

which can be expressed in UFL as::

    theta = grad(w_)
    
The bending tensor can then be calculated from the derived rotation field
in exactly the same way as for the Reissner-Mindlin model:

.. math::
    k = \frac{1}{2}(\nabla \theta + (\nabla \theta)^T) 

or in UFL::

    k = variable(sym(grad(theta)))

The function :py:func:`variable` annotation is important and will allow us to
take differentiate with respect to the bending tensor ``k`` to derive the
bending moment tensor, as we will see below.

Again, identically to the Reissner-Mindlin model we can calculate the bending
energy density as::

    D = (E*t**3)/(24.0*(1.0 - nu**2))
    psi_M = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

For the definition of the CDG stabilisation terms and the (weak) enforcement of
the Dirichlet boundary conditions on the rotation field, we need to explicitly
derive the moment tensor :math:`M`. Following standard arguments in elasticity,
a stress measure (here, the moment tensor) can be derived from the bending
energy density by taking its derivative with respect to the strain measure
(here, the bending tensor):

.. math::
    M = \frac{\partial \psi_M}{\partial k}

or in UFL::

    M = diff(psi_M, k)

We now move onto the CDG stabilisation terms.

Consider a triangulation :math:`\mathcal{T}` of the domain :math:`\omega` with
the set of interior edges is denoted :math:`\mathcal{E}_h^{\mathrm{int}}`.
Normals to the edges of each facet are denoted :math:`n`.  Functions evaluated
on opposite sides of a facet are indicated by the subscripts :math:`+` and
:math:`-`.

The Lagrangian formulation of the CDG stabilisation term is then:

.. math::
    L_{\mathrm{CDG}}(w) = \sum_{E \in \mathcal{E}_h^{\mathrm{int}}} \int_{E} - [\!\![ \theta ]\!\!]  \cdot \left< M \cdot (n \otimes n) \right > + \frac{1}{2} \frac{\alpha}{\left< h_E \right>} \left< \theta \cdot n \right> \cdot \left< \theta \cdot n \right> \; \mathrm{d}s

Furthermore, :math:`\left< u \right> = \frac{1}{2} (u_{+} + u_{-})` operator,
:math:`[\!\![ u ]\!\!]  = u_{+} \cdot n_{+} + u_{-} \cdot n_{-}`, :math:`\alpha
\ge 0` is a penalty parameter and :math:`h_E` is a measure of the cell size. We
choose the penalty parameter to be on the order of the norm of the bending
stiffness matrix :math:`\dfrac{Et^3}{12}`.

This can be written in UFL as::
    
    alpha = E*t**3
    h = CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2.0 
    
    n = FacetNormal(mesh)
    
    M_n = inner(M, outer(n, n))
    
    L_CDG = -inner(jump(theta, n), avg(M_n))*dS + \
               (1.0/2.0)*(alpha('+')/h_avg)*inner(jump(theta, n), jump(theta, n))*dS

We now define our Dirichlet boundary conditions on the transverse displacement
field::

    class AllBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Boundary conditions on displacement
    all_boundary = AllBoundary()
    bcs_w = [DirichletBC(W, Constant(0.0), all_boundary)]

Because the rotation field :math:`\theta` does not enter our weak formulation
directly, we must weakly enforce the Dirichlet boundary condition on the
derivatives of the transverse displacement :math:`\nabla w`.

We begin by marking the exterior facets of the mesh where we want to apply
boundary conditions on the rotation::

    facet_function = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    facet_function.set_all(0)
    all_boundary.mark(facet_function, 1)

and then define an exterior facet :py:class:`Measure` object from that
subdomain data::

    ds = Measure("ds")(subdomain_data=facet_function)

In this example, we would like :math:`\theta_d = 0` everywhere on the boundary:: 
 
    theta_d = Constant((0.0, 0.0))

The definition of the exterior facets and Dirichlet rotation field were trivial
in this demo, but you could extend this code straightforwardly to
non-homogeneous Dirichlet conditions.

The weak boundary condition enforcement term can be written:
     
.. math::
    L_{\mathrm{BC}}(w) = \sum_{E \in \mathcal{E}_h^{\mathrm{D}}} \int_{E} - \theta_e  \cdot (M \cdot (n \otimes n))  + \frac{1}{2} \frac{\alpha}{h_E} (\theta_e \cdot n)  \cdot (\theta_e \cdot n)  \; \mathrm{d}s

where :math:`\theta_e = \theta - \theta_d` is the effective rotation field, and
:math:`\mathcal{E}_h^{\mathrm{D}}` is the set of all exterior facets of the triangulation
:math:`\mathcal{T}` where we would like to apply Dirichlet boundary conditions, or in UFL::
 
    theta_effective = theta - theta_d 
    L_BC = -inner(inner(theta_effective, n), M_n)*ds(1) + \
            (1.0/2.0)*(alpha/h)*inner(inner(theta_effective, n), inner(theta_effective, n))*ds(1) 

The remainder of the demo is as usual::

    f = Constant(1.0)
    W_ext = f*w_*dx

    L = psi_M*dx - W_ext + L_CDG + L_BC

    F = derivative(L, w_, w_t)
    J = derivative(F, w_, w)

    A, b = assemble_system(J, -F, bcs=bcs_w)
    solver = PETScLUSolver("mumps")
    solver.solve(A, w_.vector(), b)
    XDMFFile("output/w.xdmf").write(w_)

Unit testing
============

::

    def test_close():
        import numpy as np
        assert(np.isclose(w_((0.5, 0.5)), 1.265E-6, atol=1E-3, rtol=1E-3))
