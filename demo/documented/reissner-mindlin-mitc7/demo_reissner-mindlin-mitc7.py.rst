..    # vim: set fileencoding=utf8 :
     
.. _ReissnerMITC7:

=========================================
Clamped Reissner-Mindlin plate with MITC7
=========================================

This demo is implemented in a single Python file
:download:`demo_reissner-mindlin-mitc7.py`.

This demo program solves the out-of-plane Reissner-Mindlin equations on the
unit square with uniform transverse loading with fully clamped boundary
conditions.  The MITC7 reduction operator is used, instead of the Durán
Liberman one as in the :ref:`ReissnerClamped` demo.

We express the MITC7 projection operator in the Unified Form Language.
Lagrange multipliers are required on the edge and in the interior of each
element to tie together the shear strain calculated from the primal variables 
and the conforming shear strain field.

Unlike the other Reissner-Mindlin documented demos, e.g :ref:`ReissnerClamped`,
where the FEniCS-Shells :py:func:`assemble` function is used to eliminate all
of the additional degrees of freedom at assembly time, we keep the full problem
with all of the auxilliary variables here.

We begin as usual by importing the required modules::

    from dolfin import *
    from ufl import EnrichedElement, RestrictedElement
    from fenics_shells import *

and creating a mesh::

    mesh = UnitSquareMesh(32, 32)

The MITC7 element for the Reissner-Mindlin plate problem consists of:

- a second-order scalar-valued element for the transverse displacement field
  :math:`w \in \mathrm{CG}_2`,
- second-order bubble-enriched vector-valued element for the rotation field
  :math:`\theta \in [\mathrm{CG}_2]^2`, 
- the reduced shear strain :math:`\gamma_R` is discretised in the space
  :math:`\mathrm{NED}_2`, the second-order vector-valued Nédélec element of the
  first kind,
- and two Lagrange multiplier fields :math:`p` and :math:`r` which tie together
  the shear strain calculated from the primal variables :math:`\gamma = \nabla
  w - \theta` and the reduced shear strain :math:`\gamma_R`. :math:`p` and
  :math:`r` are discretised in the space :math:`\mathrm{NED}_2` restricted to
  the element edges and the element interiors, respectively.

The final element definition is::

    element = MixedElement([FiniteElement("Lagrange", triangle, 2),
                            VectorElement(EnrichedElement(FiniteElement("Lagrange", triangle, 2) + FiniteElement("Bubble", triangle, 3))),
                            FiniteElement("N1curl", triangle, 2),
                            RestrictedElement(FiniteElement("N1curl", triangle, 2), "edge"),
                            VectorElement("DG", triangle, 0)])

The definition of the bending and shear energies and external work are standard and identical to those in :ref:`ReissnerClamped`::

    U = FunctionSpace(mesh, element) 
    
    u_ = Function(U)
    w_, theta_, R_gamma_, p_, r_  = split(u_)
    u = TrialFunction(U)
    u_t = TestFunction(U)

    E = Constant(10920.0)
    nu = Constant(0.3)
    kappa = Constant(5.0/6.0)
    t = Constant(0.001)
    
    k = sym(grad(theta_))
    D = (E*t**3)/(24.0*(1.0 - nu**2))
    psi_M = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)
    
    psi_T = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

    f = Constant(1.0)
    W_ext = inner(f*t**3, w_)*dx

We require that the shear strain calculated using the displacement unknowns
:math:`\gamma = \nabla w - \theta` be equal, in a weak sense, to the conforming
shear strain field :math:`\gamma_R \in \mathrm{NED}_2` that we used to define
the shear energy above.  We enforce this constraint using `two` Lagrange
multiplier fields field :math:`p \in \mathrm{NED}_2` restricted to the edges
and :math:`r \in \mathrm{NED}_2` restricted to the interior of the element. We
can write the Lagrangian of this constraint as:

.. math::
    L_R(\gamma, \gamma_R, p, r) = \int_{e} \left( \left\lbrace \gamma_R - \gamma \right\rbrace \cdot t \right) \cdot \left( p \cdot t \right) \; \mathrm{d}s + \int_{T} \left\lbrace \gamma_R - \gamma \right\rbrace \cdot r \; \mathrm{d}x

where :math:`T` are all cells in the mesh, :math:`e` are all of edges of the
cells in the mesh and :math:`t` is the tangent vector on each edge.

This operator can be written in UFL as::

    n = FacetNormal(mesh)
    t = as_vector((-n[1], n[0]))

    gamma = grad(w_) - theta_
    
    inner_e = lambda x, y: (inner(x, t)*inner(y, t))('+')*dS + \
                           (inner(x, t)*inner(y, t))*ds

    inner_T = lambda x, y: inner(x, y)*dx

    L_R = inner_e(gamma - R_gamma_, p_) + inner_T(gamma - R_gamma_, r_)

We set homogeneous Dirichlet conditions for the reduced shear strain, edge
Lagrange multipliers, transverse displacement and rotations::

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = [DirichletBC(U.sub(0), Constant(0.0), all_boundary),
           DirichletBC(U.sub(1), project(Constant((0.0, 0.0)), U.sub(1).collapse()), all_boundary),
           DirichletBC(U.sub(2), Constant((0.0, 0.0)), all_boundary),
           DirichletBC(U.sub(3), Constant((0.0, 0.0)), all_boundary)]

Before assembling in the normal way::

    L = psi_M*dx + psi_T*dx + L_R - W_ext
    F = derivative(L, u_, u_t)
    J = derivative(F, u_, u)

    A, b = assemble_system(J, -F, bcs=bcs)

    solver = PETScLUSolver("mumps")
    solver.solve(A, u_.vector(), b)

    w_, theta_, R_gamma_, p_, r_ = u_.split() 

Finally we output the results to XDMF to the directory ``output/``::

    save_dir = "output"
    fields = {"theta": theta_, "w": w_}
    for name, field in fields.items():
        field.rename(name, name)
        field_file = XDMFFile("%s/%s.xdmf"%(save_dir,name))
        field_file.write(field)

The resulting ``output/*.xdmf`` files can be viewed using Paraview.

Unit testing
============

::

    def test_close():
        import numpy as np
        assert(np.isclose(w_((0.5, 0.5)), 1.265E-6, atol=1E-3, rtol=1E-3)) 
