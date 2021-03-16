from dolfin import *

from instant import inline, inline_module
#set_log_level(DEBUG)

c_code = r"""
#include <dolfin/la/GenericTensor.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/adaptivity/LocalAssembler.h>
#include <dolfin/fem/UFC.h>
#include <Eigen/Dense>
namespace dolfin {

class MITCAssembler : public AssemblerBase 
{
public:
    MITCAssembler () {}
    void assemble(GenericTensor& A, const FunctionSpace& V_o, const Form& a, const Form& a_p);
};

void MITCAssembler::assemble(GenericTensor& A, const FunctionSpace& V_o, const Form& a, const Form& a_p)
{   
    dolfin_assert(a_p);
    dolfin_assert(a_p.rank() == 2);
    dolfin_assert(a);
    dolfin_assert(a.rank() == 2);

    UFC a_p_ufc(a_p);
    dolfin_assert(a_p_ufc);

    UFC a_ufc(a);
    dolfin_assert(a_p_ufc);

    const Mesh& mesh = a_p.mesh();
    dolfin_assert(mesh); 

    std::vector<std::shared_ptr<const FunctionSpace>> V = a.function_spaces();

    // WARNING: API change in dolfin-dev to max_element_dofs()
    std::size_t rows = V[0]->dofmap()->max_cell_dimension();
    std::size_t cols = V[1]->dofmap()->max_cell_dimension();
    
    // WARNING: MatrixXd is column-major by default, and
    // LocalAssembler does the deep-copy by default in
    // column-order. Not what I expected given C++ is generally
    // always row-order.
    Eigen::MatrixXd A_cell = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd A_p_cell = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd A_o = Eigen::MatrixXd::Zero(V_o.dofmap()->max_cell_dimension(), V_o.dofmap()->max_cell_dimension());

    const MeshFunction<std::size_t>* cell_domains = a.cell_domains().get();
    const MeshFunction<std::size_t>* exterior_facet_domains = 
      a.exterior_facet_domains().get();
    const MeshFunction<std::size_t>* interior_facet_domains = 
      a.interior_facet_domains().get(); 

    std::vector<const GenericDofMap*> dofmaps;
        for (std::size_t i = 0; i < 2; ++i)
            dofmaps.push_back(V_o.dofmap().get());
    // Warning: API change in dolfin-dev to ArrayView
    std::vector<const std::vector<dolfin::la_index>* > dofs(2);

    std::vector<double> vertex_coordinates;
    ufc::cell ufc_cell;
    for (CellIterator cell(mesh); !cell.end(); ++cell) 
    {   
        A_o.setZero();
        A_p_cell.setZero();
        A_cell.setZero();
        cell->get_vertex_coordinates(vertex_coordinates);
        // TODO: Both of these calls to assemble contain an unnecessary copy
        LocalAssembler::assemble(A_p_cell, a_p_ufc, vertex_coordinates, ufc_cell, *cell, 
                                 cell_domains,
                                 exterior_facet_domains, 
                                 interior_facet_domains);
        LocalAssembler::assemble(A_cell, a_ufc, vertex_coordinates, ufc_cell, *cell,
                                 cell_domains,
                                 exterior_facet_domains,
                                 interior_facet_domains);
        std::size_t cell_dimension = V_o.dofmap()->max_cell_dimension();
        
        //std::cout << A_p_cell << std::endl << std::endl;
        //std::cout << A_cell << std::endl << std::endl;
        //std::cout << cell_dimension << std::endl << std::endl;
        // TODO: Recombine this all onto one line.
        A_o += (A_p_cell.transpose()*A_cell*A_p_cell).topLeftCorner(cell_dimension, cell_dimension);
        //std::cout << A_o << std::endl << std::endl;
        A_o += (A_p_cell.transpose()*A_cell).topLeftCorner(cell_dimension, cell_dimension);
        //std::cout << A_o << std::endl << std::endl;
        A_o += (A_cell*A_p_cell).topLeftCorner(cell_dimension, cell_dimension);
        //std::cout << A_o << std::endl << std::endl;
        
        //std::cout << A_o << std::endl << std::endl;
        for (std::size_t i = 0; i < 2; ++i)
        {
            // WARNING: Remove & for dolfin-dev
            dofs[i] = &(dofmaps[i]->cell_dofs(cell->index()));
        }
        A.add_local(A_o.data(), dofs);
    }
}
}
"""
cpp_additions = compile_extension_module(c_code)


parameters["reorder_dofs_serial"] = False

mesh = UnitTriangleMesh() 
mesh.init()
# Bit of a tough one this one!
# Standard rotation space
R = VectorFunctionSpace(mesh, "Lagrange", 2) + \
    VectorFunctionSpace(mesh, "Bubble", 3)
# Degrees of freedom associated with facet degrees of freedom 
# of second-order NED element of first-kind
RR_e = FunctionSpace(mesh, "N1curl", 2, restriction="facet")
# and of interior degrees of freedom
RR_i = VectorFunctionSpace(mesh, "DG", 0)
# and finally the full NED space. I need this full space to form the 'underlying'
# bilinear form correctly. Forming bilinear form on RR_e and RR_i doesn't work
# because of the way the restrict argument to RR_e operates.
RR = FunctionSpace(mesh, "N1curl", 2)
V = FunctionSpace(mesh, "CG", 2)

# I think the trick here is to introduce a third space
# where we define the projection from R to RR_e and RR_i
U_RR_s = MixedFunctionSpace([R, V, RR_e, RR_i])
# but generate the underyling cell tensor here
U_RR = MixedFunctionSpace([R, V, RR])
# Note that RR_e, RR_i are 'nested' in RR. Hopefully
# FEniCS is nice and lays out the DOFs like this. If not
# it will require some extra work of course.
# And all the final assembly is done into a Matrix defined
# on this space.
U = MixedFunctionSpace([R, V])

# dim(RR_e x RR_i) = dim(RR) so we should have:
assert U_RR_s.dim() == U_RR.dim()
# Clearly for the reduction to be a reduction
assert U.dim() < U_RR.dim()

# Begin defining projection operator
r, z, rr_e, rr_i = TrialFunctions(U_RR_s)
r_t, z_t, rr_e_t, rr_i_t = TestFunctions(U_RR_s)

facet_area = FacetArea(mesh)
cell_volume = CellVolume(mesh)
n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

dsp = Measure("ds", metadata={'quadrature_degree': 2})
dSp = Measure("dS", metadata={'quadrature_degree': 2})
dxp = Measure("dx", metadata={'quadrature_degree': 0})
# Complete guess/extrapolation on the cell volume bit!
#a_p = (facet_area*inner(r, t)*inner(rr_e_t, t))('+')*dSp + (facet_area*inner(r, t)*inner(rr_e_t, t))*dsp + (1.0/cell_volume)*inner(r, rr_i_t)*dxp
# Current using this to try and debug
a_p = (facet_area*inner(rr_e, t)*inner(rr_e_t, t))*dsp + (1.0/cell_volume)*inner(rr_i, rr_i_t)*dxp
A = assemble(a_p)
A_dense = A.array()
print A_dense
exit()

# Begin defining underlying bilinear form 
r, z, rr = TrialFunctions(U_RR)
r_t, z_t, rr_t = TestFunctions(U_RR)

import sys; sys.path.append("../../")
from fenics_shells.analytical.lovadina_clamped import Loading, Rotation, Displacement
E = 10920.0
nu = 0.3
tv = 10.0**-5
kappa = 5.0/6.0

theta_e = Rotation()
z_e = Displacement(t=tv, nu=nu)
f = Loading(E=E, nu=nu)

E = Constant(E)
nu = Constant(nu)
t = Constant(tv)
kappa = Constant(kappa)

e = lambda theta: sym(grad(theta))
B = lambda e: (E/(12.0*(1.0 - nu**2)))*((1.0 - nu)*e + nu*tr(e)*Identity(2))
F = (E*kappa*t**-2)/(2.0*(1.0 + nu))

a_U_RR = F*(-inner(rr, grad(z_t)) - inner(grad(z), rr_t) + inner(rr, rr_t))*dx

r, z = TrialFunctions(U)
r_t, z_t = TestFunctions(U)
a_U = inner(B(e(r)), e(r_t))*dx + F*inner(grad(z), grad(z_t))*dx

def all_boundary(x, on_boundary):
    return on_boundary

bcs = [DirichletBC(U, Constant((0.0, 0.0, 0.0)), all_boundary)]

A = PETScMatrix()
assemble(a_U, tensor=A, finalize_tensor=False)
a_p = fem.assembling._create_dolfin_form(a_p)
a_U_RR = fem.assembling._create_dolfin_form(a_U_RR)
cpp_additions.MITCAssembler().assemble(A, U, a_U_RR, a_p)
A.apply("add")

L = f*z_t*dx
b = PETScVector()
assemble(L, tensor=b)

for bc in bcs:
    bc.apply(A, b)

u_h = Function(U)
solver = LUSolver("umfpack")
solver.solve(A, u_h.vector(), b)

r_h, z_h = u_h.split()
print z_h((0.5, 0.5))
File("MITC7.pvd") << z_h
