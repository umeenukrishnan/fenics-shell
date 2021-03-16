import dolfin as df

class CDGKirchhoff(object):
	"""
	Continuous/Discontinuous element as originally developed in Engel et al.
	Description can be found in the paper http://www.sciencedirect.com/science/article/pii/S0045782502002864
	"""
	
	def __init__(self, mesh, phi_onb):
		self.mesh = mesh
		self.phi_onb = phi_onb
		
		self.W = df.FunctionSpace(mesh, "CG", 2)
		self._w = df.TrialFunction(self.W)
		self._v = df.TestFunction(self.W)
		
	def continuous_discontinuous_form(self, M_w, M_v):
		"""
		Construct form related to numerical implementation
		Returns:
		"""
		w = self._w
		v = self._v
		
		h = df.CellSize(self.mesh)
		h_avg = (h('+') + h('-'))/2.0
		n = df.FacetNormal(self.mesh)
		
		#TODO alpha and beta should be automatically computed from the elasticity tensor norm.
		alpha = df.Constant(1.3E6)
		beta = alpha
		
		#TODO add a class in order to turn DirichletBCs on rotation in NeumannBCs as required by the C/DGformulation.
		from dolfin import grad, inner, dot, outer, jump, avg, dS, ds
		a_cd = - inner(avg(inner(M_w, outer(n, n))), jump(grad(v), n))*dS \
				 - inner(jump(grad(w), n), avg(inner(M_v, outer(n, n))))*dS \
				 + alpha('+')/h_avg*inner(jump(grad(w), n), jump(grad(v), n))*dS \
				 - inner(inner(M_w, outer(n, n)), dot(grad(v), n))*ds \
				 - inner(dot(grad(w), n), inner(M_v, outer(n, n)))*ds \
				 + beta*inner(dot(grad(w), n), dot(grad(v), n))*ds \
				 + inner(M_v, outer(n, n))*self.phi_onb*ds \
				 - beta*dot(grad(v), n)*self.phi_onb*ds
		
		return a_cd
		

import numpy as np
import unittest

class TestKirchhoffElementConvergence(unittest.TestCase):
    def _analytical_lovadina(self, nx, element_type):
        """For given number of divisions, and an element, solve the Lovadina
        clamped plate problem for a given thickness parameter t.
        """
        E = 10.92E6
        nu = 0.3
        t = 1.0
        w_onb = df.Constant((0.0))
        phi_onb = df.Constant((0.0))
        
        mesh = df.UnitSquareMesh(nx, nx)
        element = element_type(mesh, phi_onb)
        
        def all_boundary(x, on_boundary):
            return on_boundary

        bc_W = df.DirichletBC(element.W, w_onb, all_boundary)
        bcs = [bc_W]

        from kirchhoff_plate import KirchhoffPlate

        kh = KirchhoffPlate(element, bcs=bcs, E=E, nu=nu, t=t)                          

        from analytical_lovadina import Loading, Displacement        
        f = Loading(E=E, nu=nu)
        from dolfin import dx
        a_a = -f*kh.element._v*dx

        a = kh.construct_form(a_a=a_a)
        w_h = df.Function(kh.W)
        
        A, b = df.assemble_system(df.lhs(a), df.rhs(a), bcs=kh.bcs)
        solver = df.LUSolver("mumps")
        solver.solve(A, w_h.vector(), b)
    
        w_L = Displacement(t=0.0, nu=nu)

        result = {}
        result['hmax'] = mesh.hmax()
        result['hmin'] = mesh.hmin()
        result['w_l2'] = df.errornorm(w_L, w_h, norm_type='l2')/df.norm(w_h, norm_type='l2')
        result['hmax'] = mesh.hmax()
        result['dofs'] = kh.element.W.dim()

        return result

    def _runner(self, element, norms, expected_convergence_rates):
        """Given an element and norms, compare the computed convergence
        rate and the expected convergence rate and assert that the former
        is greater than the latter."""
        nxs = [16, 32, 64]
        # Doesn't make too much sense to run a convergence test 
        # with only one evaluation.
        assert(len(nxs) > 1)
        
        results = []
        for nx in nxs:
            result = self._analytical_lovadina(nx, element)
            results.append(result)

        for norm, expected_convergence_rate in zip(norms, expected_convergence_rates):
            hs = np.array([x['hmax'] for x in results])
            errors = np.array([x[norm] for x in results])
            
            actual_convergence_rate = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            err_msg = "Convergence rate in norm %s = %.3f, expected %.3f" % \
                        (norm, actual_convergence_rate, expected_convergence_rate)
  
            assert actual_convergence_rate >= expected_convergence_rate, err_msg
        
        #~ print actual_convergence_rate

    def test_CDG(self):
        """Run test for Continuous/Discontinuous element type"""
        norms = ['w_l2']
        expected_convergence_rates = [3.0]

        self._runner(CDGKirchhoff, norms, expected_convergence_rates)       


if __name__ == '__main__':
    unittest.main()
