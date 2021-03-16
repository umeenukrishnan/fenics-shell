import sys
sys.path.append("../..")
import dolfin as df
import ufl

class KirchhoffPlate(object):
	def __init__(self, element, E, nu, t, bcs=[]):
		self.element = element
		self.mesh = element.mesh
		self.W = self.element.W
		
		self.D = df.Constant(E*t**3/(12.0*(1.0 - nu**2)))
		self.nu = df.Constant(nu)
		
		self.bcs = bcs
	
	def physical_forms(self):
		w = self.element._w
		v = self.element._v
		
		from dolfin import inner, grad, tr, Identity, dx
		#~ Curvature tensor
		k = lambda dis: grad(grad(dis))
		#~ Bending moment tensor
		M = lambda kap: self.D*((1.0 - self.nu)*kap + self.nu*tr(kap)*Identity(2))
		
		a_p = inner(M(k(w)), k(v))*dx
		#TODO Write a_cd, M, and k outside the function?
		a_cd = self.element.continuous_discontinuous_form(M(k(w)), M(k(v)))
		
		return a_p, a_cd
	
	def construct_form(self, a_a=ufl.constantvalue.Zero()):
		a_p = self.physical_forms()[0]
		a_cd = self.physical_forms()[1]
		
		self.a = a_p + a_cd + a_a
		
		return self.a

if __name__ == "__main__":
	E = 10.92E06
	nu = 0.3
	t = 1.0
	w_onb = df.Constant((0.0))
	phi_onb = df.Constant((0.0))
	
	mesh = df.UnitSquareMesh(16,16)
	from kirchhoff_plate_elements import CDGKirchhoff
	element = CDGKirchhoff(mesh, phi_onb)
	
	def all_boundary(x, on_boundary):
		return on_boundary
	
	bc_W = df.DirichletBC(element.W, w_onb, all_boundary)
	bcs = [bc_W]
	
	kh = KirchhoffPlate(element, bcs=bcs, E=E, nu=nu, t=t)
	
	from analytical_lovadina import Loading, Displacement
	f = Loading(E=E, nu=nu)
	from dolfin import dx
	a_a = -f*kh.element._v*dx
	
	a = kh.construct_form(a_a=a_a)
	w_h = df.Function(kh.W)
	problem = df.LinearVariationalProblem(df.lhs(kh.a), df.rhs(kh.a), w_h, bcs=kh.bcs)
	solver = df.LinearVariationalSolver(problem)
	solver.solve()
	df.plot(w_h, interactive=True)
	
	w_L = Displacement(t=0.0, nu=nu)
	
	result = {}
	result['hmax'] = mesh.hmax()
	result['hmin'] = mesh.hmin()
	result['err_l2'] = df.errornorm(w_L, w_h, norm_type='l2')/df.norm(w_h, norm_type='l2')
	#~ TODO test the convergence in the energy norm
	result['dofs'] = kh.element.W.dim()
	from numpy import abs
	result['perr_center'] = 100*abs(w_L(0.5,0.5)-w_h(0.5,0.5))/(abs(w_L(0.5,0.5)))
	
	print result
	
