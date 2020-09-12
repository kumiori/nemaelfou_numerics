import os
import mshr
from dolfin import Measure, DirichletBC, Constant
from plate_lib import Experiment
from plate_lib import ActuationOverNematicFoundation
from dolfin import dot, outer, norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate, Expression, Identity
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, as_backend_type, Vector
from dolfin import project
import pandas as pd


class Actuation(Experiment):
	"""docstring for Actuation"""
	def __init__(self, bc_no = 0, template='', name=''):
		self.bc_no = bc_no
		super(Actuation, self).__init__(template, '{}-{}'.format(name, bc_no))

	def add_userargs(self):
		self.parser.add_argument("--rad", type=float, default=1.)
		self.parser.add_argument("--savelag", type=int, default=1)
		self.parser.add_argument("--print", type=bool, default=False)

	def parameters(self, args):
		parameters = {
			"material": {
				"nu": args.nu,
				"E": args.E,
				"gamma": 1.,
			},
			"geometry": {
				"meshsize": args.h,
				"rad": args.rad
			}
		}
		print('Parameters:')
		print(parameters)
		return parameters

	def create_mesh(self, parameters):
		from dolfin import Point, XDMFFile

		d={'rad': parameters['geometry']['rad'],
			'meshsize': parameters['geometry']['meshsize']}
		fname = 'circle'
		meshfile = "meshes/%s-%s.xml"%(fname, self.signature)

		if os.path.isfile(meshfile):
			# already existing mesh
			print("Meshfile %s exists"%meshfile)

		else:
			# mesh_template = open('scripts/sandwich_pert.template.geo' )
			print("Create meshfile, meshsize {}".format(parameters['geometry']['meshsize']))
			nel = int(parameters['geometry']['rad']/parameters['geometry']['meshsize'])
			geom = mshr.Circle(Point(0., 0.), parameters['geometry']['rad'])
			mesh = mshr.generate_mesh(geom, nel)

			mesh_xdmf = XDMFFile("meshes/%s-%s.xdmf"%(fname, self.signature))
			mesh_xdmf.write(mesh)

		form_compiler_parameters = {
			"representation": "uflacs",
			"quadrature_degree": 2,
			"optimize": True,
			"cpp_optimize": True,
		}

		self.ds = Measure("exterior_facet", domain = mesh)
		self.dS = Measure("interior_facet", domain = mesh)
		self.dx = Measure("dx", metadata=form_compiler_parameters, domain=mesh)

		return mesh

	def set_model(self):

		measures = [self.dx, self.ds, self.dS]
		actuation = ActuationOverNematicFoundation(self.z,
			self.mesh, self.parameters, measures)
		x = Expression(['x[0]', 'x[1]', '0.'], degree = 0)
		actuation.Q0n = 1/dot(x, x)*outer(x, x) - 1/3*Identity(3)
		actuation.define_variational_equation()
		self.Q0 = project(actuation.Q0n, self.Fr)
		# import pdb; pdb.set_trace()
		self.Q0.rename('Q0', 'Q0')

		self.F = actuation.F
		self.J = actuation.jacobian
		self.energy_mem = actuation.energy_mem
		self.energy_ben = actuation.energy_ben
		self.energy_nem = actuation.energy_nem
		self.work = actuation.work

		# import pdb; pdb.set_trace()

	def define_boundary_conditions(self):
		V = self.z.function_space()

		bc_clamped = [DirichletBC(V.sub(1), Constant([0., 0.]), 'on_boundary'),
				DirichletBC(V.sub(2), Constant(0.), 'on_boundary')]
		bc_free = []
		bc_vert = [DirichletBC(V.sub(2), Constant(0.), 'on_boundary')]
		bc_horiz = [DirichletBC(V.sub(1), Constant([0., 0.]), 'on_boundary')]

		bcs = [bc_clamped, bc_free, bc_vert, bc_horiz]
		return bcs[self.bc_no]

	def postprocess(self):
		print('z norm: ', self.z.vector().norm('l2'))
		M, u, v = self.z.split(True)
		# print('z norm: {}'.format(norm(z, 'l2')))

		_energy_mem = assemble(self.energy_mem(self.z)*self.dx)
		_energy_ben = assemble(self.energy_ben(self.z)*self.dx)
		_energy_nem = assemble(self.energy_nem(self.z))
		_work = assemble(self.work(self.z))

		print('energy_mem = {}'.format(_energy_mem))
		print('energy_ben = {}'.format(_energy_ben))
		print('energy_nem = {}'.format(_energy_nem))
		print('work       = {}'.format(_work))

		u.rename('u','u')
		v.rename('v','v')
		M.rename('M', 'M')

		self.file_mesh << self.mesh

		self.file_out.write(u, 3.)
		self.file_out.write(v, 3.)
		self.file_out.write(M, 3.)
		self.file_out.write(self.Q0, 3.)

		self.file_pproc.write_checkpoint(u, 'u', self.bc_no, append = True)
		self.file_pproc.write_checkpoint(v, 'v', self.bc_no, append = True)
		# self.file_pproc.write_checkpoint(M, 'M', 0, append = True)
		# self.file_pproc.write_checkpoint(Q, 'Q', 0, append, True)
		print('DEBUG: saving ts {}'.format(self.bc_no))

		energies = {'energy_nem': _energy_nem, 'energy_ben': _energy_ben, 'energy_mem': _energy_mem, 'work': _work}

		return energies

	def output(self):
		# import pdb; pdb.set_trace()
		pass

bcs = ['bc_clamped', 'bc_free', 'bc_vert', 'bc_horiz']
energies = []

for bc_no in range(len(bcs)):
	print(bcs[bc_no])
	problem = Actuation(bc_no = bc_no, template='', name='disclination')

	problem.solve()
	ener_bcs = problem.postprocess()
	energies.append(ener_bcs)

energies_pd = pd.DataFrame(energies)
energies_pd.to_json(os.path.join(problem.outdir, "energy_data.json"))


