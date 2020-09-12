import os
import mshr
from dolfin import Measure, DirichletBC, Constant
from plate_lib import Experiment
from plate_lib import ActuationOverNematicFoundation
from dolfin import dot, outer, norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate, Expression, Identity
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, as_backend_type, Vector
from dolfin import Mesh, MeshFunction, Measure, plot, as_matrix, assemble, inner, split
from dolfin import Function
import numpy as np
from string import Template
from dolfin import MPI
from subprocess import Popen, PIPE, check_output
import matplotlib.pyplot as plt
import pandas as pd

import site
import sys

site.addsitedir('scripts')

import visuals

class ActuationShear(ActuationOverNematicFoundation):
	"""docstring for ActuationShear"""
	def __init__(self, z, mesh, parameters, measures):
		super(ActuationShear, self).__init__(z, mesh, parameters, measures)

	def work(self, z):

		M, u, v = split(z)
		force = self.force()

		Q1n = self.Q1n
		Q2n = self.Q2n
		Q3n = self.Q3n
		Q4n = self.Q4n

		return -inner(Q1n, self.C(u, v))*self.dx(1)+inner(Q2n, self.C(u, v))*self.dx(2)+\
			inner(Q3n, self.C(u, v))*self.dx(3)+inner(Q4n, self.C(u, v))*self.dx(4)  - force*v*self.dx

	def linear_term(self, u_, v_):
		Q1n = self.Q1n
		Q2n = self.Q2n
		Q3n = self.Q3n
		Q4n = self.Q4n
		# import pdb; pdb.set_trace()
		return - inner(Q1n, self.C(u_, v_))*self.dx(1) - inner(Q2n, self.C(u_, v_))*self.dx(2)+\
				- inner(Q3n, self.C(u_, v_))*self.dx(3) - inner(Q4n, self.C(u_, v_))*self.dx(4) + \
				- self.force()*v_*self.dx

class Actuation(Experiment):
	"""docstring for Actuation"""
	def __init__(self, bc_no = 0, template='', name=''):
		self.bc_no = bc_no
		super(Actuation, self).__init__(template, '{}-{}'.format(name, bc_no))

	def add_userargs(self):
		self.parser.add_argument("--a", type=float, default=0., help="-1/3<= a <= 0")
		self.parser.add_argument("--c", type=float, default=0., help="0<= c <= 2/3")
		self.parser.add_argument("--rad", type=float, default=1.)
		self.parser.add_argument("--savelag", type=int, default=1)
		self.parser.add_argument("--print", type=bool, default=False)

	def parameters(self, args):

		_ta = args.a + 1/3
		_tc = args.c + 1/3
		_tb = args.a-args.c + 1/3

		print("T=",np.sqrt(_tc/_ta))
		parameters = {
			"material": {
				"nu": args.nu,
				"E": args.E,
				"gamma": 1.,
				"a": args.a,
				"c": args.c,
				"b": args.a-args.c
			},
			"geometry": {
				"meshsize": args.h,
				"rad": args.rad,
				"T": np.sqrt(_tc/_ta)
			}
		}
		print('Parameters:')
		print(parameters)
		return parameters

	def create_mesh(self, parameters):
		from dolfin import Point, XDMFFile
		import hashlib 

		d={'T': parameters['geometry']['T'], 'rad': parameters['geometry']['rad'],
			'meshsize': parameters['geometry']['meshsize']}
		fname = 'Qshear'
		meshfile = "meshes/%s-%s.xml"%(fname, self.signature)

		if os.path.isfile(meshfile):
			# already existing mesh
			print("Meshfile %s exists"%meshfile)

		else:
			mesh_template = open('scripts/Q0shear.geo')
			print("Create meshfile, meshsize {}".format(parameters['geometry']['meshsize']))
			nel = int(parameters['geometry']['rad']/parameters['geometry']['meshsize'])
			src = Template(mesh_template.read())
			geofile = src.substitute(d)

			if MPI.rank(MPI.comm_world) == 0:
				with open("scripts/Qshear-%s"%self.signature+".geo", 'w') as f:
					f.write(geofile)

					# pdb.set_trace()
				cmd = 'gmsh scripts/Qshear-{}.geo -2 -o meshes/Qshear-{}.msh'.format(self.signature, self.signature)
				print(check_output([cmd], shell=True))  # run in shell mode in case you are not run in terminal

				cmd = 'dolfin-convert -i gmsh meshes/Qshear-{}.msh meshes/Qshear-{}.xml'.format(self.signature, self.signature)
				Popen([cmd], stdout=PIPE, shell=True).communicate()


		form_compiler_parameters = {
			"representation": "uflacs",
			"quadrature_degree": 2,
			"optimize": True,
			"cpp_optimize": True,
		}


		mesh = Mesh("meshes/Qshear-%s.xml"%self.signature)
		self.domains = MeshFunction('size_t',mesh,'meshes/Qshear-%s_physical_region.xml'%self.signature)
		# dx = dx(subdomain_data=domains)
		plt.figure()
		plot(self.domains)
		visuals.setspines0()
		plt.savefig(os.path.join(self.outdir,'domains-{}.pdf'
			.format(hashlib.md5(str(d).encode('utf-8')).hexdigest())))
		self.ds = Measure("exterior_facet", domain = mesh)
		self.dS = Measure("interior_facet", domain = mesh)
		self.dx = Measure("dx", metadata=form_compiler_parameters, subdomain_data=self.domains)

		return mesh

	def set_model(self):

		measures = [self.dx, self.ds, self.dS]
		actuation = ActuationShear(self.z,
			self.mesh, self.parameters, measures)

		_a = self.parameters['material']['a']
		_b = self.parameters['material']['b']
		_c = self.parameters['material']['c']

		_ta = _a + 1/3
		_tb = _b + 1/3
		_tc = _a-_c + 1/3

		Qn = [as_matrix([[0,0,0],
						[0,0,0],
						[0,0,0]])]*4

		Qn[0] = as_matrix([[_a, -np.sqrt(_ta*_tb), np.sqrt(_ta*_tc)],
						   [np.sqrt(_ta*_tb), _b, -np.sqrt(_tb*_tc)],
						   [np.sqrt(_ta*_tc), np.sqrt(_tb*_tc), _c]])

		Qn[1] = as_matrix([[_a, np.sqrt(_ta*_tb), np.sqrt(_ta*_tc)],
						   [np.sqrt(_ta*_tb), _b, np.sqrt(_tb*_tc)],
						   [np.sqrt(_ta*_tc), np.sqrt(_tb*_tc), _c]])

		Qn[2] = as_matrix([[_a, -np.sqrt(_ta*_tb), - np.sqrt(_ta*_tc)],
						   [np.sqrt(_ta*_tb), _b, np.sqrt(_tb*_tc)],
						   [np.sqrt(_ta*_tc), np.sqrt(_tb*_tc), _c]])

		Qn[3] = as_matrix([[_a, np.sqrt(_ta*_tb), - np.sqrt(_ta*_tc)],
						   [np.sqrt(_ta*_tb), _b, - np.sqrt(_tb*_tc)],
						   [np.sqrt(_ta*_tc), np.sqrt(_tb*_tc), _c]])

		# import pdb; pdb.set_trace()

		actuation.Q1n = Qn[0]
		actuation.Q2n = Qn[1]
		actuation.Q3n = Qn[2]
		actuation.Q4n = Qn[3]

		self.Q = Qn

		actuation.define_variational_equation()


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
		# print('work weight= {}'.format(assemble(self.work_weight)))
		# print('work ratio = {}'.format(assemble(self.work_weight)/assemble(work)))
		# import pdb; pdb.set_trace()

		u.rename('u','u')
		v.rename('v','v')
		M.rename('M', 'M')

		# Q = self.Q

		self.file_mesh << self.mesh

		self.file_out.write(u, self.bc_no)
		self.file_out.write(v, self.bc_no)
		self.file_out.write(M, self.bc_no)

		print("DEBUG: written u, v, M, #", self.bc_no)

		self.file_pproc.write_checkpoint(u, 'u', self.bc_no, append = True)
		self.file_pproc.write_checkpoint(v, 'v', self.bc_no, append = True)
		# self.file_pproc.write_checkpoint(M, 'M', self.bc_no, append = True)
		# self.file_pproc.write_checkpoint(Q, 'Q', 0, append, True)

		energies = {'energy_nem': _energy_nem, 'energy_ben': _energy_ben, 'energy_mem': _energy_mem, 'work': _work}

		return energies

	def output(self):
		# import pdb; pdb.set_trace()
		pass


bcs = ['bc_clamped', 'bc_free', 'bc_vert', 'bc_horiz']
energies = []

for bc_no in range(len(bcs)):
	print(bcs[bc_no])
	problem = Actuation(bc_no = bc_no, template='', name='shear')

	problem.solve()
	ener_bcs = problem.postprocess()
	energies.append(ener_bcs)

energies_pd = pd.DataFrame(energies)
energies_pd.to_json(os.path.join(problem.outdir, "energy_data.json"))

