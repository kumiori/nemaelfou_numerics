import os
import mshr
from dolfin import Measure, DirichletBC, Constant
from plate_lib import Experiment
from dolfin import dot, outer, norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate, Expression, Identity, project
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, as_backend_type, Vector
from plate_lib import ActuationOverNematicFoundation, ActuationOverNematicFoundationPneg
from string import Template
from dolfin import MPI
from dolfin import Mesh, MeshFunction, plot

from subprocess import Popen, PIPE, check_output
import matplotlib.pyplot as plt
import pandas as pd

import site
import sys
site.addsitedir('scripts')
import visuals

class ActuationTransition(Experiment):
	"""docstring for ActuationTransition"""
	def __init__(self, s = 1., template='', name=''):
		# self.s = s
		# self.case = p
		super(ActuationTransition, self).__init__(template, name)

	def add_userargs(self):
		self.parser.add_argument("--rad", type=float, default=1.)
		self.parser.add_argument("--rad2", type=float, default=.1)
		self.parser.add_argument("--s", type=float, default=1.)
		self.parser.add_argument("--savelag", type=int, default=1)
		self.parser.add_argument("--print", type=bool, default=False)
		self.parser.add_argument("--nematic", type=str, default='e1',
									help="R|Material regime,\n"
									 " e1\n"
									 " e3\n"
									 " tilt: (e1+e3)/sqrt(2)\n")
		self.parser.add_argument("--p", choices=['zero', 'neg'], type=str, default='zero',
									help="R|Material regime,\n"
									 " p = 'zero': coupled model\n"
									 " p = 'neg': vertical foundation\n")

	def parameters(self, args):
		self.fname = self.fname + '-p_{}'.format(args.p)
		parameters = {
			"material": {
				"nu": args.nu,
				"E": args.E,
				"gamma": 1.,
				"nematic": args.nematic,
				"p": args.p
			},
			"geometry": {
				"meshsize": args.h,
				"rad": args.rad,
				"rad2": args.rad2
			},
			"load": {
				"f0": args.f0,
				"s": args.s
			}
		}
		print('Parameters:')
		print(parameters)
		return parameters

	def create_mesh(self, parameters):
		from dolfin import Point, XDMFFile
		import hashlib 

		d={'rad': parameters['geometry']['rad'], 'rad2': parameters['geometry']['rad2'],
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
			mesh_template = open('scripts/coin.geo')

			src = Template(mesh_template.read())
			geofile = src.substitute(d)

			if MPI.rank(MPI.comm_world) == 0:
				with open("scripts/coin-%s"%self.signature+".geo", 'w') as f:
					f.write(geofile)

				# cmd = 'gmsh scripts/coin-{}.geo -2 -o meshes/coin-{}.msh'.format(self.signature, self.signature)
				# print(check_output([cmd], shell=True))  # run in shell mode in case you are not run in terminal

				# cmd = 'dolfin-convert -i gmsh meshes/coin-{}.msh meshes/coin-{}.xml'.format(self.signature, self.signature)
				# Popen([cmd], stdout=PIPE, shell=True).communicate()

			# mesh_xdmf = XDMFFile("meshes/%s-%s.xdmf"%(fname, self.signature))
			# mesh_xdmf.write(mesh)

		form_compiler_parameters = {
			"representation": "uflacs",
			"quadrature_degree": 2,
			"optimize": True,
			"cpp_optimize": True,
		}
		# import pdb; pdb.set_trace()

		mesh = Mesh("meshes/coin.xml")
		self.domains = MeshFunction('size_t',mesh,'meshes/coin_physical_region.xml')
		# dx = dx(subdomain_data=domains)
		plt.figure()
		plot(self.domains)
		# visuals.setspines0()
		plt.savefig(os.path.join(self.outdir,'domains-{}.pdf'
			.format(hashlib.md5(str(d).encode('utf-8')).hexdigest())))
		self.ds = Measure("exterior_facet", domain = mesh)
		self.dS = Measure("interior_facet", domain = mesh)
		self.dx = Measure("dx", metadata=form_compiler_parameters, subdomain_data=self.domains)

		return mesh

	def set_model(self):

		measures = [self.dx, self.ds, self.dS]
		# import pdb; pdb.set_trace()
		if self.parameters['material']['p'] == 'zero':
			from plate_lib import ActuationOverNematicFoundation
			actuation = ActuationOverNematicFoundation(self.z,
				self.mesh, self.parameters, measures)
		elif self.parameters['material']['p'] == 'neg':
			from plate_lib import ActuationOverNematicFoundationPneg
			actuation = ActuationOverNematicFoundationPneg(self.z,
				self.mesh, self.parameters, measures)
		else:
			raise NotImplementedError

		x = Expression(['x[0]', 'x[1]', '0.'], degree = 0)
		self.load = Expression('s', s = self.parameters['load']['s'], degree = 0)

		e1 = Constant([1, 0, 0])
		e3 = Constant([0, 0, 1])
		# import pdb; pdb.set_trace()

		if self.parameters["material"]["nematic"] == 'e1':
			n = e1
		elif self.parameters["material"]["nematic"] == 'e3':
			n = e3
		elif self.parameters["material"]["nematic"] == 'tilt':
			n = (e1-e3)/np.sqrt(2)
		else:
			raise NotImplementedError

		Qn = outer(n, n) - 1./3. * Identity(3)

		actuation.Q0n = self.load*Qn
		self.Q0 = actuation.Q0n
		actuation.define_variational_equation()

		self.F = actuation.F
		self.J = actuation.jacobian
		self.energy_mem = actuation.energy_mem
		self.energy_ben = actuation.energy_ben
		self.energy_nem = actuation.energy_nem
		self.work = actuation.work

	def define_boundary_conditions(self):
		V = self.z.function_space()

		bc_clamped = [DirichletBC(V.sub(1), Constant([0., 0.]), 'on_boundary'),
				DirichletBC(V.sub(2), Constant(0.), 'on_boundary')]
		bc_free = []
		bc_vert = [DirichletBC(V.sub(2), Constant(0.), 'on_boundary')]
		bc_horiz = [DirichletBC(V.sub(1), Constant([0., 0.]), 'on_boundary')]

		bcs = [bc_clamped, bc_free, bc_vert, bc_horiz]
		# return bcs[self.bc_no]
		# return bcs[2] # homogeneous vertical
		return bcs[0] # clamped (homog)

	def postprocess(self):
		print('z norm: ', self.z.vector().norm('l2'))

		M, u, v = self.z.split(True)
		Q = project(self.Q0, self.Fr)
		# print('z norm: {}'.format(norm(z, 'l2')))
		print('energy_mem = {}'.format(assemble(self.energy_mem(self.z)*self.dx)))
		print('energy_ben = {}'.format(assemble(self.energy_ben(self.z)*self.dx)))
		print('energy_nem = {}'.format(assemble(self.energy_nem(self.z))))
		print('work       = {}'.format(assemble(self.work(self.z))))
		# print('work weight= {}'.format(assemble(self.work_weight)))
		# print('work ratio = {}'.format(assemble(self.work_weight)/assemble(work)))
		# import pdb; pdb.set_trace()

		u.rename('u','u')
		v.rename('v','v')
		M.rename('M', 'M')
		Q.rename('Q0', 'Q0')

		self.file_mesh << self.mesh

		self.file_out.write(u, self.load.s)
		self.file_out.write(v, self.load.s)
		self.file_out.write(M, self.load.s)
		self.file_out.write(Q, self.load.s)
		# self.file_out.write(Q, self.bc_no)

		self.file_pproc.write_checkpoint(u, 'u', self.load.s, append = True)
		self.file_pproc.write_checkpoint(v, 'v', self.load.s, append = True)
		# self.file_pproc.write_checkpoint(M, 'M', 0, append = True)
		# self.file_pproc.write_checkpoint(Q, 'Q', 0, append, True)
		energy_mem = assemble(self.energy_mem(self.z)*self.dx)
		energy_ben = assemble(self.energy_ben(self.z)*self.dx)
		energy_nem = assemble(self.energy_nem(self.z))
		energy_tot = energy_nem+energy_ben+energy_mem
		max_abs_v = np.max(np.abs(v.vector()[:]))
		# tot_energy.append(assemble(self.work(self.z)))
		return {'load': self.load.s,
			'energy_nem': energy_nem,
			'energy_mem': energy_mem,
			'energy_ben': energy_ben,
			'max_abs_v': max_abs_v}

	def output(self):
		# import pdb; pdb.set_trace()
		pass

bcs = ['bc_clamped', 'bc_free', 'bc_vert', 'bc_horiz']

import numpy as np


data = []
problem = ActuationTransition(template='', name='coin')

# problem.load.s = 1.
print('Solving s={}'.format(problem.load.s))
problem.solve()
data.append(problem.postprocess())
problem.output()


evo_data = pd.DataFrame(data)
evo_data.to_json(os.path.join(problem.outdir, "time_data.json"))



