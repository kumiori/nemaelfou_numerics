import os
# import mshr
from dolfin import Measure, DirichletBC, Constant, File, list_timings
from plate_lib import Experiment
from dolfin import dot, outer, norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate, Expression, Identity, project
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, as_backend_type, Vector
from plate_lib import Relaxed
from string import Template
from dolfin import MPI
from dolfin import Mesh, MeshFunction, plot, info
import json
import pdb
from subprocess import Popen, PIPE, check_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import site
import sys
site.addsitedir('scripts')
import visuals
import hashlib
from pathlib import Path

class RelaxedExperiment(Experiment):
	"""docstring for ActuationTransition"""
	def __init__(self, s = 1., template='', name='', p='zero', parameters = {}):
		# self.s = s
		self.p = p
		# self.counter=0.
		super(RelaxedExperiment, self).__init__(template, name, parameters)

	def add_userargs(self):
		self.parser.add_argument("--rad", type=float, default=1.)
		self.parser.add_argument("--rad2", type=float, default=.1)
		self.parser.add_argument("--savelag", type=int, default=1)
		self.parser.add_argument("--print", type=bool, default=False)
		self.parser.add_argument("--p", choices=['zero', 'neg'], type=str, default='zero',
									help="R|Material regime,\n"
									 " p = 'zero': coupled model\n"
									 " p = 'neg': vertical foundation\n")

	def parameters(self, args):
		# self.fname = self.fname + '-p_{}'.format(args.p)
		# import pdb; pdb.set_trace()
		# parameters = {
		# 	"material": {
		# 		"nu": args.nu,
		# 		"E": args.E,
		# 		"gamma": 1.,
		# 		"p": self.p
		# 	},
		# 	"geometry": {
		# 		"meshsize": args.h,
		# 		"rad": args.rad,
		# 		"rad2": args.rad2
		# 	},
		# 	"load": {
		# 		"f0": args.f0
		# 	},
		# 	"experiment": {
		# 		"debug": args.debug
		# 	}
		# }

		with open('parameters/parameters.yaml') as f:
		    parameters = yaml.load(f, Loader=yaml.FullLoader)

		print('Parameters:')
		print(parameters)
		return parameters

	def create_output(self, fname):
		args = self.args
		parameters = self.parameters
		self.signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
		print('Signature is: ', self.signature)
		regime = self.p
		if args.outdir == None:
			outdir = "output/{:s}-{}{}".format(fname, self.signature, args.postfix)
		else:
			outdir = args.outdir

		self.outdir = outdir
		Path(outdir).mkdir(parents=True, exist_ok=True)
		print('Outdir is: ', outdir)
		print('Output in: ', os.path.join(outdir, fname+'p-'+regime+'.xdmf'))
		print('P-proc in: ', os.path.join(outdir, fname+'p-'+regime+'_pproc.xdmf'))

		file_results = XDMFFile(os.path.join(outdir, fname+'p-'+regime+'.xdmf'))
		file_results.parameters["flush_output"] = True
		file_results.parameters["functions_share_mesh"] = True

		file_pproc = XDMFFile(os.path.join(outdir, fname+'p-'+regime+'_pproc.xdmf'))
		file_pproc.parameters["flush_output"] = True
		file_pproc.parameters["functions_share_mesh"] = True

		file_mesh = File(os.path.join(outdir, fname+'p-'+regime+"_mesh.xml"))

		with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
			json.dump(parameters, f)
		print('DEBUG: pproc file', os.path.join(outdir, fname+'p-'+regime+'_pproc.xdmf'))
		return file_results, file_pproc, file_mesh

	def create_mesh(self, parameters):
		from dolfin import Point, XDMFFile
		import hashlib 

		d={'rad': parameters['geometry']['rad'], 'h': parameters['geometry']['meshsize']}
		# fname = 'coinforall'
		# fname = 'coin'
		fname = 'coinforall_small'
		meshfile = "meshes/%s-%s.xml"%(fname, self.signature)

		if os.path.isfile(meshfile):
			print("Meshfile %s exists"%meshfile)

		else:
			print("Create meshfile, meshsize {}".format(parameters['geometry']['meshsize']))
			nel = int(parameters['geometry']['rad']/parameters['geometry']['meshsize'])

			# geom = mshr.Circle(Point(0., 0.), parameters['geometry']['rad'])
			# mesh = mshr.generate_mesh(geom, nel)
			mesh_template = open('scripts/coin_template.geo')

			src = Template(mesh_template.read())
			geofile = src.substitute(d)

			if MPI.rank(MPI.comm_world) == 0:
				with open("scripts/coin-%s"%self.signature+".geo", 'w') as f:
					f.write(geofile)

		form_compiler_parameters = {
			"representation": "uflacs",
			"quadrature_degree": 2,
			"optimize": True,
			"cpp_optimize": True,
		}

		mesh = Mesh("meshes/{}.xml".format(fname))
		self.domains = MeshFunction('size_t',mesh,'meshes/{}_physical_region.xml'.format(fname))
		self.ds = Measure("exterior_facet", domain = mesh)
		self.dS = Measure("interior_facet", domain = mesh)
		self.dx = Measure("dx", metadata=form_compiler_parameters, subdomain_data=self.domains)
		# self.dx = Measure("dx", subdomain_data=self.domains)
		plt.colorbar(plot(self.domains))
		plt.savefig('domains.pdf')
		return mesh

	def set_model(self):

		measures = [self.dx, self.ds, self.dS]
		# import pdb; pdb.set_trace()
		# regime = self.parameters['material']['p']
		regime = self.p
		print('Regime p={}'.format(regime))

		from plate_lib import Relaxed
		# pdb.set_trace()
		relaxed = Relaxed(self.z,
			self.mesh, self.parameters, measures)

		# pdb.set_trace()
		relaxed.define_variational_equation()

		self.emin = relaxed.emin
		self.emax = relaxed.emax
		self.phaseField = relaxed.phaseField
		self.G = relaxed.G
		self.lagrangian = relaxed.lagrangian
		self.F = relaxed.F
		self.J = relaxed.jacobian
		self.energy_mem = relaxed.energy_mem
		self.energy_ben = relaxed.energy_ben
		self.energy_nem = relaxed.energy_nem
		self.work = relaxed.work

	def set_problem(self):
		from problem import PlateProblemSNES
		return PlateProblemSNES(self.lagrangian,
								self.z, self.bcs,
								residual = self.F, jacobian = self.J)

	def set_solver(self):
		solver = PETScSNESSolver()
		snes = solver.snes()
		snes.setType("newtontr")
		ksp = snes.getKSP()
		ksp.setType("preonly")
		pc = ksp.getPC()
		pc.setType("lu")
		snes_solver_parameters = {
				# "linear_solver": "lu",
				"linear_solver": "umfpack",
				# "linear_solver": "mumps",
				"maximum_iterations": 300,
				"report": True,
				# "monitor": True,
				"line_search": "basic",
				"method": "newtonls",
				"absolute_tolerance": 1e-5,
				"relative_tolerance": 1e-5,
				"solution_tolerance": 1e-5}

		solver.parameters.update(snes_solver_parameters)
		info(solver.parameters, True)
		# import pdb; pdb.set_trace()
		return solver

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
		# return bcs[0] # clamped (homog)
		return bcs[1] # free

	def postprocess(self):
		print('z norm: ', self.z.vector().norm('l2'))
		# self.counter = self.n.theta
		counter = self.counter
		regime = self.p
		# print('Regime p={}'.format(regime))
		if regime == 'zero':
			counter = 0
		else:
			counter = 1

		print('pproc counter:{}'.format(counter))
		M, u, v = self.z.split(True)
		# pdb.set_trace()
		print('u norm = {}'.format(u.vector().norm('l2')))
		print('u n.h1 = {}'.format(norm(u,'h1')))
		print('v norm = {}'.format(v.vector().norm('l2')))
		print('M norm = {}'.format(M.vector().norm('l2')))

		em = self.emin(u, v)
		eM = self.emax(u, v)

		phase = self.phaseField(em, eM)
		phv = project(phase, self.L2)
		plt.clf()
		plt.colorbar(plot(phv))
		plt.savefig(os.path.join(self.outdir, 'phases.pdf'))


		em = project(em, self.L2)
		eM = project(eM, self.L2)

		em.rename('em', 'em')
		eM.rename('eM', 'eM')
		self.file_out.write(em, counter)
		self.file_out.write(eM, counter)

		plt.clf()
		plt.scatter(em.vector()[:], eM.vector()[:])
		plt.axvline(-1./3.)

		es = np.linspace(0, -1., 3)
		plt.plot(es, -2.*es)
		plt.plot(es, -es/2.)
		plt.plot(es, -es/2.+1./2.)
		plt.xlabel('emin')
		plt.ylabel('emax')
		plt.savefig(os.path.join(self.outdir, 'phase.pdf'))

		# print('work weight= {}'.format(assemble(self.work_weight)))
		# print('work ratio = {}'.format(assemble(self.work_weight)/assemble(work)))
		# import pdb; pdb.set_trace()

		u.rename('u','u')
		v.rename('v','v')
		M.rename('M', 'M')

		self.file_mesh << self.mesh

		self.file_out.write(u, counter)
		self.file_out.write(v, counter)
		self.file_out.write(M, counter)

		self.file_pproc.write_checkpoint(u, 'u', counter, append = True)
		self.file_pproc.write_checkpoint(v, 'v', counter, append = True)
		# self.file_pproc.write_checkpoint(M, 'M', 0, append = True)
		# import pdb; pdb.set_trace()
		energy_fou = assemble(self.G([u, v]))
		energy_mem = assemble(self.energy_mem(self.z)*self.dx)
		energy_ben = assemble(self.energy_ben(self.z)*self.dx)
		energy_nem = assemble(self.energy_nem(self.z))
		work = assemble(self.work(self.z))
		energy_tot = energy_nem+energy_ben+energy_mem-work
		max_abs_v = np.max(np.abs(v.vector()[:]))
		max_v = np.max(v.vector()[:]) 
		min_v = np.min(v.vector()[:])
		# max_u = np.min(u.vector()[:])
		# tot_energy.append(assemble(self.work(self.z)))

		print('energy_mem = {}'.format(assemble(self.energy_mem(self.z)*self.dx)))
		print('energy_ben = {}'.format(assemble(self.energy_ben(self.z)*self.dx)))
		print('energy_fou = {}'.format(energy_fou))
		print('work       = {}'.format(assemble(self.work(self.z))))

		return {'load': counter,
			'energy_fou': energy_fou,
			'energy_mem': energy_mem,
			'energy_ben': energy_ben,
			'work': work,
			'energy_tot': energy_ben+energy_mem+energy_fou-work,
			'max_abs_v': max_abs_v,
			# 'max_u': max_u,
			'max_v': max_v,
			'min_v': min_v,
			'min_u1': min(u.sub(0).vector()[:]),
			'min_u2': min(u.sub(1).vector()[:]),
			'max_u1': max(u.sub(0).vector()[:]),
			'max_u2': max(u.sub(1).vector()[:]),
			}

	def output(self):
		pass

bcs = ['bc_clamped', 'bc_free', 'bc_vert', 'bc_horiz']

import numpy as np
import yaml

with open('parameters/parameters.yaml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

data = []
problem = RelaxedExperiment(template='', name='relaxed', p='zero', parameters = parameters)
problem.counter = 0

# from problem import PlateProblemSNES
problem.solve()
data.append(problem.postprocess())
problem.output()


evo_data = pd.DataFrame(data)
print(evo_data)
evo_data.to_json(os.path.join(problem.outdir, "time_data.json"))


from dolfin import TimingClear, TimingType
from dolfin import MPI

# list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])
# t = list_timings(TimingClear.keep,
#             [TimingType.wall, TimingType.user, TimingType.system])
# Use different MPI reductions
# t_sum = MPI.sum(MPI.comm_world, t[0])
# t_min = MPI.min(MPI.comm_world, t[1])
# t_max = MPI.max(MPI.comm_world, t[2])
# t_avg = MPI.avg(MPI.comm_world, t[3])

# # Print aggregate timings to screen
# print('\n'+t_sum.str(True))
# print('\n'+t_min.str(True))
# print('\n'+t_max.str(True))
# print('\n'+t_avg.str(True))

# # Store to XML file on rank 0
# if MPI.rank(MPI.comm_world) == 0:
#     f = File(MPI.comm_self, os.path.join(problem.outdir, "timings_aggregate.xml"))
#     # f << t_sum
#     # f << t_min
#     # f << t_max
#     f << t_avg

# dump_timings_to_xml(os.path.join(problem.outdir, "timings_avg_min_max.xml"), TimingClear.clear)


