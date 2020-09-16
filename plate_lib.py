from __future__ import print_function
from dolfin import *
from ufl.operators import And
# import mshr
import numpy as np
import matplotlib.pyplot as plt
import ufl
from fenics import *
from string import Template
import hashlib 
from subprocess import Popen, PIPE, check_output
import os
from pathlib import Path
import json

import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
print(PETSc.Options().getAll())

import argparse
from urllib.parse import unquote
from time import sleep

# class PlateOverNematicFoundationPzero
# class PlateOverNematicFoundationPneg
# class RelaxationOverNematicFoundation
# class ActuationOverNematicFoundation

class PlateProblem(NonlinearProblem):
	def __init__(self, z, residual, jacobian, bcs):
		NonlinearProblem.__init__(self)
		self.z = z
		self.bcs = bcs
		self.residual = residual
		self.jacobian = jacobian

	def F(self, b, x):
		assemble(self.residual, tensor=b)
		# import pdb; pdb.set_trace()
		[bc.apply(b, x) for bc in self.bcs]

	def J(self, A, x):
		assemble(self.jacobian, tensor=A)
		# import pdb; pdb.set_trace()
		[bc.apply(A) for bc in self.bcs]

class ActuationOverNematicFoundation(object):
	"""docstring for PlateOverNematicFoundation"""
	def __init__(self, z, mesh, parameters, measures):
		super(ActuationOverNematicFoundation, self).__init__()
		self.z = z
		self.parameters = parameters
		self.mesh = mesh

		self.dx = measures[0]
		self.ds = measures[1]
		self.dS = measures[2]

		self.V = z.function_space()
		self.M_, self.u_, self.v_ = TestFunctions(self.V)
		self.dM, self.du, self.dv = TrialFunctions(self.V)

		E = parameters['material']['E']
		nu = parameters['material']['nu']

		self.mu = E/(2.0*(1.0 + nu))
		self.lmbda = E*nu/(1.0 - nu**2)
		self.f0 = parameters['load']['f0']

		# self.Q0n = self.Q0n()

	# @staticmethod
	def eps(self, u):
		return sym(grad(u))

	def M_to_voigt(self, M):
		return as_vector([M[0,0],M[1,1],M[1,0]])

	def M_from_voigt(self, m):
		return as_matrix([[m[0],m[2]],[m[2],m[1]]])

	def a(self, M, tau):
		Mm = self.M_from_voigt(self.S*self.M_to_voigt(M))
		return inner(Mm, tau)
		 # * dx

	# @staticmethod
	def a_m(self, u, v):
		lmbda = self.lmbda
		mu = self.mu
		return (inner(self.eps(u), self.eps(v)) + lmbda/(2.*mu)*tr(self.eps(u))*tr(self.eps(v)))
		# *dx

	def b(self, M, v):
		n = FacetNormal(self.mesh)
		dx = self.dx
		ds = self.ds
		dS = self.dS
		Mnn = dot(dot(M, n), n)
		return inner(M, grad(grad(v))) * dx \
			- Mnn('+') * jump(grad(v), n) * dS \
			- Mnn * dot(grad(v), n) * ds #\

	@staticmethod
	def C(u, v):
		return as_matrix([[-1/3*v,     0.,         1./2.*u[0]],\
						  [0,          -1./3.*v,   1./2.*u[1]],\
						  [1./2.*u[0], 1./2.*u[1], 2./3.*v]])

	@staticmethod
	def M_to_voigt(M):
		return as_vector([M[0,0],M[1,1],M[1,0]])

	@staticmethod
	def M_from_voigt(m):
		return as_matrix([[m[0],m[2]],[m[2],m[1]]])

	def linear_term(self, u_, v_):
		# _X = VectorFunctionSpace(self.mesh, 'CG', 1, dim=3)
		# _x = Function(_X)
		# x = Expression(['x[0]', 'x[1]', '0.'], degree = 0)
		# import pdb; pdb.set_trace()
		Q0n = self.Q0n
		return inner(Q0n, self.C(u_, v_))*self.dx + self.force()*v_*self.dx(2)

	def energy_nem(self, z):
		M, u, v = split(z)
		lmbda = self.lmbda
		mu = self.mu
		return  1./2.* inner(self.C(u, v), self.C(u, v))*dx + ((3.*lmbda+2*mu)/(3.*mu))*v*v*dx

	def energy_mem(self, z):
		M, u, v = split(z)
		# M, u, v = split(self.z)

		return 1./2.* self.a_m(u, u)

	def energy_ben(self, z):
		# M, u, v = split(self.z)
		M, u, v = split(z)

		return 1./2.* self.a(M, M)

	def force(self):
		return Constant(self.f0)

	def work(self, z):
		# M, u, v = split(self.z)
		M, u, v = split(z)
		# Q0n = 1/dot(x, x)**(.5)*outer(x, x) - 1/3*Identity(3)
		Q0n = self.Q0n
		force = self.force()
		return -inner(Q0n, self.C(u, v))*self.dx - force*v*self.dx

	def define_variational_equation(self):
		lmbda = self.lmbda
		mu = self.mu
		self.A = 1./3.*as_matrix([[lmbda/(lmbda+2*mu)+1,lmbda/(lmbda+2*mu),0], \
							 [lmbda/(lmbda+2*mu),lmbda/(lmbda+2*mu)+1,1./3.], \
							 [0,1./3.,1./3.]])
		self.S = inv(self.A)

		dx = self.dx
		ds = self.ds
		dS = self.dS

		z = self.z # Function
		M, u, v = split(z) # Functions
		M_, u_, v_ = self.M_, self.u_, self.v_ # TestFunctions
		dM, du, dv = self.dM, self.du, self.dv # TrialFunctions

		f = inner(self.C(u, v), self.C(u_, v_))*dx + ((3.*lmbda+2*mu)/(3.*mu))*v*v_*dx
		B = - self.a_m(u, u_)*dx  + self.a(M, M_)*dx  - self.b(M_, v) - f - self.b(M, v_)
		jacobian = - self.a_m(du, u_)*dx  + self.a(dM, M_)*dx \
			- inner(self.C(du, dv), self.C(u_, v_))*dx \
			- ((3.*lmbda+2*mu)/(3.*mu))*dv*v_*dx \
			- self.b(M_, dv) - self.b(dM, v_)

		# L = inner(Q0n, C(u_, v_))*dx
		L = self.linear_term(u_, v_)
		F = B - L

		self.F = F
		self.jacobian = jacobian

		return F, jacobian

	def Q0n(self):
		# return as_matrix([[0, 0., 0],\
		# 				  [0, 0, 0],\
		# 				  [0, 0, 0]])
		return Constant([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

class ActuationOverNematicFoundationPneg(ActuationOverNematicFoundation):
	"""docstring for ActuationOverNematicFoundationPneg"""
	def __init__(self, z, mesh, parameters, measures):
		super(ActuationOverNematicFoundationPneg, self).__init__(z, mesh, parameters, measures)

	@staticmethod
	def C(u, v):
		return as_matrix([[-1/3*v, 0., 0],  \
						  [0, -1./3.*v, 0], \
						  [0, 0, 2./3.*v]])


class Experiment(object):
	"""docstring for Experiment"""
	def __init__(self, template='', name=''):
		super(Experiment, self).__init__()
		self.template = template
		self.fname = name
		self.args = self.parse_args()
		self.parameters = self.parameters(self.args)
		self.file_out, self.file_pproc, self.file_mesh = self.create_output(self.fname)

		# overaload needed
		self.mesh = self.create_mesh(self.parameters)

		self.define_variational_formulation(mesh)
		self.bcs = self.define_boundary_conditions()
		self.set_model()
		# import pdb; pdb.set_trace()

		# self.problem = PlateProblem(self.z, self.F, self.J, self.bcs)

	def define_variational_formulation(self, mesh):
		mesh = self.mesh

		r = 2
		SREG = FiniteElement('HHJ', mesh.ufl_cell(), r)
		H2 = FiniteElement('CG', mesh.ufl_cell(), r + 1)
		H1 = VectorElement('CG', mesh.ufl_cell(), 1, dim = 2)

		self.L2 = FunctionSpace(mesh, 'DG', 0)
		self.V = FunctionSpace(mesh, SREG * H2)
		# V2 = FunctionSpace(mesh, SREG * H2 * H1)
		# import pdb; pdb.set_trace()

		self.V1 = FunctionSpace(mesh, H1)
		mixed_elem = MixedElement([SREG, H1, H2])
		V2 = FunctionSpace(mesh, mixed_elem)

		self.Fr = TensorFunctionSpace(mesh, 'DG', degree = 0, shape= (3,3))

		self.z = Function(V2)
		# M, u, v = split(z)
		# self.M_, self.u_, self.v_ = TestFunctions(V2)
		self.solver = self.set_solver()
	# yields self.b, self.L, self.jacobian

	def set_model(self):
		pass

	def define_boundary_conditions(self):
		pass

	def parse_args(self):

		class SmartFormatter(argparse.HelpFormatter):

			def _split_lines(self, text, width):
				if text.startswith('R|'):
					return text[2:].splitlines()  
				# this is the RawTextHelpFormatter._split_lines
				return argparse.HelpFormatter._split_lines(self, text, width)

		self.parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
		# parser.add_argument("-c", "--config", required=False,
		#                     help="JSON configuration string for this experiment")
		self.parser.add_argument("--E", type=float, default=1.)
		self.parser.add_argument("--nu", type=float, default=0.)
		self.parser.add_argument("--h", type=float, default=0.1)
		self.parser.add_argument("--outdir", type=str, default=None)
		self.parser.add_argument("--postfix", type=str, default='')
		self.parser.add_argument("--parameters", type=str, default=None)
		self.parser.add_argument("--f0", type=float, default=.0)

		self.add_userargs()

		args, unknown = self.parser.parse_known_args()

		if len(unknown):
			print('Unrecognised arguments:')
			print(unknown)
			print('continuing in 3s')
			# sleep(1)

		return args

	def parameters(self, args):
		parameters = {
			"material": {
				"nu": args.nu,
				"E": args.E,
				"gamma": 1.,
			},
			"geometry": {
				"meshsize": args.h
			},
			"load": {
				"f0": args.f0
			}
		}

		print('Parameters:')
		print(parameters)
		return parameters

	def add_userargs(self):
		pass

	def create_output(self, fname):
		args = self.args
		parameters = self.parameters
		self.signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
		print('Signature is: ', self.signature)

		if args.outdir == None:
			outdir = "output/{:s}-{}{}".format(fname, self.signature, args.postfix)
		else:
			outdir = args.outdir

		self.outdir = outdir
		Path(outdir).mkdir(parents=True, exist_ok=True)
		print('Outdir is: ', outdir)
		print('Output in: ', os.path.join(outdir, fname+'.xdmf'))
		print('P-proc in: ', os.path.join(outdir, fname+'_pproc.xdmf'))

		file_results = XDMFFile(os.path.join(outdir, fname+'.xdmf'))
		file_results.parameters["flush_output"] = True
		file_results.parameters["functions_share_mesh"] = True

		file_pproc = XDMFFile(os.path.join(outdir, fname+'_pproc.xdmf'))
		file_pproc.parameters["flush_output"] = True
		file_pproc.parameters["functions_share_mesh"] = True

		file_mesh = File(os.path.join(outdir, fname+"_mesh.xml"))

		with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
			json.dump(parameters, f)
		print('DEBUG: pproc file', os.path.join(outdir, fname+'_pproc.xdmf'))
		return file_results, file_pproc, file_mesh

	def create_mesh(self):
		pass

	def set_solver(self):
		# snes_options_z = {
		# 	"snes_type": "newtontr",
		# 	"snes_stol": 1e-6,
		# 	"snes_atol": 1e-6,
		# 	"snes_rtol": 1e-6,
		# 	"snes_max_it": 1000,
		# 	"snes_monitor": ''}

		# solver = PETScSNESSolver()
		# snes = solver.snes()
		# snes.setType(snes_options_z["snes_type"])
		# # snes.setType("newtontr")
		# ksp = snes.getKSP()
		# ksp.setType("preonly")
		# pc = ksp.getPC()
		# pc.setType("lu")


		# for option, value in snes_options_z.items():
		# 	print("setting ", option, value)
		# 	PETScOptions.set(option, value)

		# # solver.parameters.update(snes_solver_parameters)
		# # info(solver.parameters, True)

		# self.solver = solver

		# F = assemble(self.F, tensor=b)
		# [bc.apply(b, x) for bc in self.bcs]

		# J = assemble(self.jacobian, tensor=A)
		# [bc.apply(A) for bc in self.bcs]

		# import pdb; pdb.set_trace()
		# return solver
		pass

	def solve(self):
		# solver.solve(self.F == 0, self.z, self.bcs, J=self.jacobian)
		snes_solver_parameters = {
				# "linear_solver": "lu",
				# "maximum_iterations": 300,
				# "report": True,
				# "monitor": True,
				# "line_search": "basic",
				# "snes_type": "newtontr",
				# "method": "newtonls",
				"newton_solver": {
					"linear_solver": "mumps",
					"absolute_tolerance": 1e-6,
					"relative_tolerance": 1e-5
					}
				}

		# solve(self.F==0, self.z, self.bcs, J=self.J,
		# 	solver_parameters={"newton_solver": {"relative_tolerance":1e-5}})
		solve(self.F==0, self.z, self.bcs, J=self.J,
			solver_parameters=snes_solver_parameters)
		# solve(self.F==0, self.z, self.bcs, J=self.jacobian)
		# (it, reason) = self.solver.solve(self.problem, self.z.vector())



