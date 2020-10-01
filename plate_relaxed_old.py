from dolfin import *
from ufl.operators import And
from mshr import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
import site
import sys
import pdb
site.addsitedir('./old')
print(sys.path)
# pdb.set_trace()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
print(PETSc.Options().getAll())
from problem import PlateProblemSNES
# from model import emin, emax, G, DG_uv, DDG_uv
# set_log_level(0)
import argparse
from urllib.parse import unquote
from time import sleep
import os

petsc_options_solver = {
    "snes_type": "newtontr",
    "snes_stol": 1e-6,
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_max_it": 1000,
    "snes_monitor": 'True'}


class SmartFormatter(argparse.HelpFormatter):
	def _split_lines(self, text, width):
		if text.startswith('R|'):
			return text[2:].splitlines()  
		# this is the RawTextHelpFormatter._split_lines
		return argparse.HelpFormatter._split_lines(self, text, width)

parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
# parser.add_argument("-c", "--config", required=False,
#                     help="JSON configuration string for this experiment")
parser.add_argument("--E", type=float, default=1.)
parser.add_argument("--nu", type=float, default=0.)
parser.add_argument("--ell_e", type=float, default=0.1)
parser.add_argument("--h", type=float, default=0.1)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--postfix", type=str, default='')
parser.add_argument("--parameters", type=str, default=None)
parser.add_argument("--f0", type=float, default=.0)
parser.add_argument("--debug", type=bool, default=False)

args, unknown = parser.parse_known_args()

def set_solver():
	solver = PETScSNESSolver()
	snes = solver.snes()
	snes.setType(petsc_options_solver["snes_type"])
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
			"absolute_tolerance": 1e-3,
			"relative_tolerance": 1e-3,
			"solution_tolerance": 1e-3}

	solver.parameters.update(snes_solver_parameters)
	# info(solver.parameters, True)
	# import pdb; pdb.set_trace()
	return solver

outdir = 'output'
file_results = XDMFFile(os.path.join(outdir, "coin_relax.xdmf"))
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# set_log_level(INFO)
# E, nu = 1.0, 0.3
E, nu = Constant(1.0), Constant(0.0)
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/(1.0 - nu**2)

h = .1


# PETScOptions.set('log_view')
# PETScOptions.set('snes_view')
PETScOptions.set('snes_linesearch_monitor')

# PETScOptions.set('ksp_type', 'preonly')
# PETScOptions.set('ksp_gmres_restart', '30')
# PETScOptions.set('ksp_monitor_true_residual', 'True')
# PETScOptions.set('ksp_converged_reason', 'True')
# PETScOptions.set('ksp_atol', '1e-15')
# PETScOptions.set('ksp_rtol', '1e-6')
# PETScOptions.set('ksp_max_it', '1000')
# PETScOptions.set('ksp_view')
# PETScOptions.set('ksp_monitor_true_residual')
# PETScOptions.set('pc_type', 'lu')
# PETScOptions.set('pc_type', 'fieldsplit')
# PETScOptions.set('pc_fieldsplit_type', 'schur')
# PETScOptions.set('pc_fieldsplit_detect_saddle_point')

domain = Rectangle(Point(0., 0.), Point(1., 1.))
# mesh = generate_mesh(domain, 35)
# mesh = Mesh('meshes/coin.xml')
# fname = 'coinforall_small'
# fname = 'coinforall'
fname = 'coin'
# meshfile = "meshes/%s-%s.xml"%(fname, self.signature)

form_compiler_parameters = {
	"representation": "uflacs",
	"quadrature_degree": 2,
	"optimize": True,
	"cpp_optimize": True,
}

mesh = Mesh("meshes/{}.xml".format(fname))
domains = MeshFunction('size_t',mesh,'meshes/{}_physical_region.xml'.format(fname))
dx = Measure("dx", metadata=form_compiler_parameters, subdomain_data=domains)
# dx = Measure("dx", subdomain_data=domains)

# iterative_solver = False

r = 2
SREG = FiniteElement('HHJ', mesh.ufl_cell(), r)
H2 = FiniteElement('CG', mesh.ufl_cell(), r + 1)
H1 = VectorElement('CG', mesh.ufl_cell(), 1, dim = 2)
L2 = FunctionSpace(mesh, 'DG', 0)
V = FunctionSpace(mesh, SREG * H2)
# V2 = FunctionSpace(mesh, SREG * H2 * H1)

V1 = FunctionSpace(mesh, H1)
mixed_elem = MixedElement([SREG, H1, H2])
V2 = FunctionSpace(mesh, mixed_elem)

p = Expression('-t', t=0, degree=0)
K = Constant(1.)
ell_e = args.ell_e
A = 1./3.*as_matrix([[lmbda/(lmbda+2*mu)+1,lmbda/(lmbda+2*mu),0], \
                     [lmbda/(lmbda+2*mu),lmbda/(lmbda+2*mu)+1,1./3.], \
                     [0,1./3.,1./3.]])
S = inv(A)

# bc_clamped = [DirichletBC(V2.sub(2), Constant(0.), 'on_boundary'),
# 				DirichletBC(V2.sub(1), Constant((0.,0.)), 'on_boundary')]

bc_clamped = DirichletBC(V2.sub(2), Constant(0.), 'on_boundary')

z = Function(V2)
M, u, v = split(z)
M_, u_, v_ = TestFunctions(V2)

# print(assemble(v_*dx)[0:100])
# import pdb; pdb.set_trace()



(dM, du, dv) = TrialFunctions(V2)

def M_to_voigt(M):
	return as_vector([M[0,0],M[1,1],M[1,0]])

def M_from_voigt(m):
	return as_matrix([[m[0],m[2]],[m[2],m[1]]])

def a(M, tau):
	Mm = M_from_voigt(S*M_to_voigt(M))
	return inner(Mm, tau) * dx

et = Expression('t', t=0, degree = 0)

def a_m(u, v):
	return (inner(eps(u), eps(v)) + lmbda/(2.*mu)*tr(eps(u))*tr(eps(v)))*dx

def b(M, v):
	n = FacetNormal(mesh)
	Mnn = dot(dot(M, n), n)
	return inner(M, grad(grad(v))) * dx \
		- Mnn('+') * jump(grad(v), n) * dS \
		- Mnn * dot(grad(v), n) * ds #\

def pm(u):
	Id = Identity(u.geometric_dimension())
	return (inner(et*Id, eps(u)) + lmbda/(2.*mu)*tr(et*Id)*tr(eps(u)))*dx

def eps(u):
	return sym(grad(u))

def A(lmbda=0., mu=1.):
	return 1./3.*as_matrix([[lmbda/(lmbda+2*mu)+1,lmbda/(lmbda+2*mu),0], \
                     [lmbda/(lmbda+2*mu),lmbda/(lmbda+2*mu)+1,1./3.], \
                     [0,1./3.,1./3.]])

S = inv(A())

def emin(u, v):
	abs_u = pow(u[1]**2+u[0]**2+v**2, 1./2.)
	return -abs_u/2. + v/6.

def emax(u, v):
	abs_u = pow(u[1]**2+u[0]**2+v**2, 1./2.)
	return conditional(gt(-v, abs_u), -v/3., +abs_u/2. + v/6.)

def isLiquid(em, eM):
    return And(And(gt(em, -1/3), lt(eM,-2.*em+DOLFIN_EPS)), gt(eM, - em/2.-DOLFIN_EPS))

def isMartensite(em, eM):
	return And(And(lt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2+1/2))

def isElastic(em, eM):
	return And(And(lt(em, -1/3), gt(eM,-em/2.)), lt(eM, - em/2+1/2))

def phaseField(em, eM):
	elastic = conditional(isElastic(em, eM), 4, 0.)
	solid = conditional(isMartensite(em, eM), 2, elastic)
	return conditional(isLiquid(em, eM), 1, solid)

def G(U):
	u = U[0]
	v = U[1]
	em = emin(u, v)
	eM = emax(u, v)
	martEnergy = 3./2.*pow((em+1./3.), 2.) 
	stiffEnergy = 3./2.*pow((em+1./3.), 2.) + 1./2.*pow((em + 2.*eM-1.), 2.)

	isLiquid = And(And(gt(em, -1/3), lt(eM,-2.*em)), gt(eM, - em/2.))
	isMartensite = And(And(lt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2+1/2))
	isStiff = conditional(And(And(lt(em, -1/3), gt(eM,-em/2.)), lt(eM, - em/2+1/2)), stiffEnergy, 0)
	isSolid = conditional(isMartensite, martEnergy, isStiff)
	eff_density = conditional(isLiquid, 0., isSolid)
	return eff_density*dx + (3*lmbda+2*mu)/(6.*mu)*v*v*dx

def H(v):
	em = v/6.
	eM = conditional(gt(-v, 0), -v/3., + v/6.)

	martEnergy = 3./2.*pow((em-1./3.), 2.)
	stiffEnergy = 3./2.*pow((em-1./3.), 2.) + 1./2.*pow((em + 2.*eM-1.), 2.)

	isLiquid = And(And(gt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2))
	isMartensite = And(And(lt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2+1/2))

	isStiff = conditional(And(And(lt(em, -1/3), gt(eM,-em/2.)), lt(eM, - em/2+1/2)), stiffEnergy, 0)
	isSolid = conditional(isMartensite, martEnergy, isStiff)
	eff_density = conditional(isLiquid, 0., isSolid)
	return eff_density*dx + lmbda/(2.*mu)*v*v*dx

def phase(u, v):
	# phase of dofs
	em = emin(u, v)
	eM = emax(u, v)
	isLiquid = And(And(gt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2))
	isMartensite = And(And(lt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2+1/2))

	isStiff = conditional(And(And(lt(em, -1/3), gt(eM,-em/2.)), lt(eM, - em/2+1/2)), 3, -1)
	isSolid = conditional(isMartensite, 2, isStiff)
	# phase = conditional(isLiquid, 1, isSolid)
	phase = conditional(isLiquid, 1, 0)
	return phase

def DG_uv(U, V):
	u = U[0]
	v = U[1]
	u_ = V[0]
	v_ = V[1]
	em = emin(u, v)
	eM = emax(u, v)
	abs_u = pow((pow(u[0],2)+pow(u[1],2)),1/2)

	martEnergy_u = 3./2.*(em - 1./3.)*2*(-1./2.)*inner(u, u_)/abs_u
	stiffEnergy_u = 3./2.*(em - 1./3.)*2*(-1./2.)*inner(u, u_)/abs_u \
						+ (em + 2.*eM-1)*inner(u, u_)/abs_u

	martEnergy_v = 3./2.*(em - 1./3.)*2*1./3.*v_
	stiffEnergy_v = 3./2.*(em - 1./3.)*2*1./3.*v_ \
						+ (em + 2.*eM-1)*v_

	isLiquid     = And(And(gt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2))
	isMartensite = And(And(lt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2+1/2))

	isStiff = conditional(And(And(lt(em, -1/3), gt(eM,-em/2.)), lt(eM, - em/2+1/2)), 
		stiffEnergy_u + stiffEnergy_v, 0)
	isSolid = conditional(isMartensite, martEnergy_u + martEnergy_v, isStiff)
	eff_density_uv = conditional(isLiquid, 0., isSolid)
	return eff_density_uv*dx + (3*lmbda+2*mu)/(3.*mu)*v*v_*dx

def DDG_uv(U, V, dV):
	u = U[0]
	v = U[1]
	u_ = V[0]
	v_ = V[1]
	du = dV[0]
	dv = dV[1]
	em = emin(u, v)
	eM = emax(u, v)
	abs_u = pow((pow(u[0],2)+pow(u[1],2)),1/2)

	martEnergy_uu = -3./2.*(1./3.*v/abs_u - 1./2.)*inner(u_, du)
	stiffEnergy_uu = -3./2.*(1./3.*v/abs_u - 1./2.)*inner(u_, du) \
						+ 1./2.*((v/abs_u + 1./2.))*inner(du, u_)

	martEnergy_vv = 1./3.*dv*v_
	stiffEnergy_vv = 1./3.*dv*v_ + dv*v_

	isLiquid     = And(And(gt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2))
	isMartensite = And(And(lt(em, -1/3), lt(eM,-2*em)), gt(eM, - em/2+1/2))

	isStiff = conditional(And(And(lt(em, -1/3), gt(eM,-em/2.)), lt(eM, - em/2+1/2)),
		stiffEnergy_uu + stiffEnergy_vv, 0)
	isSolid = conditional(isMartensite, martEnergy_uu + martEnergy_vv, isStiff)
	eff_density_uv = conditional(isLiquid, 0., isSolid)
	return eff_density_uv*dx + (3*lmbda+2*mu)/(6.*mu)*dv*v_*dx

def postprocess(z, t):
	M, u, v = z.split(True)

	print('u norm = {}'.format(u.vector().norm('l2')))
	print('u n.h1 = {}'.format(norm(u,'h1')))
	print('v norm = {}'.format(v.vector().norm('l2')))
	print('v max, min = {}/{}'.format(np.max(v.vector()[:]), np.min(v.vector()[:])))
	print('M norm = {}'.format(M.vector().norm('l2')))

	em = emin(u, v)
	eM = emax(u, v)

	phase = phaseField(em, eM)
	phv = project(phase, L2)
	plt.clf()
	plt.colorbar(plot(phv))
	plt.savefig('phases.pdf')

	u.rename('u','u')
	v.rename('v','v')
	M.rename('M', 'M')
	file_results.write(u, t)
	file_results.write(v, t)
	file_results.write(M, t)


	em = project(em, L2)
	eM = project(eM, L2)
	# import pdb; pdb.set_trace()

	em.rename('em', 'em')
	eM.rename('eM', 'eM')
	file_results.write(em, t)
	file_results.write(eM, t)

	plt.clf()
	plt.scatter(em.vector()[:], eM.vector()[:])
	plt.axvline(-1./3.)
	es = np.linspace(0, -1., 3)
	plt.plot(es, -2.*es)
	plt.plot(es, -es/2.)
	plt.plot(es, -es/2.+1./2.)
	plt.xlabel('emin')
	plt.ylabel('emax')
	plt.savefig('phase.pdf')

	fou_energy.append(assemble(G([u, v])))
	mem_energy.append(assemble(1./2.*a_m(u, u)))
	ben_energy.append(assemble(1./2.*a(M, M)))
	tot_energy.append(assemble(1./2.*a(M, M)- p*v*dx + 1./2.*	G([u, v]) + 1./2.*a_m(u, u)))


	return {'fou_energy': fou_energy, 'mem_energy': mem_energy, 'ben_energy': ben_energy, 'tot_energy': tot_energy}

print('full problem')
# p.t = -1.
et.t = 0.
f0 = args.f0
print('elastic length: {}'.format(ell_e))
lagrangian = 1./2.*(a(M, M) - a_m(u, u) - 1./ell_e**2.*G([u, v])) - b(M, v) + p*v*dx(2) + pm(u)
residual = - a_m(u, u_)  + a(M, M_)  	\
			- b(M_, v)  - b(M, v_)  	\
			- 1./ell_e**2.*DG_uv([u, v], [u_, v_]) \
			+ Constant(f0)*v_*dx(2) + pm(u_)
jacobian = - a_m(du, u_) + a(dM, M_) 		\
			- b(M_, dv) - b(dM, v_) 		\
			- 1./ell_e**2.*DDG_uv([u, v], [u_, v_], [du, dv])
problem = PlateProblemSNES(lagrangian, z, bc_clamped, residual = residual, jacobian = jacobian)

F = assemble(residual)
J = assemble(jacobian)

print('J linf: {:.5f}'.format(J.norm('linf')))
print('F   l2: {:.5f}'.format(F.norm('l2')))

solver  = set_solver()


(it, conv) = solver.solve(problem, z.vector())
M, u, v = z.split(True)

plt.clf()
plt.colorbar(plot(u, mode='displacement'))
plt.savefig('full-u.pdf')

plt.clf()
plt.colorbar(plot(v))
plt.savefig('full-v.pdf')

tot_energy = []
fou_energy = []
mem_energy = []
ben_energy = []
fou_energy = []

import pandas as pd

data = postprocess(z, 0)
dataf = pd.DataFrame(data)
print(dataf)
dataf.to_json(os.path.join(outdir, "time_data.json"))


