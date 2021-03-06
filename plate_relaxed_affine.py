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
site.addsitedir('scripts')
print(sys.path)

import visuals
from subprocess import Popen, PIPE, check_output
import subprocess


from string import Template
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
import json
from pathlib import Path
from dolfin.cpp.log import log, LogLevel, set_log_level
import yaml
import hashlib

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
parser.add_argument("--parameters", type=str, default=None)

args, unknown = parser.parse_known_args()


parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
parser.add_argument("--parameters", type=str, default=None)


args, unknown = parser.parse_known_args()
# outdir = 'output/relax-traction-{}'.format(args.postfix) if args.postfix != '' else 'output/relax'

with open('parameters/slab.yaml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
outdir = 'output/relax-affine-{}-{}'.format(parameters['material']['p'], signature)

Path(outdir).mkdir(parents=True, exist_ok=True)
log(LogLevel.INFO, 'Outdir: {}'.format(outdir))
# parameters = parameters(args)
with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
    json.dump(parameters, f)

# sys.exit()

# import pdb; pdb.set_trace()

def set_solver(parameters = {}):
    solver = PETScSNESSolver()
    snes = solver.snes()
    snes.setType(petsc_options_solver["snes_type"])
    ksp = snes.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    # snes_solver_parameters = {
    #         # "linear_solver": "lu",
    #         # "linear_solver": "umfpack",
    #         "linear_solver": "mumps",
    #         "maximum_iterations": 100,
    #         "report": True,
    #         # "preconditioner": 'petsc_amg',
    #         # "sign": 'nonnegative',
    #         "line_search": "basic",
    #         "method": "newtonls",
    #         "absolute_tolerance": 1e-5,
    #         "relative_tolerance": 1e-5,
    #         "solution_tolerance": 1e-5,
    #         "krylov_solver": {
    #             "monitor_convergence": True,
    #             "report": True
    #         }
    #         }

    prefix = ''
    log(LogLevel.INFO, "---- Setting additional slepc options for eigen solver -----")
    for (parameter, value) in parameters['solver'].items():
        if value is not None:
            log(LogLevel.DEBUG, "DEBUG: setting {} {}".format(prefix + parameter, value))
            dolfin.PETScOptions.set(prefix + parameter, value)
    log(LogLevel.DEBUG, "------------------------------------------------------------")
    snes.setFromOptions()
    # info(solver.parameters, True)
    # import pdb; pdb.set_trace()
    return solver

def create_mesh(parameters):
    from dolfin import Point, XDMFFile
    import hashlib 

    domain = Rectangle(Point(0., 0.), Point(parameters['geometry']['Lx'], parameters['geometry']['Ly']))
    # domain = Circle(Point(0., 0.), 1.)
    meshsize = parameters['geometry']['meshsize']
    mesh = generate_mesh(domain, 1/meshsize)

    # domains = MeshFunction('size_t',mesh,'meshes/{}_physical_region.xml'.format(fname))
    ds = Measure("exterior_facet", domain = mesh)
    dS = Measure("interior_facet", domain = mesh)
    dx = Measure("dx", metadata=form_compiler_parameters)

    return mesh

fname = 'slab'

file_results = XDMFFile(os.path.join(outdir, "{}_relax.xdmf".format(fname)))
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# set_log_level(INFO)
# E, nu = 1.0, 0.3
E, nu = Constant(1.0), Constant(0.0)
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/(1.0 - nu**2)

h = .1

form_compiler_parameters = {
    "representation": "uflacs",
    "quadrature_degree": 2,
    "optimize": True,
    "cpp_optimize": True,
}


mesh = create_mesh(parameters)

# domains = MeshFunction('size_t',mesh,'meshes/{}_physical_region.xml'.format(fname))
dx = Measure("dx", metadata=form_compiler_parameters)
# dx = Measure("dx", metadata=form_compiler_parameters, subdomain_data=domains)
# dx = Measure("dx", subdomain_data=domains)
# plt.figure()
# plt.colorbar(
# dolfin.plot(domains)
# iterative_solver = False
# plt.savefig(os.path.join(outdir, 'domains.pdf'))

log(LogLevel.INFO, 'num vertices {}'.format(mesh.num_vertices()))

# log(LogLevel.INFO, 'Number of dofs: {}'.format(mesh.num_vertices()*(1+parameters['general']['dim'])))
plt.clf()
plot(mesh)
plt.savefig(os.path.join(outdir, 'mesh.pdf'))

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
log(LogLevel.INFO, "Num dofs {}".format(V2.dim()))

pressure = Expression('t', t=0, degree=0)
K = Constant(1.)
ell_e = 1.
A = 1./3.*as_matrix([[lmbda/(lmbda+2*mu)+1,lmbda/(lmbda+2*mu),0], \
                     [lmbda/(lmbda+2*mu),lmbda/(lmbda+2*mu)+1,1./3.], \
                     [0,1./3.,1./3.]])
S = inv(A)

bc_clamped = [DirichletBC(V2.sub(2), Constant(0.), 'on_boundary'),
                DirichletBC(V2.sub(1), Constant((0.,0.)), 'on_boundary')]
bc_free = []
bc_horiz = DirichletBC(V2.sub(2), Constant(0.), 'on_boundary')
bc_vert = [DirichletBC(V2.sub(2), Constant(0.), 'on_boundary')]

bc_flex = [DirichletBC(V2.sub(0), Constant([[1., 0], [0., 0.]]),
    'near(x[0], 0) or near(x[0], {})'.format(
        parameters['geometry']['Lx']))
    ]


hate = np.matrix([[1, 0], [0, 1]])
g = Expression(['x[0]*e11 + x[1]*e12', 
                'x[0]*e21 + x[1]*e22'], e11=hate.tolist()[0][0], 
                                        e12=hate.tolist()[0][1], 
                                        e21=hate.tolist()[1][0], 
                                        e22=hate.tolist()[1][1],
                                        degree = 1)

# V2= SREG, H1, H2
bc_aff_mem = [ DirichletBC(V2.sub(1), g, 'on_boundary'),
                # DirichletBC(V2.sub(1).sub(0), Constant(1.), 'near(x[0], {}) and on_boundary'.format(parameters['geometry']['Lx'])),
                # DirichletBC(V2.sub(1).sub(1), Constant(0.), 'near(x[0], {}) and on_boundary'.format(parameters['geometry']['Lx'])),
                ]


bcs = [bc_clamped, bc_free, bc_vert, bc_horiz]

z = Function(V2)
M, u, v = split(z)
M_, u_, v_ = TestFunctions(V2)

# print(assemble(v_*dx)[0:100])

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
    clambdamu = (lmbda*mu)/(lmbda+2.*mu)
    return (inner(eps(u), eps(v)) + clambdamu*tr(eps(u))*tr(eps(v)))*dx

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


if parameters['material']['p']=='zero':
    log(LogLevel.INFO, 'material regime p={}'.format(parameters['material']['p']))

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
        return eff_density*dx

    def DG_uv( U, V):
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

        return eff_density_uv*dx

    def DDG_uv( U, V, dV):
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

        return eff_density_uv*dx

elif parameters['material']['p']=='neg':
    log(LogLevel.INFO, 'material regime p={}'.format(parameters['material']['p']))

    def G(U):
        v = U[1]
        stiffEnergy = 1/2*2./3.*pow((v-1.), 2.)
        martEnergy= 1/2*2./3.*pow((v+1./2.), 2.)

        # isLiquid = And(And(gt(em, -1/3), lt(eM,-2.*em)), gt(eM, - em/2.))
        isLiquid = And(gt(v, -1/2), lt(v,1.))
        isElastic = gt(v, 1.)
        isMartensite = lt(v, -1./2.)
        isSolid = conditional(isMartensite, martEnergy, stiffEnergy)

        eff_density = conditional(isLiquid, 0., isSolid)
        return eff_density*dx

    def DG_uv( U, V):
        v = U[1]
        v_ = V[1]
        # em = emin(u, v)
        # eM = emax(u, v)
        stiffEnergy = 2./3.*(v-1.)*v_
        martEnergy= 2./3.*(v+1./2.)*v_

        # isLiquid = And(And(gt(em, -1/3), lt(eM,-2.*em)), gt(eM, - em/2.))
        isLiquid = And(gt(v, -1/2), lt(v,1.))
        isElastic = gt(v, 1.)
        isMartensite = lt(v, -1./2.)
        isSolid = conditional(isMartensite, martEnergy, stiffEnergy)

        eff_density = conditional(isLiquid, 0., isSolid)
        return eff_density*dx

    def DDG_uv( U, V, dV):
        v = U[1]
        v_ = V[1]
        dv = dV[1]
        stiffEnergy = 2./3.*dv*v_
        martEnergy= 2./3.*dv*v_

        # isLiquid = And(And(gt(em, -1/3), lt(eM,-2.*em)), gt(eM, - em/2.))
        isLiquid = And(gt(v, -1/2), lt(v,1.))
        isElastic = gt(v, 1.)
        isMartensite = lt(v, -1./2.)
        isSolid = conditional(isMartensite, martEnergy, stiffEnergy)

        eff_density = conditional(isLiquid, 0., isSolid)
        return eff_density*dx

else:
    log(LogLevel.CRITICAL, 'Not implemented')
    sys.exit()

def postprocess(z, t, it=0):
    M, u, v = z.split(True)
    if t<0:
        counter = it
    else:
        counter = t

    print('u norm = {}'.format(u.vector().norm('l2')))
    print('u n.h1 = {}'.format(norm(u,'h1')))
    print('v norm = {}'.format(v.vector().norm('l2')))
    print('v max, min = {}/{}'.format(np.max(v.vector()[:]), np.min(v.vector()[:])))
    print('M norm = {}'.format(M.vector().norm('l2')))
    print('processing timestep {}'.format(counter))

    em = emin(u, v)
    eM = emax(u, v)

    phase = phaseField(em, eM)
    phv = project(phase, L2)
    plt.clf()
    plt.colorbar(plot(phv))
    plt.savefig(os.path.join(outdir, 'phases-{}.pdf'.format(counter)))


    em = project(em, L2)
    eM = project(eM, L2)
    print('e max, min = {}/{}'.format(np.max(eM.vector()[:]), np.min(em.vector()[:])))

    # eps = s
    u.rename('u','u')
    v.rename('v','v')
    M.rename('M', 'M')
    # sigma.rename('sigma', 'sigma')
    # eps.rename('eps', 'eps')

    em.rename('em', 'em')
    eM.rename('eM', 'eM')
    phv.rename('phase', 'phase')




    file_results.write(u, counter)
    file_results.write(v, counter)
    file_results.write(M, counter)
    file_results.write(phv, counter)
    file_results.write(em, counter)
    file_results.write(eM, counter)
    print('DEBUG: written file_results, timestep {}'.format(counter))

    plt.clf()
    plt.scatter(em.vector()[:], eM.vector()[:])
    plt.axvline(-1./3.)
    es = np.linspace(0, -1., 3)
    plt.plot(es, -2.*es)
    plt.plot(es, -es/2.)
    plt.plot(es, -es/2.+1./2.)
    plt.xlabel('$e_{min}$')
    plt.ylabel('$e_{max}$')
    plt.savefig(os.path.join(outdir, 'diagram-{}.pdf'.format(counter)))

    # import pdb; pdb.set_trace()

    fou_energy = assemble(1./2.*G([u, v])+1./2.*Clambdamu*v*v*dx)
    mem_energy = assemble(1./2.*a_m(u, u)-3./4.*inner(eps(u), M)*dx)
    ben_energy = assemble(1./2.*a(M, M)-3./4.*inner(eps(u), M)*dx)
    tot_energy = assemble(1./2.*a(M, M) - 3./2.*inner(eps(u), M)*dx                     \
                            + 1./2.*a_m(u, u) + 1./2.*G([u, v])+1./2.*Clambdamu*v*v*dx      \
                            - pressure*v*dx(2) )
    work = assemble(pressure*v*dx(2))
    tot_energy = mem_energy+ben_energy+fou_energy
    max_v = np.max(v.vector()[:])
    min_em = np.min(em.vector()[:])
    max_eM = np.max(eM.vector()[:])
    max_abs_v = np.max(np.abs(v.vector()[:]))
    max_v = np.max(v.vector()[:]) 
    min_v = np.min(v.vector()[:])
    xterm = assemble(-3./4.*inner(eps(u), M)*dx)

    return {'load': t, 'fou_energy': fou_energy, 'mem_energy': mem_energy, 'ben_energy': ben_energy, 'tot_energy': tot_energy,
            'max_abs_v': max_abs_v,
            'min_em': min_em,
            'max_eM': max_eM,
            'work': work,
            'max_v': max_v,
            'min_v': min_v,
            'min_u1': min(u.sub(0).vector()[:]),
            'min_u2': min(u.sub(1).vector()[:]),
            'max_u1': max(u.sub(0).vector()[:]),
            'max_u2': max(u.sub(1).vector()[:]),
            'iso_found': assemble(1./2.*Clambdamu*v*v*dx),
            'xterm': xterm}

print('full problem')
# p.t = -1.
# et.t = 0.
f0 = Constant(0.)
Clambdamu = (3.*lmbda+2*mu)/(6.*mu)

print('Elastic length: {}'.format(ell_e))
print('Avg cell size {}'.format(.5*(mesh.hmax() + mesh.hmin())))

lagrangian =  1./2.*(a(M, M) - 3.*inner(M, eps(u))*dx - a_m(u, u))          \
            - 1./2.*G([u, v])                           \
            - 1./2.*Clambdamu*v*v*dx                    \
            - b(M, v) + pressure*v*dx + pm(u)

            # - 3./2.*inner(M, eps(u_))*dx - 3./2.*inner(eps(u), M_)*dx   \
            # - Constant(1/ell_e**2.)*Clambdamu*v*v_*dx                                         \
residual  = a(M, M_) - a_m(u, u_)                                       \
            - b(M_, v)  - b(M, v_)                                      \
            - 1./2.*DG_uv([u, v], [u_, v_])                             \
            + pressure*v_*dx + pm(u_)

            # - Constant(1/ell_e**2.)*Clambdamu*dv*v_*dx
jacobian = - a_m(du, u_) + a(dM, M_)        \
            - 3./2.*inner(dM, eps(u_))*dx - 3./2.*inner(M_, eps(du))*dx             \
            - b(M_, dv) - b(dM, v_)                                     \
            - 1./2.*DDG_uv([u, v], [u_, v_], [du, dv])  \



# without foundation

# lagrangian = 1./2.*(a(M, M) - 3.*inner(M, eps(u))*dx - a_m(u, u))             \
#           - b(M, v) + pressure*v*dx(2) + pm(u)
# residual  = a(M, M_) - a_m(u, u_)                                     \
#           - 3./2.*inner(M, eps(u_))*dx - 3./2.*inner(eps(u), M_)*dx   \
#           - b(M_, v)  - b(M, v_)                                      \
#           - Clambdamu*v*v_*dx                                         \
#           + pressure*v_*dx(2) + pm(u_)

# jacobian = - a_m(du, u_) + a(dM, M_)      \
#           - 3./2.*inner(dM, eps(u_))*dx - 3./2.*inner(M_, eps(du))*dx \
#           - b(M_, dv) - b(dM, v_)                                     \
#           - Clambdamu*dv*v_*dx

# solver  = set_solver()
# pressure.t = args.f0
# problem = PlateProblemSNES(lagrangian, z, bc_clamped, residual = residual, jacobian = jacobian)
# (it, conv) = solver.solve(problem, z.vector())




problem = PlateProblemSNES(lagrangian, z, bc_aff_mem, residual = residual, jacobian = jacobian)

F = assemble(residual)
J = assemble(jacobian)

log(LogLevel.INFO, 'J linf: {:.5f}'.format(J.norm('linf')))
log(LogLevel.INFO, 'F   l2: {:.5f}'.format(F.norm('l2')))

solver  = set_solver(parameters)

# z = Function(V2)
(it, conv) = solver.solve(problem, z.vector())
log(LogLevel.INFO, 'DEBUG: converged reason: {}'.format(conv))
log(LogLevel.INFO, 'J linf: {:.5f}'.format(J.norm('linf')))
log(LogLevel.INFO, 'F   l2: {:.5f}'.format(F.norm('l2')))
data = postprocess(z, t=0, it=0)
# log(LogLevel.INFO, '{}'.format(data))
# bc_aff_mem
import pandas as pd
# import pdb; pdb.set_trace()

dataf = pd.DataFrame(data, index=[0])
log(LogLevel.INFO,'{}'.format(dataf))
dataf.to_json(os.path.join(outdir, "data.json"))


sys.exit()






import pandas as pd

time_data = []
solver  = set_solver(parameters)
loads = np.linspace(0., f0, 5)
for (step, t) in enumerate(loads):
    log(LogLevel.INFO, t, step)
    pressure.t = t
    log(LogLevel.INFO, '')
    log(LogLevel.INFO, '')
    log(LogLevel.INFO, 'Solving problem with p={:.3f}'.format(t))
    log(LogLevel.INFO, '')
    problem = PlateProblemSNES(lagrangian, z, bcs[0], residual = residual, jacobian = jacobian)

    F = assemble(residual)
    J = assemble(jacobian)

    log(LogLevel.INFO, 'J linf: {:.5f}'.format(J.norm('linf')))
    log(LogLevel.INFO, 'F   l2: {:.5f}'.format(F.norm('l2')))

    # z = Function(V2)
    (it, conv) = solver.solve(problem, z.vector())
    log(LogLevel.INFO, 'DEBUG: converged reason: {}'.format(conv))

    data = postprocess(z, t=t, it=step)
    # dataf = pd.DataFrame(data)
    time_data.append(data)

# log(LogLevel.INFO, time_data)
dataf = pd.DataFrame(time_data)
log(LogLevel.INFO, dataf)
dataf.to_json(os.path.join(outdir, "time_data.json"))

# pdb.set_trace()

plt.figure()
plt.plot(loads, dataf['tot_energy'].values, ':', lw=4, label="Total")
plt.plot(loads, dataf['mem_energy'].values, marker = 'o', label="Mem")
plt.plot(loads, dataf['fou_energy'].values, marker = 'o', label="Fou")
plt.plot(loads, dataf['ben_energy'].values, marker = 'o', label="Ben")
plt.plot(loads, dataf['xterm'].values, marker = 'o', label="xterm")
plt.xlabel('Force')
plt.ylabel('Energies')
plt.legend()
visuals.setspines()
plt.savefig(os.path.join(outdir, 'energies.pdf'))
#plt.show()
plt.close()

plt.figure()
plt.plot(loads, dataf['max_abs_v'], ':', lw=4, label="max abs v")
plt.plot(loads, dataf['max_v'], ':', lw=4, label="$v_{max}$")
plt.plot(loads, dataf['min_v'], ':', lw=4, label="$v_{min}$")
plt.plot(loads, np.max((dataf['max_u1'].values,dataf['max_u2'].values), axis=0), ':', lw=4, label="max ui")
plt.legend()
visuals.setspines()
plt.savefig(os.path.join(outdir, 'displacements.pdf'))


