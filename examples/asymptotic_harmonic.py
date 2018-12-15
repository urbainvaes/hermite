#!/usr/bin/env python
#
# Copyright (C) 2018 Urbain Vaes
#
# This file is part of hermipy, a python/C++ library for automating the
# Hermite Galerkin method.
#
# hermipy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hermipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import hermipy.equations as eq
import hermipy.quad as quad
import hermipy.series as series
import hermipy.function as func

import numpy as np
import numpy.linalg as la
import sympy as sym

# Declare variables, operators and parameters {{{
sym.init_printing()
equation = eq.McKean_Vlasov_harmonic_noise
x, y, z, f = equation.x, equation.y, equation.z, equation.f
params = equation.params()
params['γ'], params['θ'] = 0, 0
forward, params = equation.equation(params), equation.params()
β, Vp, θ, m, ε = (params[k] for k in ('β', 'Vp', 'θ', 'm', 'ε'))

# Map the forward operator to a "backward operator"
factor = sym.exp(- sym.Rational(1, 2) * (y*y + z*z))
operator = eq.map_operator(forward, f, factor).expand()

epsilon = params['ε']
L0 = (operator*epsilon**2).expand().subs(epsilon, 0)
L1 = (operator*epsilon - L0/epsilon).expand().subs(epsilon, 0)
L2 = (operator - L1/epsilon - L0/epsilon**2).expand()

# Quadrature used to solve cell problems
degree, nquad, σy, σz = 10, 20, 1, 1
quad_num = quad.Quad.gauss_hermite(nquad, dirs=[1, 2], mean=[0, 0],
                                   cov=[[σy, 0], [0, σz]])

# Discretization in Hermite space
fyz = sym.Function('fyz')(y, z)

op = quad_num.discretize_op(L0.subs(f, fyz), degree,
                            sparse=False, index_set="triangle")
# }}}
# Expansion of the solution {{{
nterms = 7
zeros = [sym.Integer(0)] * nterms
u, centered, unk = zeros, zeros.copy(), zeros.copy()


def iL0(term):
    t_rhs = quad_num.transform(term, degree)
    solution = la.solve(op.matrix[1:, 1:], t_rhs.coeffs[1:])
    solution = np.array([0, *solution])
    sol_series = series.Series(solution, t_rhs.position, significant=14)
    symbolic = sol_series.to_function().as_xyz()
    symbolic = func.Function.sanitize(symbolic, max_denom=1e8)

    print("--> Solving cell problem with rhs: " + str(term))
    print("----> Solution: " + str(symbolic))
    diff = (L0.subs(f, symbolic).doit() - term)
    if len(diff.expand().free_symbols) > 0:
        print("----> Error: does not match!")
        sym.pprint(diff)
        import ipdb
        ipdb.set_trace()
    return symbolic


for i in range(nterms):
    print("Solving for power " + str(i))

    unk[i] = sym.Function('u{}'.format(i))(x)
    u[i] = unk[i]

    rhs = sym.Integer(0)
    if i > 0:
        rhs += - L1.subs(f, u[i - 1])
    if i > 1:
        rhs += - L2.subs(f, u[i - 2])

    split = func.Function(rhs.doit().expand(), dim=3).split()

    for term in split:
        x_part = term[-1] * term[0].as_xyz()
        yz_part = term[1].as_xyz() * term[2].as_xyz()
        t_rhs = quad_num.transform(yz_part, degree)
        centered[i] += round(t_rhs.coeffs[0], 10) * x_part
        u[i] += x_part * iL0(yz_part)
    u[i] = func.Function.sanitize(u[i])
    centered[i] = func.Function.sanitize(centered[i])

# }}}
# Some manual calculations {{{
# Operator
fx = sym.Function('fx')(x)
LFP = ((1/β)*sym.exp(-β*Vp)*(fx*sym.exp(β*Vp)).diff(x)).diff(x)

# Centering condition for u₀
print("Equation for u₀: ")
sym.pprint(centered[2])
Z = sym.symbols('Z', real=True)
solution_0 = sym.exp(-β*Vp)/Z
if not centered[2].subs(unk[0], solution_0).doit().cancel() == 0:
    print("Error!")
    exit(0)

# Centering condition for u₁
for i in range(3, 6):
    print("Equation for u_{}: ".format(i-2))
    sym.pprint(centered[i])

    # The solution is 0
    for j in range(nterms):
        u[j] = u[j].subs(unk[i-2], 0).doit().expand()
        centered[j] = centered[j].subs(unk[i-2], 0).doit().expand()

# Centering condition for u₂
print("Equation for u₄: ")
sym.pprint(centered[6])

C1, C2 = sym.symbols('C1 C2', real=True)

fokker_planck = - ((1/β)*sym.exp(-β*Vp)*(unk[4]*sym.exp(β*Vp)).diff(x)).diff(x)
remainder = centered[6] - fokker_planck

beta3 = (remainder*β**3).expand().subs(β, 0)
beta2 = ((remainder - beta3/β**3) * β**2).expand().subs(β, 0)
beta1 = ((remainder - beta3/β**3 - beta2/β**2) * β).expand().subs(β, 0)
beta0 = remainder - beta3/β**3 - beta2/β**2 - beta1/β

assert (unk[0].diff(x)*Vp.diff(x)).diff(x, x, x, x) \
    + (unk[0].diff(x, x, x)*Vp.diff(x)).diff(x, x) == beta2

assert (Vp.diff(x)*Vp.diff(x)*unk[0].diff(x)).diff(x, x, x) \
    - (Vp.diff(x)*Vp.diff(x, x)*unk[0].diff(x)).diff(x, x) - beta1 == 0

int_fokker_planck = - (1/β)*sym.exp(-β*Vp)*(unk[4]*sym.exp(β*Vp)).diff(x)

int_beta1 = (Vp.diff(x)*Vp.diff(x)*unk[0].diff(x)).diff(x, x) \
            - (Vp.diff(x)*Vp.diff(x, x)*unk[0].diff(x)).diff(x)

int_beta2 = (unk[0].diff(x)*Vp.diff(x)).diff(x, x, x) \
            + (unk[0].diff(x, x, x)*Vp.diff(x)).diff(x)

int_beta3 = unk[0].diff(x, x, x, x, x)

integral1 = int_fokker_planck + \
            int_beta1/β + int_beta2/β**2 + int_beta3/β**3 + C1

assert (integral1.diff(x) - centered[6]).expand() == 0

# C1 = 0
integral1 = integral1.subs(C1, 0)

integrand = ((integral1.subs(unk[0], solution_0).doit().expand()
                       .subs(unk[4], solution_0*unk[4]).doit().expand()
              / solution_0).expand())

integral2 = 3*(Vp.diff(x, x))**2/(2*β) - Vp.diff(x, x, x, x)/β**2 \
            - Vp.diff(x, x)*Vp.diff(x)**2 - unk[4]/β \
            + 2/β*Vp.diff(x)*Vp.diff(x, x, x) - Vp.diff(x, x)**2/β \
            + (Vp.diff(x)*Vp.diff(x, x)**2).integrate(x) + C2

assert (integrand - integral2.diff(x)).expand() == 0

solution_4 = sym.solve(integral2, unk[4])[0] * solution_0

assert centered[6].subs(unk[0], solution_0)\
                  .subs(unk[4], solution_4).doit().expand() == 0
# }}}
# Projection on x - z {{{
quady = quad.Quad.gauss_hermite(nquad, dirs=[1], mean=[0], cov=[[σy]])
solution, proj_xz = u[0] + ε*u[1] + ε**2*u[2] + ε**3*u[3] + ε**4*u[4], 0
split = func.Function(solution.expand(), dim=3).split()
n_proj = 5
proj_xz = [0]*n_proj
for i in range(n_proj):
    split = func.Function(u[i].expand(), dim=3).split()
    for term in split:
        xz_part = term[-1] \
            * term[0].as_xyz() \
            * term[2].as_xyz()
        integ = quady.integrate(term[1].as_xyz())
        proj_xz[i] += integ * xz_part
        proj_xz[i] = func.Function.sanitize(proj_xz[i])
# }}}
# Check solution {{{

print("Solution: ")
sym.pprint(func.Function.sanitize(solution).factor())

solution_x = 0
split = func.Function(solution.expand(), dim=3).split()

for term in split:
    x_part = term[-1] * term[0].as_xyz()
    yz_part = term[1].as_xyz() * term[2].as_xyz()
    solution_x += x_part * quad_num.integrate(yz_part)

solution_x = func.Function.sanitize(solution_x)\
            .subs(unk[0], solution_0)\
            .subs(unk[4], solution_4)

print("x-projection of the solution: ")
sym.pprint(func.Function.sanitize(solution_x).factor())

handle = func.Function.sanitize(solution_x).factor()

# Checks
solution = solution.subs(unk[0], solution_0).subs(unk[4], solution_4)

for i in range(nterms):
    u[i] = u[i].subs(unk[0], solution_0)\
               .subs(unk[4], solution_4)

assert (operator/epsilon**4).expand()\
       .subs(f, solution + ε**5*u[5] + ε**6*u[6])\
       .doit().expand().subs(epsilon, 0) == 0


# }}}
# Plot {{{
n_points = 100
potential = (x**4/4 - x**2/2) + θ/2*(x**2 - 2*x*m)
quad_gauss = quad.Quad.gauss_hermite(n_points, dirs=[0, 2], cov=[[1, 0], [0, 1]])
quadx = quad.Quad.gauss_hermite(100, dirs=[0], cov=[[1]])

r = sym.Rational
θn, mn, βn, εn = r(0), r(0, 5), r(1), r(1, 2)
potential_n = potential.subs(((θ, θn), (m, mn), (β, βn), (ε, εn)))
Z_n = quadx.integrate(sym.exp(-βn*potential_n), flat=True)
solution_0_n = sym.exp(-βn*potential_n)/Z_n
solution_4_n = solution_4.subs(Vp, potential).doit()
solution_4_n = solution_4_n.factor().subs(((θ, θn), (m, mn), (β, βn), (ε, εn), (Z, Z_n)))
C2_n = quadx.integrate(sym.solve(solution_4_n, C2)[0] * solution_0_n, flat=True)
solution_4_n = solution_4_n.subs(C2, C2_n)
assert abs(quadx.integrate(solution_0_n, flat=True) - 1) < 1e-8
assert abs(quadx.integrate(solution_4_n, flat=True) - 0) < 1e-8

rho_z = 1/sym.sqrt(2*sym.pi) * sym.exp(-z*z/2)

proj_xz_n = [0]*n_proj
for i, p in enumerate(proj_xz):
    proj_xz_n[i] = (proj_xz[i]*rho_z).subs(unk[0], solution_0)\
                                  .subs(unk[4], solution_4).doit()
    proj_xz_n[i] = proj_xz_n[i].subs(Vp, potential).doit()
    proj_xz_n[i] = proj_xz_n[i].factor().subs(((θ, θn), (m, mn), (β, βn),
                                               (ε, εn), (Z, Z_n), (C2, C2_n)))
    assert abs(quad_gauss.integrate(proj_xz_n[i], flat=True) - i == 0) < 1e-8

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=22)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)

quad_visu = quad.Quad.newton_cotes([n_points, n_points], [2, 2], dirs=[0, 2])

for i in range(5):
    fig, ax = plt.subplots()
    exponent = '^' + str(i) if i is not 1 else ''
    cont = quad_visu.plot(proj_xz_n[i], ax=ax, title='$\\varepsilon{}$'.format(exponent))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\eta$')
    plt.colorbar(cont, ax=ax, pad=.01)
    for c in cont.collections:
        c.set_edgecolor("face")
    plt.tight_layout()
    plt.savefig("asymptotic-harmonic-{}.eps".format(i), bbox_inches='tight')
plt.close('all')

fig, axes = plt.subplots(2, 2)
for i in (0, 1):
    for j in (0, 1):
        n = 1 + 2*i + j
        cont = quad_visu.plot(proj_xz_n[n], ax=axes[i][j])
        plt.colorbar(cont, ax=axes[i][j], pad=.01)
        for c in cont.collections:
            c.set_edgecolor("face")
# plt.tight_layout()
# plt.savefig("test.eps", bbox_inches='tight')
plt.show()
# }}}
