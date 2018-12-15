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

import os
import math
import argparse
import sympy as sym
import numpy as np
import matplotlib
import hermipy as hm
import hermipy.equations as eq

hm.settings['tensorize'] = True
hm.settings['sparse'] = True
hm.settings['trails'] = False
hm.settings['cache'] = True

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', '--directory', type=str)
parser.add_argument('-i', '--interactive', action='store_true')
parser.add_argument('-p', '--parallel', action='store_true')
parser.add_argument('-tq', '--convergence_quadratic', action='store_true')
parser.add_argument('-tcd', '--convergence_degree', action='store_true')
parser.add_argument('-tce', '--convergence_epsilon', action='store_true')
parser.add_argument('-tb', '--bifurcation', action='store_true')
parser.add_argument('-bmin', '--beta_min', type=float)
parser.add_argument('-bmax', '--beta_max', type=float)
parser.add_argument('-sstep', '--arclength', type=float)
parser.add_argument('-m0', '--mass0', type=float)
parser.add_argument('-e', '--epsilon', type=str)
parser.add_argument('-b', '--beta', type=float)
parser.add_argument('-t', '--theta', type=str)
parser.add_argument('-g', '--gamma', type=float)
parser.add_argument('-m', '--mass', type=float)
parser.add_argument('-d', '--degree', type=int)
args = parser.parse_args()

# Directory for output files
dir_out = ""
if args.directory:
    dir_out = args.directory + "/"
    os.makedirs(dir_out, exist_ok=True)

# Matplotlib configuration
matplotlib.rc('font', size=14)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)

if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

# Dimension of the problem
dim = 3

# Equation parameters
r, s = sym.Rational, sym.symbols
equation = eq.McKean_Vlasov_harmonic_noise
x, y, z, f = equation.x, equation.y, equation.z, equation.f

if args.convergence_epsilon:
    args.epsilon = -1
    args.gamma = -1

if args.bifurcation:
    args.beta, args.mass = -1, -1
    minit = args.mass0 if args.mass0 else 1
    βmin = args.beta_min if args.beta_min else 1
    βmax = args.beta_max if args.beta_max else 6
    sstep = args.arclength if args.arclength else .1

if args.convergence_quadratic:
    args.beta = 1


def set_param(arg, default, symbol):
    if arg and r(arg) < 0:
        return s(symbol)
    if arg:
        return r(arg)
    return default


β = set_param(args.beta, r(5), 'β')
ε = set_param(args.epsilon, r(1, 5), 'ε')
γ = set_param(args.gamma, 0, 'γ')
θ = set_param(args.theta, 0, 'θ')
m = set_param(args.mass, 0, 'm')

params = {'β': β, 'ε': ε, 'γ': γ, 'θ': θ, 'm': m}
Vp = x**2/2 if args.convergence_quadratic else x**4/4 - x**2/2

# Numerical parameters
if args.convergence_epsilon:
    s2x, s2y, s2z = r(1, 30), r(1), r(1)
elif args.convergence_quadratic:
    s2x, s2y, s2z = r(1, 5), r(1, 5), r(1, 5)
else:
    s2x, s2y, s2z = r(1, 20), r(1, 5), r(1, 5)

index_set = 'cube' if args.convergence_quadratic else 'rectangle'
degree = args.degree if args.degree else 50
kwargs0 = {'degree': degree, 'index_set': index_set}
n_points_num = 2*degree + 1

# Potential for approximation
Vqx = sym.Rational(1/2)*x*x/s2x
Vqy = sym.Rational(1/2)*y*y/s2y
Vqz = sym.Rational(1/2)*z*z/s2z

# Fokker Planck for McKean-Vlasov equation
params.update({'Vp': Vp})
forward = equation.equation(params)

# Map to appropriate space
if args.convergence_quadratic:
    factor_x = sym.exp(- 1/2 * Vqx)
    factor_pq = sym.exp(- 1/2 * (Vqy + Vqz))
elif args.bifurcation:
    factor_x = sym.exp(- 1/2 * Vqx)
    factor_pq = sym.exp(- 1/2 * (y*y/2 + Vqy + z*z/2 + Vqz))
else:
    factor_x = sym.exp(- 1/2 * (Vqx + β*Vp))
    factor_pq = sym.exp(- 1/2 * (y*y/2 + Vqy + z*z/2 + Vqz))

factor = factor_x * factor_pq

# Calculation of the solution
new_q = hm.Quad.gauss_hermite
cov = [[s2x, 0, 0], [0, s2y, 0], [0, 0, s2z]]
quad = new_q(n_points_num, dim=3, mean=[0]*3, cov=cov, factor=factor)
min_quad = new_q(n_points=degree+1, dim=3, mean=[0]*3, cov=cov, factor=factor)

# Projections
qx, qy, qz = quad.project(0), quad.project(1), quad.project(2)
wx = qx.factor * qx.factor / qx.position.weight()
wy = qy.factor * qy.factor / qy.position.weight()
wz = qz.factor * qz.factor / qz.position.weight()
qxy, qxz, qyz = quad.project([0, 1]), quad.project([0, 2]), quad.project([1, 2])

# Integral operators
Ix = qx.transform(wx, degree=degree, index_set=index_set)
Iy = qy.transform(wy, degree=degree, index_set=index_set)
Iz = qz.transform(wz, degree=degree, index_set=index_set)

# Moment operators
mx1 = qx.transform(wx * x, degree=degree, index_set=index_set)
my1 = qy.transform(wy * y, degree=degree, index_set=index_set)
mz1 = qz.transform(wz * z, degree=degree, index_set=index_set)

# Marginals in white noise case
βinit = βmax if args.bifurcation else β
minit = minit if args.bifurcation else m
Vx = Vp + θ*(x - minit)**2/2
ux = sym.exp(-βinit*Vx) / qx.integrate(sym.exp(-βinit*Vx), flat=True)
uy = sym.exp(-y*y/2) / qy.integrate(sym.exp(-y*y/2), flat=True)
uz = sym.exp(-z*z/2) / qz.integrate(sym.exp(-z*z/2), flat=True)

if args.interactive:
    plt.ion()
    fig, ax = plt.subplots(2, 2)


def plot(t, βplot=β):

    # Retrieve degree
    d = t.degree

    # Integral operators
    Ix_d = Ix.subdegree(d)
    Iy_d = Iy.subdegree(d)
    Iz_d = Iz.subdegree(d)

    # Projection on x-q subspace
    txy, txz = Iz_d*t, Iy_d*t

    # Projections
    tx, ty, tz = Iy_d*(Iz_d*t), Ix_d*(Iz_d*t), Ix_d*(Iy_d*t)

    # Moments
    mx = str((mx1.subdegree(d)*tx).coeffs[0])
    my = str((my1.subdegree(d)*ty).coeffs[0])
    mz = str((mz1.subdegree(d)*tz).coeffs[0])

    ax[0][0].clear()
    plot_position = True
    field = txz if plot_position else txy
    qxz.plot(field, ax=ax[0][0], bounds=False, vmin=0, extend='min')

    ax[0][1].clear()
    qx.plot(tx, ax=ax[0][1], title="1st moment - X: " + mx)
    if not args.bifurcation:
        qx.plot(ux, ax=ax[0][1])

    ax[1][0].clear()
    qy.plot(uy, ax=ax[1][0])
    qy.plot(ty, ax=ax[1][0], title="1st moment - P: " + my)

    ax[1][1].clear()
    qz.plot(uz, ax=ax[1][1])
    qz.plot(tz, ax=ax[1][1], title="1st moment - Q: " + mz)

    plt.draw()
    plt.pause(0.1)


def convergence_degree():

    degrees = list(range(5, degree + 1))
    mat = quad.discretize_op(forward, degrees[-1], index_set=index_set)

    eye = quad.varf('1', degree=degree, index_set=index_set)
    dt, Ns, scheme = 2**-9, int(1e4), "backward"
    dmin, d, degrees = 10, degree, []
    errors, mins, eigs = [], [], []

    # Initial condition
    t = quad.transform(ux*uy*uz, degree=degree, index_set=index_set)

    # Exact solution if Vp is quadratic
    if args.convergence_quadratic:
        solution = eq.solve_gaussian(forward, f, [x, y, z])
        t_exact = min_quad.transform(solution, **kwargs0)
        norm_sol = float(Ix*(Iy*(Iz*t_exact)))
        print('Norm of exact solution: ', abs(norm_sol - 1))
        t_exact = t_exact / norm_sol
        if args.interactive:
            plot(t_exact)

    while d >= dmin:
        r_mat, eye = mat.subdegree(d), eye.subdegree(d)
        t = t.subdegree(d)

        # Integral operators
        Ix_d = Ix.subdegree(d)
        Iy_d = Iy.subdegree(d)
        Iz_d = Iz.subdegree(d)

        Δ = np.inf
        for i in range(Ns):

            if args.interactive:
                plot(t)

            print("d: " + str(d) + ", i: " + str(i) +
                  ", dt: " + str(dt) + ", Δ: " + str(Δ))

            # Backward Euler
            if scheme == 'backward':
                total_op = eye - dt*r_mat
                new_t = total_op.solve(t)

            # Crank-Nicholson
            if scheme == 'crank':
                crank_left = eye - dt*r_mat*(1/2)
                crank_right = eye + dt*r_mat*(1/2)
                new_t = crank_left.solve(crank_right(t))

            # Normalization
            integral = (Ix_d*(Iy_d*(Iz_d*new_t))).coeffs[0]
            new_t = new_t / integral

            # Error
            Δ = ((t - new_t)*(t - new_t)).coeffs[0]/dt

            # Time adaptation
            threshold, dt_max = .01, 64
            if Δ*dt > 2*threshold:
                dt = dt / 2.
                continue
            elif Δ*dt < threshold and dt < dt_max:
                dt = dt * 2.
            t = new_t

            if Δ < 1e-15:

                if d == degree and not args.convergence_quadratic:
                    t_exact = t

                else:
                    error_series = t.subdegree(degree) - t_exact
                    error_series_x = Iy*(Iz*error_series)
                    error = min_quad.norm(error_series, n=1, flat=True)
                    error_x = qx.norm(error_series_x, n=1, flat=True)
                    min_ = np.min(qx.eval(Iy_d*(Iz_d*t)))
                    eig = r_mat(t).coeffs[0]/t.coeffs[0]
                    errors.append(error)
                    mins.append(abs(min_))
                    eigs.append(abs(eig))
                    degrees.append(d)
                    print(min_, eig, error, error_x)

                if args.interactive:
                    plot(t)

                d = d - 1
                break

    cond = np.asarray(degrees)*0 + 1
    xplot, yplot = np.extract(cond, degrees), np.extract(cond, errors)
    np.save(dir_out + "degrees", xplot),
    np.save(dir_out + "error_l1", yplot)
    fig, ax = plt.subplots()
    ax.semilogy(xplot, yplot, 'b.',
                label="$\\|\\rho_{{ {} }} - \\rho_d\\|_1$".format(degree))
    coeffs = np.polyfit(xplot, np.log10(yplot), 1)
    ax.semilogy(xplot, 10**coeffs[1] * 10**(coeffs[0]*xplot), 'b-')
    yplot = np.extract(cond, eigs)
    np.save(dir_out + "error_eig", yplot)
    ax.semilogy(xplot, yplot, 'r.', label="$|\\lambda_0(d)|$")
    coeffs = np.polyfit(xplot, np.log10(yplot), 1)
    ax.semilogy(xplot, 10**coeffs[1] * 10**(coeffs[0]*xplot), 'r-')
    ax.set_xlabel("$d$")
    plt.legend(loc='upper right')
    plt.savefig(dir_out + "errors.eps", bbox_inches='tight')


if args.convergence_degree or args.convergence_quadratic:
    convergence_degree()


def convergence_epsilon():

    # Initial condition
    t = quad.transform(ux*uy*uz, degree=degree, index_set=index_set)

    eye = quad.varf('1', degree=degree, index_set=index_set)
    dt, Ns, scheme = 2**-9, int(1e4), "backward"

    # Calculate white noise solution
    use_white = True
    kwargs = {'degree': degree, 'index_set': index_set}
    # forward0x = eq.Fokker_Planck_1d.equation({'Vp': Vp, 'β': β})
    # var0x = qx.discretize_op(forward0x, **kwargs)
    # _, [tx0] = var0x.eigs(k=1, which='LR')
    forward0 = forward.subs(γ, 1).subs(ε, 1)
    var0 = quad.discretize_op(forward0, **kwargs)
    _, [t30] = var0.eigs(k=1, which='LR')
    integral = float(Ix*(Iy*(Iz*t30)))
    t30, tx0 = t30 / integral, Iy*(Iz*t30) / integral

    εs = [2**(-i/4) for i in range(40)]
    e3, ex, t3, tx = [], [], [], []

    for iε, εi in enumerate(εs):

        forward_ε = forward.subs(γ, 0).subs(ε, εi)
        r_mat = quad.discretize_op(forward_ε, **kwargs)

        Δ, Δx = np.inf, np.inf
        for i in range(Ns):

            if args.interactive:
                plot(t)

            print("ε: " + str(εi) + ", i: " + str(i) + ", dt: " + str(dt)
                  + ", Δ: " + str(Δ) + ", Δx: " + str(Δx))

            # Backward Euler
            if scheme == 'backward':
                total_op = eye - dt*r_mat
                new_t = total_op.solve(t)

            # Crank-Nicholson
            if scheme == 'crank':
                crank_left = eye - dt*r_mat*(1/2)
                crank_right = eye + dt*r_mat*(1/2)
                new_t = crank_left.solve(crank_right(t))

            # Normalization
            new_t = new_t / float(Ix*(Iy*(Iz*new_t)))

            # Error
            Δ = np.sqrt(((t - new_t)*(t - new_t)).coeffs[0])/dt

            # Time adaptation
            threshold = .1
            dt_max = 64
            if Δ*dt > 2*threshold or dt > dt_max:
                dt = dt / 2.
                continue
            elif Δ*dt < threshold and dt < dt_max:
                dt = dt * 2.
            t = new_t

            converged = 1e-12
            if Δ < converged:
                break

        if args.interactive:
            plot(t)

        t3.append(t)
        tx.append(Iy*(Iz*t))
        error3, errorx = t30 - t3[-1], tx0 - tx[-1]
        e3.append(qxy.norm(Iy*error3, n=1, flat=True))
        ex.append(qx.norm(errorx, n=1, flat=True))

        if iε > 1:
            logε = np.log2(εs[0:iε+1])
            coeffs = np.polyfit(logε, np.log2(e3), 1)
            print(coeffs)
            coeffs_x = np.polyfit(logε, np.log2(ex), 1)
            print(coeffs_x)

    # Use last estimation as exact solution and remove it
    if not use_white:
        rem = 8
        t30, tx0, t3, tx = t3[-1], tx[-1], t3[0:-rem], tx[0:-rem]
        εs, ex, e3 = εs[0:-rem], ex[0:-rem], e3[0:-rem]
        ex = [qx.norm(tx0 - txi, n=1, flat=True) for txi in tx]
        e3 = [qxy.norm(Iy*(t30 - t3i), n=1, flat=True) for t3i in t3]

    ax = plt.subplots()[1]
    cmap = matplotlib.cm.get_cmap('viridis_r')
    for i, (εi, txε) in enumerate(zip(εs, tx)):
        if i % 2 is not 0:
            continue
        label = "$\\varepsilon = 2^{{ -{} }} $".format(i/4)
        kwargs = {'color': cmap(εi), 'label': label}
        qx.plot(txε, ax=ax, **kwargs)
    ax.set_title("")
    plt.legend()
    plt.savefig(dir_out + "convergence-epsilon-" + str(β) + ".eps",
                bbox_inches='tight')

    fig, ax = plt.subplots()
    xplot = logε = np.asarray(εs)
    yplot1 = np.asarray(e3)
    yplot2 = np.asarray(ex)
    np.save(dir_out + "convergence-eps-eps", xplot)
    np.save(dir_out + "convergence-eps-e3", yplot1)
    np.save(dir_out + "convergence-eps-ex", yplot2)
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)
    ax.set_xlabel('$\\varepsilon$')
    ax.plot(xplot, yplot1, 'b.', label="$|\\rho^{{x,q}} - \\rho^{{x,q}}_0|_1$")
    ax.plot(xplot, yplot2, 'r.', label="$|\\rho^x - \\rho^x_0|_1$")
    coeffs = np.polyfit(np.log2(xplot[5:]), np.log2(yplot1[5:]), 1)
    ax.plot(xplot, 2**coeffs[1] * xplot**coeffs[0], 'b-',
            label='$y = {:.2f} \\, \\times \\, \\varepsilon^{{ {:.2f} }}$'.
            format(2**coeffs[1], coeffs[0]))
    coeffs = np.polyfit(np.log2(xplot[7:]), np.log2(yplot2[7:]), 1)
    ax.plot(xplot, 2**coeffs[1] * xplot**coeffs[0], 'r-',
            label='$y = {:.2f} \\, \\times \\, \\varepsilon^{{ {:.2f} }}$'.
            format(2**coeffs[1], coeffs[0]))
    plt.legend(loc='lower right')
    plt.savefig(dir_out + "errors-epsilon-" + str(β) + ".eps",
                bbox_inches='tight')
    plt.show()


if args.convergence_epsilon:
    convergence_epsilon()


def bifurcation():

    kwargs = {'degree': degree, 'index_set': index_set}
    m_operator = forward.diff(m)
    r_operator = (forward - m*m_operator).cancel()
    m_mat = quad.discretize_op(m_operator, **kwargs)
    eye = quad.varf('1', **kwargs)

    # Initial condition
    βnum, mnum, betas, ms = βmax, 1, [], []
    t = quad.transform(ux.subs(β, βnum)*uy*uz, **kwargs)

    # Calculate effective diffusion
    # ! Different index_set !
    kwargs_langevin = {'degree': degree // 2, 'index_set': 'cube'}
    langevin = eq.Langevin.forward(1).subs(y, z).subs(x, y)
    mat = qyz.discretize_op(langevin, **kwargs_langevin)
    [l0], [e0] = mat.eigs(k=1, which='LR')
    mat = mat - np.real(l0)*qyz.varf('1', **kwargs_langevin)
    varz = qyz.varf('z', **kwargs_langevin)
    # solution = la.solve(mat.matrix[1:, 1:], (varz(e0)).coeffs[1:])
    # sol_series = qyz.series([0, *solution], index_set=index_set)
    sol_series = mat.solve(varz(e0))
    diff = - float(varz(e0)*sol_series)
    print("Effective diffusion coefficient: ", diff)

    dt, dt_max, Ns, scheme = 2**-9, 256, int(1e4), "backward"
    while βnum > βmin:

        r_operator_this = r_operator.subs(β, βnum)
        r_mat = quad.discretize_op(r_operator_this, **kwargs)

        Δ = np.inf
        for i in range(Ns):

            if args.interactive:
                plot(t)

            print("β: " + str(βnum) + ", i: " + str(i) + ", dt: " + str(dt)
                  + ", m: " + str(mnum) + ", Δ: " + str(Δ))

            operator = r_mat + mnum*m_mat

            # Backward Euler
            if scheme == 'backward':
                total_op = eye - dt*operator
                new_t = total_op.solve(t)

            # Crank-Nicholson
            if scheme == 'crank':
                crank_left = eye - dt*operator*(1/2)
                crank_right = eye + dt*operator*(1/2)
                new_t = crank_left.solve(crank_right(t))

            # Inverse power method
            # if scheme == 'inverse':
            #     new_t = operator.solve(t)

            # Normalization
            new_t = new_t / float(Ix*(Iy*(Iz*new_t)))
            new_m = float(mx1*(Iy*(Iz*new_t)))

            # Error
            Δ = np.sqrt(float((t - new_t)*(t - new_t)))/dt
            Δ = Δ + (i > 0)*abs(new_m - mnum)/dt

            # Time adaptation
            threshold = .1
            if Δ*dt > 2*threshold or dt > dt_max:
                dt = dt / 2.
                continue
            elif Δ*dt < threshold and dt < dt_max:
                dt = dt * 2.
            t, mnum = new_t, new_m

            if Δ < 1e-8:
                break

        betas.append(βnum)
        ms.append(mnum)

        gmm, dsdβ = 20, 1
        if len(ms) > 1:
            Δm, Δβ = ms[-1] - ms[-2], betas[-1] - betas[-2]
            dsdβ = np.sqrt(gmm*Δm*Δm + Δβ*Δβ) / abs(Δβ)
        newβ = βnum - sstep/dsdβ

        if math.floor(βnum) != math.floor(newβ):
            βnum = math.floor(βnum)

            plt.ioff()
            ax = plt.subplots(1, 1)[1]
            qxy.plot(Iy*t, bounds=False, ax=ax)
            plt.savefig(dir_out + 'solution-beta=' + str(βnum) + '.eps',
                        bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(1, 1)
            density = sym.exp(-βnum*(Vp + θ*(x - mnum)**2/2))
            density = density / qx.integrate(density, flat=True)
            qx.plot(density, ax=ax, label="White noise")
            qx.plot(Iy*(Iz*t), bounds=False, ax=ax,
                    label="$\\varepsilon = " + str(ε) + "$")
            plt.legend()
            plt.savefig(dir_out + 'solution-proj-beta=' + str(βnum) + '.eps',
                        bbox_inches='tight')
            plt.close()
            plt.ion()

        βnum = newβ

    np.save(dir_out + "epsilon=" + str(ε).replace('/', 'o') + "-betas", np.asarray(betas))
    np.save(dir_out + "epsilon=" + str(ε).replace('/', 'o') + "-ms", np.asarray(ms))


if args.bifurcation:
    bifurcation()
