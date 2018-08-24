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

import argparse
import sympy as sym
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import hermipy as hm
import hermipy.equations as eq
import hermipy.core as core

hm.settings['tensorize'] = True
hm.settings['sparse'] = True
hm.settings['trails'] = False
hm.settings['cache'] = True

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--interactive', action='store_true')
args = parser.parse_args()

# Dimension of the problem
dim = 3

# Equation parameters
index_set = 'rectangle'
r = sym.Rational
equation = eq.McKean_Vlasov_harmonic_noise
x, y, z, f = equation.x, equation.y, equation.z, equation.f
params = {'β': r(5), 'ε': r(1, 5), 'γ': 0, 'θ': 0, 'm': 0}
Vp, β = x**4/4 - x**2/2, params['β']

# Numerical parameters
s2x, s2y, s2z, degree = r(1, 20), r(1, 5), r(1, 5), 50
n_points_num = 2*degree + 1

# Potential for approximation
Vqx = sym.Rational(1/2)*x*x/(β*s2x)
Vqy = sym.Rational(1/2)*y*y/s2y
Vqz = sym.Rational(1/2)*z*z/s2z

# Fokker Planck for McKean-Vlasov equation
params.update({'Vp': Vp})
forward = equation.equation(params)

# Map to appropriate space
factor_x = sym.exp(- β / 2 * (Vqx + Vp))
factor_pq = sym.exp(- 1/2 * (y*y/2 + Vqy + z*z/2 + Vqz))
factor = factor_x * factor_pq

# Calculation of the solution
new_q = hm.Quad.gauss_hermite
cov = [[s2x, 0, 0], [0, s2y, 0], [0, 0, s2z]]
quad = new_q(n_points_num, dim=3, mean=[0]*3, cov=cov, factor=factor)

import ipdb; ipdb.set_trace()
min_quad = new_q(n_points=degree+1, mean=[0]*3, cov=cov, factor=factor)

# Projections
qx, qy, qz = quad.project(0), quad.project(1), quad.project(2)
wx = qx.factor * qx.factor / qx.position.weight()
wy = qy.factor * qy.factor / qy.position.weight()
wz = qz.factor * qz.factor / qz.position.weight()
qxy, qxz = quad.project([0, 1]), quad.project([0, 2])

# Integral operators
Ix = qx.transform(wx, degree=degree, index_set=index_set)
Iy = qy.transform(wy, degree=degree, index_set=index_set)
Iz = qz.transform(wz, degree=degree, index_set=index_set)

# Moment operators
mx1 = qx.transform(wx * x, degree=degree, index_set=index_set)
my1 = qy.transform(wy * y, degree=degree, index_set=index_set)
mz1 = qz.transform(wz * z, degree=degree, index_set=index_set)

# Marginals in white noise case
ux = sym.exp(-β*Vp) / qx.integrate(sym.exp(-β*Vp), flat=True)
uy = sym.exp(-y*y/2) / qy.integrate(sym.exp(-y*y/2), flat=True)
uz = sym.exp(-z*z/2) / qz.integrate(sym.exp(-z*z/2), flat=True)

# Discretization of the operator
degrees = list(range(5, degree + 1))
mat = quad.discretize_op(forward, degrees[-1], index_set=index_set)
solutions, v0, eig_vec = [], None, None

if args.interactive:
    plt.ion()
    fig, ax = plt.subplots(2, 2)


def plot(t):

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
    qxz.plot(txz, ax=ax[0][0], bounds=False, vmin=0, extend='min')
    qxy.plot(txy, ax=ax[0][0], bounds=False, vmin=0, extend='min')

    ax[0][1].clear()
    qx.plot(ux, ax=ax[0][1])
    qx.plot(tx, ax=ax[0][1], title="1st moment - X: " + mx)

    ax[1][0].clear()
    qy.plot(uy, ax=ax[1][0])
    qy.plot(ty, ax=ax[1][0], title="1st moment - P: " + my)

    ax[1][1].clear()
    qz.plot(uz, ax=ax[1][1])
    qz.plot(tz, ax=ax[1][1], title="1st moment - Q: " + mz)

    plt.draw()
    plt.pause(0.1)


def convergence():

    eye = quad.varf('1', degree=degree, index_set=index_set)
    dt, Ns, scheme = 2**-9, int(1e4), "backward"
    dmin, d, degrees, errors = 10, degree, [], []

    # Initial condition
    t = quad.transform(ux*uy*uz, degree=degree, index_set=index_set)

    while d >= dmin:
        # import ipdb; ipdb.set_trace()
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

                if d == degree:
                    t_exact = t

                else:
                    errors.append(t.subdegree(degree) - t_exact)
                    # print(errors[-1])

                if args.interactive:
                    plot(t)

                d = d - 1
                break

    import ipdb; ipdb.set_trace()


convergence()


if args.interactive:
    plt.ion()
    fig, ax = plt.subplots(2, 2)

for d in degrees:
    print(d)
    npolys = core.iterator_size(dim, d, index_set=index_set)
    if d is not degrees[0]:
        v0 = np.zeros(npolys)
        for i in range(len(eig_vec.coeffs)):
            v0[i] = eig_vec.coeffs[i]
    sub_mat = mat.subdegree(d)
    eig_vals, [eig_vec] = sub_mat.eigs(k=1, v0=v0, which='LR')
    t = eig_vec * np.sign(eig_vec.coeffs[0])

    t = t / quad.integrate(t, flat=True, tensorize=False)
    solutions.append(t)

    if args.interactive:

        Ix_d = Ix.subdegree(d)
        Iy_d = Iy.subdegree(d)
        Iz_d = Iz.subdegree(d)

        # Projection on x-q subspace
        integral = (Ix_d*(Iy_d*(Iz_d*t))).coeffs[0]
        txy, txz = Iz_d*t, Iy_d*t

        # Projections
        tx, ty, tz = Iy_d*(Iz_d*t), Ix_d*(Iz_d*t), Ix_d*(Iy_d*t)

        # Moments
        mx = str((mx1.subdegree(d)*tx).coeffs[0])
        my = str((my1.subdegree(d)*ty).coeffs[0])
        mz = str((mz1.subdegree(d)*tz).coeffs[0])

        ax[0][0].clear()
        qxz.plot(txz, ax=ax[0][0], bounds=False, vmin=0, extend='min')
        qxy.plot(txy, ax=ax[0][0], bounds=False, vmin=0, extend='min')

        ax[0][1].clear()
        qx.plot(ux, ax=ax[0][1])
        qx.plot(tx, ax=ax[0][1], title="1st moment - X: " + mx)

        ax[1][0].clear()
        qy.plot(uy, ax=ax[1][0])
        qy.plot(ty, ax=ax[1][0], title="1st moment - P: " + my)

        ax[1][1].clear()
        qz.plot(uz, ax=ax[1][1])
        qz.plot(tz, ax=ax[1][1], title="1st moment - Q: " + mz)

        plt.draw()
        plt.pause(.01)


finest = ground_state
finest_eval = solutions[-1]

# Plot of the finest solution
fig, ax = plt.subplots(1, 1)
quad.plot(finest, factor, ax=ax)
plt.show()

# Associated errors
errors, degrees = [], degrees[0:-1]
for sol in solutions[0:-1]:
    error = quad.norm(sol - finest_eval, flat=True)
    errors.append(error)
    print(error)

log_errors = np.log(errors)
poly_approx = np.polyfit(degrees, log_errors, 1)
errors_approx = np.exp(np.polyval(poly_approx, degrees))
error = la.norm(log_errors - np.log(errors_approx), 2)

plt.semilogy(degrees, errors, 'k.')
plt.semilogy(degrees, errors_approx)
plt.show()