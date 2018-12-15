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

import hermipy.core as core
import hermipy.lib as lib
import hermipy.position as pos
import hermipy.function as func

import numpy as np
import numpy.linalg as la

from matplotlib.ticker import MaxNLocator


very_small = 1e-10


class Series:

    @staticmethod
    def tensorize(args):

        def _tensorize(s1, s2):
            if not isinstance(s1, Series) or not isinstance(s2, Series):
                raise ValueError("s1 and s2 have to be Series")

            if s1.position.dim == 0:
                return Series(s1.coeffs[0]*s2.coeffs, s2.position,
                              factor=s2.factor, index_set=s2.index_set)

            if s2.position.dim == 0:
                return Series(s2.coeffs[0]*s1.coeffs, s1.position,
                              factor=s1.factor, index_set=s1.index_set)

            # Inner product only makes sense in weighted space
            # Product in appropriate weighted space
            common = set(s1.position.dirs).intersection(s2.position.dirs)
            f1 = s1.factor.project(list(common))
            f2 = s2.factor.project(list(common))

            if not f1 == f2:
                raise ValueError("f1 == f2 should be True")

            if not s1.degree == s2.degree:
                raise ValueError("s1 and s2 should have the same degree")

            if not s1.index_set == s2.index_set:
                raise ValueError("s1 and s2 should have the same index_set")

            f1, f2 = s1.factor.sym/f1.sym, s2.factor.sym/f2.sym
            d1, d2 = s1.position.dirs, s2.position.dirs
            c1, c2 = s1.coeffs, s2.coeffs
            result = core.inner(c1, c2, d1, d2, index_set=s1.index_set)
            position = pos.Position.tensorize([s1.position, s2.position])
            factor = func.Function(f1*f2, dirs=position.dirs)
            return Series(result, position,
                          factor=factor, index_set=s1.index_set)

        result = args[0]
        for a in args[1:]:
            result = _tensorize(result, a)

        return result

    def __init__(self, coeffs, position,
                 factor=1, index_set="triangle", significant=0):

        self.coeffs = coeffs
        self.position = position
        self.index_set = index_set

        dim, npolys = self.position.dim, len(self.coeffs)

        if position.dirs == []:
            self.degree = 0
        else:
            self.degree = core.iterator_get_degree(dim, npolys,
                                                   index_set=index_set)

            assert len(self.multi_indices()) == len(self.coeffs)

        if significant is not 0:
            for i, c in enumerate(self.coeffs):
                self.coeffs[i] = round(c, significant)

        self.factor = func.Function(factor, dirs=self.position.dirs)

        if not self.coeffs.flags['C_CONTIGUOUS']:
            self.coeffs = self.coeffs.copy(order='C')

    def __eq__(self, other):
        if not isinstance(other, Series):
            raise ValueError("Invalid argument")
        return self.position == other.position \
            and self.factor == other.factor \
            and la.norm(self.coeffs - other.coeffs) < very_small

    def __add__(self, other):

        if isinstance(other, (int, float, np.float64)):
            new_coeffs = self.coeffs + other

        elif isinstance(other, Series):
            if not self.position == other.position or \
               not self.index_set == other.index_set or \
               not self.factor == other.factor:
                raise ValueError("Invalid arguments")
            new_coeffs = self.coeffs + other.coeffs

            #  TODO: Add support for addition of different degrees / index sets

        else:
            raise TypeError("Invalid type!)")

        return Series(new_coeffs, self.position,
                      factor=self.factor, index_set=self.index_set)

    def __mul__(self, other):

        if isinstance(other, (int, float, np.float64)):
            new_coeffs = self.coeffs * other
            return Series(new_coeffs, self.position,
                          factor=self.factor, index_set=self.index_set)

        elif isinstance(other, Series):
            if not self.index_set == other.index_set:
                raise ValueError("Index sets must match")
            return Series.tensorize([self, other])

        else:
            raise TypeError("Invalid type: " + str(type(other)))

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + other * (-1)

    def __neg__(self):
        return self * (-1)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, np.float64)):
            raise ValueError("Invalid argument")
        new_coeffs = self.coeffs / other
        return Series(new_coeffs, self.position,
                      factor=self.factor, index_set=self.index_set)

    def __repr__(self):
        m_list = self.multi_indices()
        assert len(m_list) == len(self.coeffs)
        result = ""
        for m, c in zip(m_list, self.coeffs):
            result += str(m) + ": " + str(c) + "\n"
        return result

    def __float__(self):
        if self.position.dim is not 0:
            raise ValueError("Dimension must be 0")
        return float(self.coeffs[0])

    def __getitem__(self, index):
        return self.coeffs[index]

    def project(self, directions):
        if not isinstance(directions, list):
            directions = [directions]
        rel_dirs = [self.position.dirs.index(d) for d in directions]
        p_coeffs = core.project(self.coeffs, self.position.dim, rel_dirs,
                                index_set=self.index_set)
        p_pos = self.position.project(directions)
        factor = self.factor.project(directions)
        return Series(p_coeffs, p_pos, factor=factor,
                      index_set=self.index_set)

    def subdegree(self, degree):

        # At the moment, only works if index set is consistent
        if self.index_set == "cross_nc":
            raise ValueError("cross_nc not supported")
        n_polys = core.iterator_size(self.position.dim, degree,
                                     index_set=self.index_set)

        if degree <= self.degree:
            coeffs = self.coeffs[0:n_polys]
        else:
            zeros = np.zeros(n_polys - len(self.coeffs))
            coeffs = np.hstack([self.coeffs, zeros])

        return Series(coeffs, self.position,
                      factor=self.factor, index_set=self.index_set)

    def multi_indices(self):
        if self.position.dim == 0:
            return [None]
        return core.iterator_list_indices(self.position.dim, self.degree,
                                          index_set=self.index_set)

    def plot(self, ax=None, title=None):

        show_plt = ax is None
        if show_plt:
            import matplotlib.pyplot as plt
            ax = plt.subplots(1)[1]

        m = self.multi_indices()
        if self.position.dim is 1:
            mx = m[:, 0]
            pl = ax.bar(mx, self.coeffs)
        elif self.position.dim is 2:
            coeffs = self.coeffs
            # coeffs = self.coeffs / max(abs(self.coeffs))
            # coeffs = (abs(coeffs) > 1e-10) * coeffs
            mx, my = m[:, 0], m[:, 1]
            # zoom = 1e3/max(mx)
            # positives = zoom * (coeffs > 0) * coeffs
            # negatives = zoom * (coeffs < 0) * coeffs * (-1)
            # ax.scatter(mx, my, s=positives, c='g', marker='o')
            pl = ax.scatter(mx, my, c=abs(coeffs),
                            cmap='ocean_r', s=1e2, edgecolor='k')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            # zeros = abs(coeffs) < 1e-12
            # ax.scatter(mx, my, s=zeros, c='r', marker='o')

            # for i, txt in enumerate(coeffs):
            #     ax.annotate("{:.2f}".format(txt), (mx[i] + .1, my[i] +
            #     .1), size=8)

        if title is None:
            title = "Coefficients of the Hermite expansion"
        ax.set_title(title)

        if show_plt:
            if self.position.dim is 2:
                plt.colorbar(pl, ax=ax)
            plt.show()
        else:
            return pl

    def to_function(self):
        if not self.position.is_diag:
            raise ValueError("Position must be diagonal")

        def rec_a(i):
            return float(1/np.sqrt(i+1))

        def rec_b(i):
            return - float(np.sqrt(i)/np.sqrt(i+1))

        dirs = self.position.dirs
        hermite_dirs = []

        for i, d in enumerate(dirs):
            var = func.Function.xyz[d]
            h0 = func.Function('1', dirs=dirs)
            μ, σ = self.position.mean[i], self.position.cov[i][i]
            f1 = np.sqrt(σ) * (var - μ)
            h1 = func.Function(f1, dirs=dirs)
            hermite_dirs.append([h0, h1])
            for j in range(1, self.degree):
                hermite_dirs[i].append(hermite_dirs[i][-1]
                                       * hermite_dirs[i][1] * rec_a(j)
                                       + hermite_dirs[i][-2] * rec_b(j))

        result = func.Function('0', dirs=dirs)
        for im, m in enumerate(self.multi_indices()):
            result_m = func.Function(self.coeffs[im], dirs=dirs)
            for i in range(0, self.position.dim):
                result_m *= hermite_dirs[i][m[i]]
            result += result_m
        return result

    def to_cross(self, degree):
        if not self.index_set == "triangle" or \
           not degree + self.position.dim - 1 <= self.degree:
            raise ValueError("Invalid argument")
        coeffs = self.coeffs[lib.cross_in_triangle(self.position.dim, degree)]
        return Series(coeffs, self.position, index_set="cross")
