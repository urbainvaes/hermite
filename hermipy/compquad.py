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

import hermipy.quad as quad


class CompQuad():

    def __init__(self, quads, qweights):

        if not len(quads) >= 1 or \
           not len(quads) == len(qweights):
            raise ValueError("Invalid arguments!")

        position0 = quads[0].position

        for q in quads:
            if not q.position == position0:
                raise ValueError("Invalid argument(s)!")

        self.quads = quads
        self.qweights = qweights
        self.position = position0

    # @classmethod
    # def smolyak(cls, n_points_max, **kwargs):

    def __eq__(self, other):

        if isinstance(other, quad.Quad):
            other = CompQuad([other], [1])

        if not isinstance(other, CompQuad):
            raise ValueError("Invalid argument!")

        if len(self.quads) != len(other.quads):
            return False

        return all(q1 == q2 for q1, q2 in zip(self.quads, other.quads))

    def integrate(self, f):
        result = 0
        for i, q in enumerate(self.quads):
            result += q.integrate(f) * self.qweights[i]
        return result

    def transform(self, f_grid, degree, index_set="triangle"):
        result = 0
        for i, q in enumerate(self.quads):
            series = q.transform(f_grid, degree, index_set=index_set)
            result += series * self.qweights[i]
        return result

    def varf(self, f_grid, degree, sparse=None, index_set="triangle"):
        result = 0
        for i, q in enumerate(self.quads):
            varf = q.varf(f_grid, degree, sparse=sparse, index_set=index_set)
            result += varf * self.qweights[i]
        return result

    def varfd(self, function, degree, directions, sparse=None,
              index_set="triangle"):
        return quad.Quad.varfd(self, function, degree, directions,
                               sparse=sparse, index_set=index_set)

    def discretize_op(self, op, func, degree, order,
                      sparse=None, index_set="triangle"):
        return quad.Quad.discretize_op(self, op, func, degree, order,
                                       sparse=sparse, index_set = index_set)

    def project(self, projection):
        new_quads = [q.project(directions) for q in self.quads]
        return CompQuad(new_quads, self.qweights)

# vim: foldmethod=marker foldnestmax=2
