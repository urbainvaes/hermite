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

import hermipy.function as func
import numpy as np
import numpy.linalg as la
import sympy as sym


class Position:

    very_small = 1e-10

    @staticmethod
    def tensorize(args):

        # Check that directions appear at most twice
        dirs = []
        for a in args:
            if not isinstance(a, Position) or not a.is_diag:
                raise ValueError("Invalid arguments")
            for d in a.dirs:
                if not dirs.count(d) <= 1:
                    raise ValueError("Direction appears more than once")
                dirs.append(d)

        def _tensorize(p1, p2):
            diff1 = [d for d in p1.dirs if d not in p2.dirs]
            diff2 = [d for d in p2.dirs if d not in p1.dirs]
            dirs_result = sorted(diff1 + diff2)
            mean, cov, types = [], [], []

            # Check removed directions match
            for d in [d for d in p1.dirs if d in p2.dirs]:
                i1, i2 = p1.dirs.index(d), p2.dirs.index(d)
                assert p1.mean[i1] == p2.mean[i2]
                assert p1.cov[i1][i1] == p2.cov[i2][i2]

            # Tensorization
            for d in dirs_result:
                pos = p1 if d in p1.dirs else p2
                index = pos.dirs.index(d)
                mean.append(pos.mean[index])
                cov.append(pos.cov[index][index])
                types.append(pos.types[index])
            return Position(dirs=dirs_result, mean=mean,
                            cov=np.diag(cov), types=types)

        result = args[0]
        for a in args[1:]:
            result = _tensorize(result, a)

        return result

    def __init__(self, dim=None, mean=None, cov=None, dirs=None, types=None):

        if mean is not None:
            self.dim = len(mean)
        elif cov is not None:
            self.dim = len(cov)
        elif dirs is not None:
            self.dim = len(dirs)
        elif dim is not None:
            self.dim = dim
        else:
            raise ValueError("All args are None!")

        # Checks
        if mean is not None and not self.dim == len(mean) or \
           cov is not None and not self.dim == len(cov) or \
           dirs is not None and not self.dim == len(dirs) or \
           dim is not None and not self.dim == dim or \
           types is not None and not self.dim == len(types):
            raise ValueError("Invalid arguments dim/dirs")

        # Defaults to first directions
        self.dirs = list(range(dim)) if dirs is None else dirs

        self.mean = np.zeros(self.dim) if mean is None \
            else np.asarray(mean, float)

        self.cov = np.eye(self.dim) if cov is None \
            else np.asarray(cov, float)

        self.types = ["hermite"]*self.dim if types is None else types

        eigval, eigvec = la.eig(self.cov)
        self.factor = np.matmul(eigvec, np.sqrt(np.diag(eigval)))

        diag_cov = np.diag(np.diag(self.cov))

        self.is_diag = True
        if len(self.dirs) > 0 and la.norm(self.cov - diag_cov, 2) > 1e-10:
            self.is_diag = False

    def __eq__(self, other):
        return self.dirs == other.dirs \
            and la.norm(self.mean - other.mean, 2) < self.very_small \
            and la.norm(self.cov - other.cov, 2) < self.very_small \
            and self.types == other.types

    def __mul__(self, other):
        if not isinstance(other, Position):
            raise ValueError("Invalid argument")
        return Position.tensorize([self, other])

    def __repr__(self):
        result = "Dimension: " + str(self.dim) + "\n"
        result += "Directions: " + str(self.dirs) + "\n"
        result += "Mean: " + str(self.mean) + "\n"
        result += "Covariance:\n" + str(self.cov) + "\n"
        result += "Types:\n" + str(self.types)
        return result

    def weight(self):
        ind = [i for i, t in enumerate(self.types) if t == "hermite"]
        if len(ind) == 0:
            return sym.Rational(1)
        sub_mean, sub_cov = self.mean[ind], self.cov[np.ix_(ind, ind)]
        var = [func.Function.xyz[self.dirs[i]] for i in ind]
        inv_cov = la.inv(sub_cov)
        potential = 0.5 * inv_cov.dot(var - sub_mean).dot(var - sub_mean)
        normalization = 1/(sym.sqrt((2*sym.pi)**self.dim * la.det(sub_cov)))
        return normalization * sym.exp(-potential)

    def weights(self):
        if not self.is_diag:
            raise ValueError("Invalid: is_diag must be True")
        return [self.project(d).weight() for d in self.dirs]

    def project(self, directions):
        if isinstance(directions, int):
            directions = [directions]
        if not self.is_diag or not directions == sorted(directions):
            raise ValueError("Invalid arguments")
        dirs, dim = directions, len(directions)
        mean, cov, types = np.zeros(dim), np.zeros((dim, dim)), [""]*dim
        rel_dirs = [self.dirs.index(d) for d in directions]
        for rp, rs in enumerate(rel_dirs):
            mean[rp] = self.mean[rs]
            cov[rp][rp] = self.cov[rs][rs]
            types[rp] = self.types[rs]
        return Position(dim=dim, mean=mean, cov=cov, dirs=dirs, types=types)
