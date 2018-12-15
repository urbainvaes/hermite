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

import hermipy as hm
import hermipy.core as core
import hermipy.lib as lib
import hermipy.stats as stats

import numpy as np
import numpy.linalg as la
import sympy as sym


very_small = 1e-10


class Quad:

    @staticmethod
    def tensorize(args):
        if not len(args) > 0 or \
           not isinstance(args[0], Quad):
            raise ValueError("Invalid argument(s)!")
        position = hm.Position.tensorize([a.position for a in args])
        nodes, weights = [0]*len(position.dirs), [0]*len(position.dirs)
        factor = sym.Integer(1)
        for a in args:
            if not isinstance(a, Quad):
                raise ValueError("Invalid argument!")
            factor *= a.factor.sym
            for i, d in enumerate(a.position.dirs):
                nodes[position.dirs.index(d)] = a.nodes[i]
                weights[position.dirs.index(d)] = a.weights[i]
        factor = hm.Function(factor, dirs=position.dirs)
        return Quad(nodes, weights, position, factor)

    def __init__(self, nodes, weights, position=None, factor=1,
                 mean=None, cov=None, dirs=None, types=None):

        """Create a quadrature object

        Args:
            nodes: The nodes of the quadrature.
            weights: The weights of the quadrature, in the basis.

        Returns:
            True if successful, False otherwise.

        Note:
            Do not include the  parameter in the Args section.

        """

        #  FIXME: This requires same number of points in each direction
        self.nodes = np.asarray(nodes, float)
        self.weights = np.asarray(weights, float)

        dim = len(self.nodes)
        self.position = position if position is not None else \
            hm.Position(dim=dim, mean=mean, cov=cov, dirs=dirs, types=types)

        # Factor used for hermite transform, discretize
        self.factor = hm.Function(factor, dirs=self.position.dirs)
        self.factor.sym = self.factor.sym.expand()
        self.do_fourier = [int(t == "fourier") for t in self.position.types]

    @classmethod
    def gauss_hermite(cls, n_points, dim=None, dirs=None, **kwargs):

        if dirs is not None:
            dim = len(dirs)

        if dim is not None:
            n_points = np.full(dim, n_points)

        elif isinstance(n_points, int):
            n_points = [n_points]

        nodes, weights = lib.hermegauss_nd(n_points)
        return cls(nodes, weights, dirs=dirs, **kwargs)

    @classmethod
    def fourier(cls, n_points, bounds=None,
                position=None, factor=1, dirs=None):

        # Determine dimension
        if position is not None:
            dim = position.dim
        if dirs is not None:
            dim = len(dirs)
        elif bounds is not None:
            dim = len(bounds)
        elif not isinstance(n_points, int):
            dim = len(n_points)
        else:
            dim = 1

        # Construct position if undefined
        if position is None:

            # Default arguments
            if bounds is None:
                bounds = [(-np.pi, np.pi)]*dim
            if dirs is None:
                dirs = list(range(dim))

            assert dim == len(bounds)
            assert dim == len(dirs)

            mean, cov = np.zeros(dim), np.zeros((dim, dim))
            for i in range(dim):
                left, right = bounds[i][0], bounds[i][1]
                # FIXME: The notation mean / cov is not adequate for Fourier
                # series. At the moment this is just a hack
                mean[i] = left/2 + right/2
                cov[i][i] = ((right - left)/(2*np.pi))**2
            position = hm.Position(dim=dim, dirs=dirs, mean=mean, cov=cov,
                                   types=["fourier"]*dim)

        if isinstance(n_points, int):
            n_points = [n_points]*dim
        assert dim == len(n_points)
        nodes, weights = [], []
        for i in range(dim):
            nodes.append(-np.pi + 2*np.pi*np.arange(n_points[i])/n_points[i])
            weights.append(2*np.pi/n_points[i]*np.ones(n_points[i]))
        return cls(nodes, weights, position=position, factor=factor)

    @classmethod
    def newton_cotes(cls, n_points, extrema, **kwargs):
        nodes, weights = [], []
        for i, extremum in enumerate(extrema):
            nodes.append(np.linspace(-extremum, extremum, n_points[i]))
            mesh_size = 2*extremum/(n_points[i] - 1)
            weights_simpson = np.zeros(n_points[i]) + 1
            weights_simpson[0], weights_simpson[-1] = .5, .5
            gaussian_weight = 1/np.sqrt(2*np.pi) * np.exp(-nodes[-1]**2/2.)
            weights.append(weights_simpson * gaussian_weight * mesh_size)
        return cls(nodes, weights, **kwargs)

    def __mul__(self, other):
        return Quad.tensorize([self, other])

    def __eq__(self, other):
        if not isinstance(other, Quad):
            raise ValueError("Invalid argument!")

        return self.position == other.position \
            and self.factor == other.factor \
            and la.norm(self.nodes - other.nodes) < very_small \
            and la.norm(self.weights - other.weights) < very_small

    def __str__(self):
        return "Quadrature in dimension " + str(self.position.dim) \
               + "\n- Associated factor: " + str(self.factor) \
               + "\n- Types: " + str(self.position.types)

    def __repr__(self):
        return self.__str__()

    def tensorize_at(arg_num: int):
        def tensorize_arg(func):
            def wrapper(*args, **kwargs):
                tensorize = kwargs['tensorize'] if 'tensorize' in kwargs \
                        else None
                do_tensorize = hm.settings['tensorize'] if \
                    tensorize is None else tensorize

                # <- Fix bug with integrate(series)
                if isinstance(args[arg_num], hm.Series):
                    do_tensorize = False
                # ->

                if not do_tensorize:
                    return func(*args, **kwargs)

                quad, function = args[0], args[arg_num]
                if not quad.position.is_diag or isinstance(function, np.ndarray):
                    return func(*args, **kwargs)

                results = []
                if not isinstance(function, hm.Function):
                    function = hm.Function(function, dirs=quad.position.dirs)

                for add in function.split(legacy=False):
                    multiplicator = add[frozenset()]
                    del add[frozenset()]
                    func_dirs = {}
                    for dirs, term in add.items():
                        new_args = list(args).copy()
                        new_args[0] = quad.project(list(dirs))
                        new_args[arg_num] = term
                        func_dirs[dirs] = func(*new_args, **kwargs)

                    if hm.settings['debug']:
                        print("Tensorizing results")

                    kwargs_func = {'sparse': kwargs['sparse']} \
                        if 'sparse' in kwargs else {}
                    t = type(list(func_dirs.values())[0])
                    if t is hm.Varf or t is hm.Series:
                        values = list(func_dirs.values())
                        tensorized = t.tensorize(values, **kwargs_func)
                    else:
                        tensorized = core.tensorize(func_dirs, **kwargs_func)
                    results.append(tensorized*float(multiplicator.sym))
                return sum(results[1:], results[0])
            return wrapper
        return tensorize_arg

    def mapped_nodes(self):
        coords_nodes = []
        for i in range(self.position.dim):
            coord = 'x_{}'.format(i)
            coords_nodes.append(self.discretize(coord))
        return np.asarray(np.vstack(coords_nodes)).T

    def discretize(self, f):
        if not isinstance(f, hm.Function):
            f = hm.Function(f, dirs=self.position.dirs)
        function = core.discretize(f.ccode(), self.nodes,
                                   self.position.mean, self.position.factor)
        return function

    @tensorize_at(1)
    def integrate(self, function, flat=False, tensorize=None):

        if isinstance(function, hm.Series):
            function = self.eval(function)

        if not isinstance(function, np.ndarray):
            function = self.discretize(function)

        if flat:
            w_grid = self.discretize(self.position.weight())

            # Potentially not robust!
            # Zero the nans
            w_grid = 1e-300*(w_grid == 0) + w_grid
            function = function / w_grid

        return core.integrate(function, self.nodes, self.weights,
                              do_fourier=self.do_fourier)

    # Norm 1 or 2, in weighted or not
    def norm(self, function, n=2, flat=False):

        if isinstance(function, hm.Series):
            function = self.eval(function)

        if n is 2:
            return np.sqrt(self.integrate(function**2, flat=flat))
        elif n is 1:
            return self.integrate(abs(function), flat=flat)

    @tensorize_at(1)
    @stats.debug()
    @stats.log_stats()
    def transform(self, function, degree, index_set="triangle",
                  significant=0, tensorize=None):

        if not isinstance(function, np.ndarray):
            function = hm.Function(function, dirs=self.position.dirs)
            function = self.discretize(function)

        factor = self.discretize(self.factor)
        mapped = function / (1e-300*(factor == 0) + factor)

        coeffs = core.transform(degree, mapped, self.nodes, self.weights,
                                forward=True, do_fourier=self.do_fourier,
                                index_set=index_set)

        return hm.Series(coeffs, self.position,
                         factor=self.factor, index_set=index_set,
                         significant=significant)

    def eval(self, series):

        if isinstance(series, np.ndarray):
            series = hm.Series(series, self.position, factor=self.factor)

        degree, coeffs = series.degree, series.coeffs

        inv = la.inv(series.position.factor)
        translation = inv.dot(self.position.mean - series.position.mean)
        factor = inv * self.position.factor
        if la.norm(factor - np.diag(np.diag(factor)), 2) > 1e-8:
            raise ValueError("Incompatible covariance matrices")
        mapped_nodes = self.nodes.copy()
        for i, n in enumerate(self.nodes):
            mapped_nodes[i] = n * factor[i][i] + translation[i]

        result = core.transform(degree, coeffs, mapped_nodes, self.weights,
                                forward=False, do_fourier=self.do_fourier,
                                index_set=series.index_set)

        return result*self.discretize(series.factor)

    @tensorize_at(1)
    @stats.debug()
    @stats.log_stats()
    def varf(self, f_grid, degree, sparse=None,
             index_set="triangle", tensorize=None):
        sparse = hm.settings['sparse'] if sparse is None else sparse
        if not isinstance(f_grid, np.ndarray):
            f_grid = self.discretize(f_grid)
        var = core.varf(degree, f_grid, self.nodes, self.weights,
                        sparse=sparse, do_fourier=self.do_fourier,
                        index_set=index_set)
        return hm.Varf(var, self.position,
                       factor=self.factor, index_set=index_set)

    @tensorize_at(1)
    @stats.debug()
    @stats.log_stats()
    def varfd(self, function, degree, directions,
              sparse=None, index_set="triangle", tensorize=None):
        directions = filter(lambda d: d in self.position.dirs, directions)
        sparse = hm.settings['sparse'] if sparse is None else sparse
        var = self.varf(function, degree, sparse=sparse,
                        index_set=index_set, tensorize=False)
        mat = var.matrix
        eigval, _ = la.eig(self.position.cov)
        for d in directions:
            rel_dir = self.position.dirs.index(d)
            mat = core.varfd(self.position.dim, degree, rel_dir,
                             mat, do_fourier=self.do_fourier[rel_dir],
                             index_set=index_set)
            mat = mat/np.sqrt(eigval[rel_dir])
        return hm.Varf(mat, self.position,
                       factor=self.factor, index_set=index_set)

    @stats.debug()
    @stats.log_stats()
    def discretize_op(self, op, degree,
                      sparse=None, index_set="triangle"):

        if not isinstance(op, hm.Operator):
            op = hm.Operator(op, dirs=self.position.dirs)

        if self.factor != hm.Function(1, dirs=self.position.dirs):
            op = op.map(self.factor)

        if not op.dirs == self.position.dirs:
            raise ValueError("Invalid argument: directions don't match")
        splitop = op.split()
        sparse = hm.settings['sparse'] if sparse is None else sparse

        if splitop == {}:
            return self.varf(0, degree, sparse=sparse, index_set=index_set)

        varf_operator = 0
        for m, coeff in splitop.items():
            enum_dirs = enumerate(self.position.dirs)
            d_vector = sum([[d]*m[i] for i, d in enum_dirs], [])
            varf_part = self.varfd(coeff, degree, d_vector, sparse=sparse,
                                   index_set=index_set)
            varf_operator = varf_operator + varf_part
        return varf_operator

    # Only works with ints
    def project(self, directions):
        """Project the quadrature.

        Args:
            directions: a list of integers containing the directions along
            which the quadrature is to be projected.

        Returns:
            A `Quad` object.

        """
        factor = self.factor.project(directions)
        position = self.position.project(directions)
        nodes, weights = [], []
        for d in position.dirs:
            nodes.append(self.nodes[self.position.dirs.index(d)])
            weights.append(self.weights[self.position.dirs.index(d)])
        return Quad(nodes, weights, position=position, factor=factor)

    def series(self, coeffs, index_set="triangle"):
        """Return a series from a vector of coefficients.

        Args:
            coeffs: a numpy array of coefficients.

            index_set: the index set corresponding to the series.

        Returns:
            A `Series` object.

        """
        return hm.Series(coeffs, self.position,
                         factor=self.factor, index_set=index_set)

    def plot_hf(self, multi_index, ax=None, bounds=True, **kwargs):

        if not len(multi_index) == self.position.dim:
            raise ValueError("Invalid argument!")
        dim = len(multi_index)

        show_plt = ax is None
        if show_plt:
            import matplotlib.pyplot as plt
            ax = plt.subplots(1)[1]

        deg_max, index_set = sum(multi_index), "triangle"
        hf = np.zeros(core.iterator_size(dim, deg_max, index_set=index_set))
        hf[core.iterator_index(multi_index, index_set="triangle")] = 1
        hf = self.series(hf, index_set=index_set)
        plot = self.plot(hf, ax=ax, bounds=False, **kwargs)

        # For hermite functions
        if bounds:
            adim_width = [np.sqrt(2) * np.sqrt(2*d + 1) for d in multi_index]

            pos, bounds = self.position, []
            for i in range(dim):
                width = adim_width[i] * np.sqrt(pos.cov[i][i])
                bounds.append([pos.mean[i] - width, pos.mean[i] + width])

            if dim >= 1:
                ax.axvline(x=bounds[0][0])
                ax.axvline(x=bounds[0][1])

            if dim == 2:
                ax.axhline(y=bounds[1][0])
                ax.axhline(y=bounds[1][1])

        if show_plt:
            if self.position.dim == 2:
                plt.colorbar(plot, ax=ax)
            plt.show()
        else:
            return plot

    def plot(self, arg, factor=None, ax=None,
             contours=0, bounds=False, title=None, **kwargs):

        if not self.position.is_diag:
            raise ValueError("Invalid argument: position must be diag!")

        show_plt = ax is None
        if show_plt:
            import matplotlib.pyplot as plt
            ax = plt.subplots(1)[1]

        if isinstance(arg, tuple(sym.core.all_classes)):
            arg = hm.Function(arg, dirs=self.position.dirs)

        if isinstance(arg, hm.Function):
            if factor is not None:
                raise ValueError("Invalid argument!")
            solution = self.discretize(arg)

        elif isinstance(arg, np.ndarray):
            solution = arg

        elif isinstance(arg, hm.Series):

            series = arg

            # Fix me maybe?
            if series.coeffs.dtype == np.dtype('complex128'):
                series.coeffs = np.real(series.coeffs).copy(order='C')

            if factor is None:
                solution = self.eval(series)

            elif not isinstance(factor, np.ndarray):
                factor = hm.Function(factor, dirs=self.position.dirs)
                factor = self.discretize(factor)
                solution = self.eval(series)*factor

            if bounds:

                degree_bounds = series.degree
                bounds, adim_width = [], np.sqrt(2)*np.sqrt(2*degree_bounds+1)
                for i in range(series.position.dim):
                    mean = series.position.mean[i]
                    cov = series.position.cov[i][i]
                    bounds.append([mean - adim_width * np.sqrt(cov),
                                   mean + adim_width * np.sqrt(cov)])

                if self.position.dim >= 1:
                    ax.axvline(x=bounds[0][0])
                    ax.axvline(x=bounds[0][1])
                if self.position.dim == 2:
                    ax.axhline(y=bounds[1][0])
                    ax.axhline(y=bounds[1][1])

        else:
            raise TypeError("Unsupported type: " + str(type(arg)))

        n_nodes = []
        r_nodes = []
        for i, d in enumerate(self.position.dirs):
            direction = hm.Function.xyz[d]
            n_nodes.append(len(self.nodes[i]))
            r_nodes.append(self.project(d).discretize(direction))
        solution = solution.reshape(*n_nodes).T

        if self.position.dim == 1:
            plot = ax.plot(*r_nodes, solution, **kwargs)
        elif self.position.dim == 2:
            plot = ax.contourf(*r_nodes, solution, 100, **kwargs)
            for c in plot.collections:
                c.set_edgecolor("face")
            if contours > 0:
                ax.contour(*r_nodes, solution, levels=contours, colors='k')

        _min, _max = np.min(solution), np.max(solution)

        if title is None:
            title = "Min: {:.3f}, Max: {:.3f}".format(_min, _max)
        ax.set_title(title)

        if show_plt:
            if self.position.dim == 2:
                plt.colorbar(plot, ax=ax)
            plt.show()
        else:
            return plot

    def streamlines(self, fx, fy, factor=None, ax=None, **kwargs):
        if not self.position.is_diag or \
           not isinstance(fx, type(fy)) or \
           self.position.dim is not 2:
            raise ValueError("Invalid argument(s)!")

        show_plt = ax is None
        if show_plt:
            import matplotlib.pyplot as plt
            ax = plt.subplots(1)[1]

        if isinstance(fx, tuple(sym.core.all_classes)):
            fx = hm.Function(fx, dirs=self.position.dirs)
            fy = hm.Function(fy, dirs=self.position.dirs)

        if isinstance(fx, hm.Function):
            if factor is not None:
                raise ValueError("Invalid argument!")
            fx, fy = self.discretize(fx), self.discretize(fy)

        elif isinstance(fx, hm.Series):

            if factor is None:
                sx, sy = self.eval(fx), self.eval(fy)

            elif not isinstance(factor, np.ndarray):
                factor = hm.Function(factor, dirs=self.position.dirs)
                factor = self.discretize(factor)
                sx, sy = self.eval(fx)*factor, self.eval(fy)*factor

        else:
            raise TypeError("Unsupported type: " + str(type(fx)))

        n_nodes = []
        r_nodes = []
        for i in range(self.position.dim):
            direction = hm.Function.xyz[self.position.dirs[i]]
            n_nodes.append(len(self.nodes[i]))
            r_nodes.append(self.project(i).discretize(direction))
        sx, sy = sx.reshape(*n_nodes).T, sy.reshape(*n_nodes).T
        # magnitude = sx**2 + sy**2
        x, y = r_nodes[0], r_nodes[1]

        # See https://github.com/matplotlib/matplotlib/issues/9269
        def calc_psi2d(x, y, fx, fy):  # solenoidal flows
            # psi by integrating dpsi = -fy*dx + fx*dy, psi[0,0]=0
            ny, nx = fx.shape
            psi = np.zeros((ny, nx))
            for jx in range(1, nx):
                psi[0, jx] = psi[0, jx-1] - fy[0, jx] * (x[jx] - x[jx-1])
            for jy in range(1, ny):
                psi[jy, :] = psi[jy-1, :] + fx[jy, :] * (y[jy] - y[jy-1])
            return psi

        psi = calc_psi2d(x, y, sx, sy)
        _min, _max, thresh = np.min(psi), np.max(psi), .1
        if abs(_min) < thresh*(_max - _min):
            _min = thresh*(_max - _min)
        if abs(_max) < thresh*(_max - _min):
            _max = - thresh*(_max - _min)
        levels = np.linspace(_min, _max, 10)
        streams = ax.contour(x, y, psi, levels=levels, **kwargs)

        # ax.quiver(x, y, sx, sy)
        # streams = ax.streamplot(r_nodes[0], r_nodes[1], sx, sy,
        #                         color=magnitude, density=0.6, cmap='autumn')

        return plt.show() if show_plt else streams

# vim: foldmethod=marker
