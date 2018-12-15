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
import sympy
import itertools
import math


# Linear operator!
class Operator():

    f = sympy.Function('f')

    def __init__(self, expr, dirs=None):

        if isinstance(expr, Operator):
            self.sym = expr.sym
            self.dirs = expr.dirs
            return

        aux = func.Function(expr)

        if aux.sym == 0:
            if dirs is None:
                raise ValueError("Invalid argument!")
            self.sym, self.dirs = aux.sym, dirs
            return

        functions = list(aux.sym.atoms(sympy.function.AppliedUndef))
        if len(functions) is not 1:
            raise TypeError("There should be exactly 1 functional argument!")

        self.sym = aux.sym.subs(functions[0].func, self.f)
        variables = list(self.sym.atoms(sympy.function.AppliedUndef))[0].args
        self.dirs = [func.Function.x_sub.index(v) for v in variables]
        assert len(self.dirs) is len(functions[0].args)

    def __eq__(self, other):
        return self.dirs == other.dirs and \
            self.sym == other.sym

    def __repr__(self):
        variables = [func.Function.x_sub[d] for d in self.dirs]
        return str(self.f(*variables)) + " --> " + str(self.sym)

    def __mul__(self, other):
        if not isinstance(other, func.Function):
            other = func.Function(other, dirs=self.dirs)
        if not self.dirs == other.dirs:
            raise ValueError("Invalid argument!")
        return Operator(self.sym*other.sym, dirs=self.dirs)

    def __add__(self, other):
        if not isinstance(other, Operator):
            other = Operator(other, dirs=self.dirs)
        if not self.dirs == other.dirs:
            raise ValueError("Invalid argument!")
        return Operator(self.sym + other.sym, dirs=self.dirs)

    def __sub__(self, other):
        if not isinstance(other, Operator):
            other = Operator(other, dirs=self.dirs)
        if not self.dirs == other.dirs:
            raise ValueError("Invalid argument!")
        return Operator(self.sym - other.sym, dirs=self.dirs)

    def __neg__(self):
        return Operator(- self.sym, dirs=self.dirs)

    __str__ = __repr__
    __radd__ = __add__
    __rmul__ = __mul__

    def __call__(self, arg):

        if not isinstance(arg, (func.Function, Operator)):
            try:
                arg = Operator(arg, dirs=self.dirs)
            except TypeError:
                arg = func.Function(arg, dirs=self.dirs)
        variables = [func.Function.x_sub[d] for d in self.dirs]
        sym = self.sym.subs(self.f(*variables), arg.sym).doit()
        return Operator(sym, dirs=self.dirs) if isinstance(arg, Operator) \
            else func.Function(sym, dirs=self.dirs)

    def as_xyz(self):
        sub, xyz = func.Function.x_sub, func.Function.xyz
        return self.sym.subs(((sub[i], xyz[i]) for i in range(len(sub))))

    def map(self, factor):
        if not isinstance(factor, func.Function):
            factor = func.Function(factor, dirs=self.dirs)
        if not factor.dirs == self.dirs:
            raise ValueError("Directions differ")
        variables = [func.Function.x_sub[d] for d in self.dirs]
        unknown = self.f(*variables)
        sym = self.sym.subs(unknown, (unknown*factor).sym)
        sym = (sym.doit()/factor.sym).expand()
        return Operator(sym, dirs=self.dirs)

    def split(self):
        MAX_ORDER = 5
        variables = [func.Function.x_sub[d] for d in self.dirs]
        unknown, dim = self.f(*variables), len(variables)
        result, rem = {}, self.sym.expand()
        mult = list(m for m in itertools.product(range(MAX_ORDER + 1),
                    repeat=dim) if sum(m) <= MAX_ORDER)
        for m in mult:
            if rem == 0:
                break
            test, der = 1, unknown
            for i, v in zip(m, variables):
                test *= v**i/math.factorial(i)
                der = sympy.diff(der, v, i)
            remargs = rem.args if isinstance(rem, sympy.add.Add) else [rem]
            term, rem = 0, 0
            for arg in remargs:  # Convoluted to avoid rounding errors
                termarg = arg.subs(unknown, test).doit()
                if termarg == 0:
                    rem += arg
                else:
                    term += termarg
            if isinstance(term, tuple(sympy.core.all_classes)):
                term = sympy.simplify(term)
            if term != 0:
                result[tuple(m)] = func.Function(term, dirs=self.dirs)
        if rem is not 0:
            raise ValueError("Nonzero remainder")
        return result
