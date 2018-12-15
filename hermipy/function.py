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

import sympy
import re


class Function():

    # xyz notation
    xyz = list(sympy.symbols('x y z w v', real=True))

    # subscript notation
    x_sub = [sympy.symbols('x{}'.format(i), real=True)
             for i in range(len(xyz))]

    conv = {}
    for i, var_xyz in enumerate(xyz):
        conv[str(var_xyz)] = x_sub[i]
        conv[str(x_sub[i])] = x_sub[i]

    @staticmethod
    def tensorize(args):
        dirs, sym = set(), sympy.Integer(1)
        for a in args:
            if not isinstance(a, Function) or \
               not dirs.intersection(a.dirs) == set():
                raise ValueError("Invalid arguments")
            dirs, sym = dirs.union(a.dirs), sym*a.sym
        return Function(sym, dirs=sorted(dirs))

    def __init__(self, expr, dirs=None, dim=None):

        if isinstance(expr, Function):
            self.sym = expr.sym
            self.dirs = expr.dirs
            return

        if not isinstance(expr, tuple(sympy.core.all_classes)):
            expr = sympy.sympify(expr, locals=self.conv)

        for s in expr.free_symbols:
            if str(s) in self.conv:
                expr = expr.subs(s, self.conv[str(s)])

        self.sym = expr.expand()
        variables = expr.free_symbols.intersection(self.x_sub)
        if dirs is not None:
            self.dirs = dirs
            if dim is not None or not self.dirs == sorted(self.dirs):
                raise ValueError("Invalid arguments")
        elif dim is not None:
            self.dirs = list(range(dim))
        else:
            # We do not minimize the number of dimensions
            dim = 0 if variables == set() else \
                  max(self.x_sub.index(s) for s in variables) + 1
            self.dirs = list(range(dim))
        assert variables.issubset({self.x_sub[d] for d in self.dirs})

    def __eq__(self, other):
        return self.dirs == other.dirs and \
            self.sym == other.sym

    def __repr__(self):
        variables = [self.x_sub[d] for d in self.dirs]
        return str(variables) + " --> " + str(self.sym)

    def __mul__(self, other):
        if not isinstance(other, Function):
            other = Function(other, dirs=self.dirs)
        if not self.dirs == other.dirs:
            raise ValueError("Invalid argument")
        sym = self.sym * other.sym
        return Function(sym, dirs=self.dirs)

    def __truediv__(self, other):
        if not isinstance(other, Function):
            other = Function(other, dirs=self.dirs)
        if not self.dirs == other.dirs:
            raise ValueError("Invalid argument")
        sym = self.sym / other.sym
        return Function(sym, dirs=self.dirs)

    def __add__(self, other):
        if not isinstance(other, Function):
            other = Function(other, dirs=self.dirs)
        if not self.dirs == other.dirs:
            raise ValueError("Invalid argument")
        sym = self.sym + other.sym
        return Function(sym, dirs=self.dirs)

    __str__ = __repr__
    __rmul__ = __mul__
    __radd__ = __add__

    def as_xyz(self):
        result = self.sym
        for i, sub_i in enumerate(self.x_sub):
            result = result.subs(sub_i, self.xyz[i])
        return result

    def ccode(self):
        function = sympy.ccode(self.sym)
        for i, direction in enumerate(self.dirs):
            function = re.sub(r'\bx{}'.format(direction),
                              'v[{}]'.format(i), function)
        return function

    def project(self, dirs):
        dirs = dirs if isinstance(dirs, list) else [dirs]
        split_fun = self.split(legacy=False)
        assert len(split_fun) == 1

        # Absorb constant in projection on lowest index
        if self.dirs[0] in dirs:
            result = split_fun[0][frozenset()].sym
        else:
            result = sympy.Integer(1)

        for v, term in split_fun[0].items():
            if v.issubset(dirs) and v != frozenset():
                result *= term.sym
        return Function(result, dirs=dirs)

    def split(self, legacy=True):
        dim = len(self.dirs)
        is_add = isinstance(self.sym, sympy.add.Add)
        add_terms = self.sym.args if is_add else [self.sym]
        to_return = []

        for term in add_terms:
            is_mul = isinstance(term, sympy.mul.Mul)
            mul_terms = term.args if is_mul else [term]
            result = {frozenset({d}): 1 for d in self.dirs}

            # For constant multiplier
            result[frozenset()] = 1

            for arg in mul_terms:
                vars_arg = arg.free_symbols.intersection(self.x_sub)
                key = frozenset(self.x_sub.index(v) for v in vars_arg)
                keys_intersect, value = [], arg

                if key in result:
                    result[key] *= value
                    continue

                for existing_key in result:
                    if existing_key.intersection(key) != frozenset():
                        key = existing_key.union(key)
                        value *= result[existing_key]
                        keys_intersect.append(existing_key)

                for k in keys_intersect:
                    del result[k]

                result[key] = value

            if legacy:
                tmp, multiplicator = [], result[frozenset()]
                if len(result) == dim + 1:
                    for d in self.dirs:
                        tmp.append(Function(result[frozenset({d})], dirs=[d]))
                else:
                    tmp.append(Function(term/multiplicator, dirs=self.dirs))
                tmp.append(multiplicator)
                result = tmp
            else:
                for dirs, arg in result.items():
                    result[dirs] = Function(arg, dirs=list(dirs))

            to_return.append(result)
        return to_return

    @staticmethod
    def sanitize(expr, max_denom=1e8):
        result, expr = sympy.Integer(0), (sympy.Integer(0) + expr).expand()
        is_add = isinstance(expr, sympy.add.Add)
        add_terms = expr.args if is_add else [expr]
        for aterm in add_terms:
            result_tmp = sympy.Integer(1)
            is_mul = isinstance(aterm, sympy.mul.Mul)
            mul_terms = aterm.args if is_mul else [aterm]
            for mterm in mul_terms:
                if isinstance(mterm, sympy.numbers.Float):
                    if abs(mterm) < 1e-14:
                        mterm = 0
                    mterm = sympy.Rational(mterm).limit_denominator(max_denom)
                result_tmp *= mterm
            result += result_tmp
        return result
