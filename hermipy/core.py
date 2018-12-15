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

import scipy.sparse as ss
import numpy as np
import hermite_cpp as hm

from hermipy.cache import cache
from hermipy.stats import log_stats, debug

# Conversion functions {{{


def convert_to_cpp_vec(vec):
    cpp_vec = hm.double_vec()
    cpp_vec.extend(vec)
    return cpp_vec


def convert_to_cpp_mat(mat):
    cpp_mat = hm.double_mat()
    for vec in mat:
        cpp_mat.append(convert_to_cpp_vec(vec))
    return cpp_mat


def convert_to_cpp_cube(cube):
    cpp_cube = hm.double_cube()
    for mat in cube:
        cpp_cube.append(convert_to_cpp_mat(mat))
    return cpp_cube


@log_stats()
def to_cpp_array(*args):
    if len(args) > 1:
        return (to_cpp_array(arg) for arg in args)
    array, dim = args[0], 0
    if isinstance(array, (list, np.ndarray)):
        dim = 1
        if isinstance(array[0], (list, np.ndarray)):
            dim = 2
            if isinstance(array[0][0], (list, np.ndarray)):
                dim = 3
    if dim is 1:
        array = convert_to_cpp_vec(array)
    elif dim is 2:
        array = convert_to_cpp_mat(array)
    elif dim is 3:
        array = convert_to_cpp_cube(array)
    return array


@log_stats()
def convert_to_cpp_sparse(mat):
    if not isinstance(mat, ss.csr_matrix):
        raise ValueError("Invalid argument!")
    data = hm.double_vec()
    indices = hm.int_vec()
    indptr = hm.int_vec()
    data.extend(mat.data)
    indices.extend(mat.indices.astype('uint32'))
    indptr.extend(mat.indptr.astype('uint32'))
    size1, size2 = mat.shape[0], mat.shape[1]
    return hm.to_spmat(data, indices, indptr, size1, size2)


@log_stats()
def to_csr(sp_matrix):
    if not isinstance(sp_matrix, hm.sparse_matrix):
        raise ValueError("Invalid argument!")
    rcv = np.array(hm.row_col_val(sp_matrix))
    shape = (sp_matrix.size1(), sp_matrix.size2())
    return ss.csr_matrix((rcv[2], (rcv[0], rcv[1])), shape=shape)


@log_stats()
def to_cpp(*args):
    if len(args) > 1:
        return (to_cpp(arg) for arg in args)
    arg = args[0]
    if isinstance(arg, list):
        return to_cpp(np.asarray(arg, dtype=float))
    if isinstance(arg, np.ndarray):
        return to_cpp_array(arg)
    elif isinstance(arg, ss.csr_matrix):
        return convert_to_cpp_sparse(arg)
    else:
        return arg


@log_stats()
def to_numpy(arg):
    if isinstance(arg, hm.double_vec):
        return np.array(arg)
    if isinstance(arg, hm.double_mat):
        return hm.to_numpy(arg)
    elif isinstance(arg, hm.boost_mat):
        return hm.to_numpy(arg)
    elif isinstance(arg, hm.double_cube):
        return np.array(arg)
    elif isinstance(arg, hm.sparse_matrix):
        return to_csr(arg)


# }}}
# Wrapper functions {{{


@cache()
@debug()
@log_stats()
def discretize(function, nodes, translation, dilation):
    nodes, translation, dilation = to_cpp_array(nodes, translation, dilation)
    return np.array(hm.discretize(function, nodes, translation, dilation))


@cache()
@debug()
@log_stats()
def integrate(fgrid, nodes, weights, do_fourier=None):
    if do_fourier is None:
        do_fourier = [0]*len(nodes)
    do_fourier_cpp = hm.int_vec()
    do_fourier_cpp.extend(do_fourier)
    fgrid, nodes, weights = to_cpp_array(fgrid, nodes, weights)
    result = hm.integrate(fgrid, nodes, weights, do_fourier_cpp)
    return result


@cache()
@debug()
@log_stats()
def transform(degree, fgrid, nodes, weights, forward,
              do_fourier=None, index_set="triangle"):
    if do_fourier is None:
        do_fourier = [0]*len(nodes)
    do_fourier_cpp = hm.int_vec()
    do_fourier_cpp.extend(do_fourier)
    degree = int(degree)
    fgrid, nodes, weights = to_cpp_array(fgrid, nodes, weights)
    return np.array(hm.transform(degree, fgrid, nodes, weights,
                                 do_fourier_cpp, forward, index_set))


def triple_products(degree):
    return np.array(hm.triple_products(degree))


def triple_products_fourier(degree):
    return np.array(hm.triple_products_fourier(degree))


@cache()
@debug()
@log_stats()
def varf(degree, fgrid, nodes, weights, sparse=False,
         do_fourier=None, index_set="triangle"):
    if do_fourier is None:
        do_fourier = [0]*len(nodes)
    do_fourier_cpp = hm.int_vec()
    do_fourier_cpp.extend(do_fourier)
    fgrid, nodes, weights = to_cpp_array(fgrid, nodes, weights)
    args = [degree, fgrid, nodes, weights, do_fourier_cpp, index_set]
    function = hm.varf_sp if sparse else hm.varf
    return to_numpy(log_stats()(function)(*args))


@cache()
@debug()
@log_stats()
def varfd(dim, degree, direction, var,
          do_fourier=0, index_set="triangle"):
    if isinstance(var, np.ndarray):
        var = hm.to_boost_mat(var)
    elif isinstance(var, ss.csr_matrix):
        var = convert_to_cpp_sparse(var)
    result = log_stats()(hm.varfd)(dim, degree, direction, var,
                                   do_fourier, index_set)
    return to_numpy(result)


@cache()
@debug()
@log_stats()
def tensorize(inp, dim=None, direction=None,
              sparse=False, index_set="triangle"):

    if isinstance(inp, dict):

        any_elem = list(inp.values())[0]

        if isinstance(any_elem, (float, int)):
            return np.prod([inp[k] for k in inp])

        assert isinstance(any_elem, (np.ndarray, ss.csr_matrix))
        dim = len(any_elem.shape)
        assert dim is 1 or dim is 2

        cpp_arrays = hm.double_mat() if dim is 1 else hm.double_cube()
        cpp_dirs_mat = hm.int_mat()
        for dirs, arg in inp.items():
            cpp_dirs = hm.int_vec()
            cpp_dirs.extend(list(dirs))
            cpp_dirs_mat.append(cpp_dirs)

            # Works only with dense arguments at the moment
            if isinstance(arg, ss.csr_matrix):
                arg = np.array(arg.todense())
            arg = to_cpp_array(arg)
            cpp_arrays.append(arg)
        args = [cpp_arrays, cpp_dirs_mat, index_set]

    elif isinstance(inp, list):
        assert dim is None and direction is None
        is_scalar = isinstance(inp[0], (float, int))
        if is_scalar and dim is None and direction is None:
            return np.prod(inp)
        if isinstance(inp[0], ss.csr_matrix):
            inp = [np.array(m.todense()) for m in inp]
        inp = to_cpp_array(inp)
        args = [inp, index_set]

    else:
        assert dim is not None and direction is not None
        if isinstance(inp, ss.csr_matrix):
            inp = np.array(inp.todense())
        inp = to_cpp_array(inp)
        args = [inp, dim, direction, index_set]

    if sparse:
        tensorize_fun = hm.tensorize_sp
        convert_fun = to_csr
    else:
        tensorize_fun = hm.tensorize
        convert_fun = np.array

    return convert_fun(tensorize_fun(*args))


@cache()
@debug()
@log_stats()
def project(inp, dim, directions, index_set="triangle"):
    if isinstance(directions, int):
        directions = [directions]
    inp = to_cpp(inp)
    directions_cpp = hm.int_vec()
    directions_cpp.extend(directions)
    return to_numpy(hm.project(inp, dim, directions_cpp, index_set))


@cache()
@debug()
@log_stats()
def inner(s1, s2, d1, d2, index_set="triangle"):
    s1, s2 = to_cpp_array(s1, s2)
    d1_cpp, d2_cpp = hm.int_vec(), hm.int_vec()
    d1_cpp.extend(d1)
    d2_cpp.extend(d2)
    return np.array(hm.inner(s1, s2, d1_cpp, d2_cpp, index_set))


# Iterator functions {{{
@cache()
@debug()
@log_stats()
def iterator_get_degree(dim, n_polys, index_set="triangle"):
    cpp_func = {
            "triangle": hm.triangle_get_degree,
            "cross": hm.cross_get_degree,
            "cross_nc": hm.cross_nc_get_degree,
            "cube": hm.cube_get_degree,
            "rectangle": hm.rectangle_get_degree,
            }
    if index_set not in cpp_func:
        raise ValueError("Unknown index set")
    return cpp_func[index_set](dim, n_polys)


@cache()
@debug()
@log_stats()
def iterator_index(mult_ind, index_set="triangle"):

    if isinstance(mult_ind, np.ndarray):
        mult_ind = list(mult_ind.astype('uint32'))
    ind_vec = hm.int_vec()
    ind_vec.extend(mult_ind)

    cpp_func = {
            "triangle": hm.triangle_index,
            "cube": hm.cube_index
            }

    if index_set not in cpp_func:
        raise ValueError("Unknown index set")
    return int(cpp_func[index_set](ind_vec))


@cache()
@debug()
@log_stats()
def iterator_size(dim, degree, index_set="triangle"):
    cpp_func = {
            "triangle": hm.triangle_size,
            "cross": hm.cross_size,
            "cross_nc": hm.cross_nc_size,
            "cube": hm.cube_size,
            "rectangle": hm.rectangle_size,
            }
    if index_set not in cpp_func:
        raise ValueError("Unknown index set")
    return int(cpp_func[index_set](dim, degree))


@cache()
@debug()
@log_stats()
def iterator_list_indices(dim, degree, index_set="triangle"):
    cpp_func = {
            "triangle": hm.triangle_list_indices,
            "cross": hm.cross_list_indices,
            "cross_nc": hm.cross_nc_list_indices,
            "cube": hm.cube_list_indices,
            "rectangle": hm.rectangle_list_indices,
            }
    if index_set not in cpp_func:
        raise ValueError("Unknown index set")
    result = cpp_func[index_set](dim, degree)
    return np.asarray(result, dtype=int)

# }}}
# }}}
