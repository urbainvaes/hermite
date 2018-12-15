import numpy as np
import scipy.sparse as ss
import numpy.linalg as la
import numpy.random as rand
import unittest
import time

import hermite_cpp as cpp
import hermipy.core as core


def timeit(function):
    def wrapper(*args, **kwargs):
        tb = time.time()
        result = function(*args, **kwargs)
        te = time.time()
        return result, te - tb
    return wrapper


class TestConverters(unittest.TestCase):

    def test_conversions(self):
        n = 5000
        a = rand.rand(n, n)

        
        b1, t_to_cpp_1 = timeit(cpp.to_mat)(a)
        c1, t_to_pyt_1 = timeit(cpp.to_numpy)(b1)
        self.assertAlmostEqual(la.norm(a - c1), 0)

        b2, t_to_cpp_2 = timeit(cpp.to_bmat)(a)
        c2, t_to_pyt_2 = timeit(cpp.to_numpy)(b2)
        self.assertAlmostEqual(la.norm(a - c2), 0)

        b3, t_to_cpp_3 = timeit(cpp.to_boost_mat)(a)
        c3, t_to_pyt_3 = timeit(cpp.to_numpy)(b3)
        self.assertAlmostEqual(la.norm(a - c3), 0)

        print_times = False
        if print_times:
            print("\nTimes of conversions to cpp")
            print(t_to_cpp_1, t_to_cpp_2, t_to_cpp_3)

            print("Times of conversions to python")
            print(t_to_pyt_1, t_to_pyt_2, t_to_pyt_3)

    def test_simple_sparse(self):
        matrix = [[0, 1, 0, 0, 0],
                  [0, 2, 0, 3, 0],
                  [0, 0, 4, 0, 0],
                  [5, 6, 0, 0, 0],
                  [7, 0, 0, 0, 8]]
        matrix = np.array(matrix, dtype=float)
        csr_mat = ss.csr_matrix(matrix)
        spmat = core.convert_to_cpp_sparse(csr_mat)
        cpp_mat = cpp.full(spmat)
        self.assertTrue(la.norm(cpp.to_numpy(cpp_mat) - matrix) < 1e-10)


if __name__ == '__main__':
    unittest.main()
