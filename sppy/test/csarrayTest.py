import logging
import sys
import unittest
import numpy
import numpy.testing as nptst
import math
import pickle
import sppy
from sppy import csarray


class csarrayTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        self.A = csarray((5, 5))

        nrow = 5
        ncol = 7
        self.B = csarray((nrow, ncol))
        self.B[0, 1] = 1
        self.B[1, 3] = 5.2
        self.B[3, 3] = -0.2
        self.B[0, 6] = -1.23
        self.B[4, 4] = 12.2

        nrow = 100
        ncol = 100
        self.C = csarray((nrow, ncol))
        self.C[0, 1] = 1
        self.C[10, 3] = 5.2
        self.C[30, 34] = -0.2
        self.C[0, 62] = -1.23
        self.C[4, 41] = 12.2

        self.D = csarray((5, 5))
        self.D[0, 0] = 23.1
        self.D[2, 0] = -3.1
        self.D[3, 0] = -10.0
        self.D[2, 1] = -5
        self.D[3, 1] = 5

        self.E = csarray((0, 0))

        self.F = csarray((6, 6), dtype=numpy.int64)
        self.F[0, 0] = 23
        self.F[2, 0] = -3
        self.F[3, 0] = -10
        self.F[2, 1] = -5
        self.F[3, 1] = 5

        self.G = csarray((6, 6), storagetype="row")
        self.G[0, 0] = 23
        self.G[2, 0] = -3
        self.G[3, 0] = -10
        self.G[2, 1] = -5
        self.G[3, 1] = 5

        self.H = csarray((5, 7), storagetype="row")
        self.H[0, 1] = 1
        self.H[1, 3] = 5.2
        self.H[3, 3] = -0.2
        self.H[0, 6] = -1.23
        self.H[4, 4] = 12.2

        self.I = csarray((5, 5), storagetype="row")

        self.a = csarray(10, dtype=numpy.float64)
        self.a[0] = 23
        self.a[3] = 1.2
        self.a[4] = -8

        self.b = csarray(10, dtype=numpy.int64)
        self.b[0] = 23
        self.b[5] = 1
        self.b[8] = -8

        self.c = csarray((3,), dtype=numpy.float64)

        self.d = csarray((0,), dtype=numpy.float64)

        self.storagetypes = ["row", "col"]

    def testInit(self):
        A = csarray((5, 7))
        self.assertEqual(A.shape, (5, 7))

        A = csarray((1, 1))
        self.assertEqual(A.shape, (1, 1))

        A = csarray((1, 0))
        self.assertEqual(A.shape, (1, 0))

        A = csarray((0, 0))
        self.assertEqual(A.shape, (0, 0))

        a = csarray((5))
        self.assertEqual(a.shape, (5,))

        a = csarray(0)
        self.assertEqual(a.shape, (0,))

        # Test bad input params
        self.assertRaises(ValueError, csarray, (0, 1, 2))
        self.assertRaises(ValueError, csarray, "a")
        self.assertRaises(ValueError, csarray, (5, 5), numpy.float64, "abc")

        # TODO: Test other dtypes
        A = csarray((5, 5))
        self.assertEqual(A.dtype, numpy.float64)

        self.assertEqual(self.F[0, 0], 23)
        self.F[1, 1] = 51.2
        self.assertEqual(self.F[1, 1], 51)

        # Test assignment with a numpy array
        A = numpy.array([[3.1, 0, 100], [1.11, 0, 4], [0, 0, 5.2]])
        B = csarray(A)

        nptst.assert_array_equal(B.toarray(), A)

        A = numpy.array([3.1, 0, 100])
        B = csarray(A)

        nptst.assert_array_equal(B.toarray(), A)

        B = csarray(A, dtype=numpy.int8)
        nptst.assert_array_equal(B.toarray(), numpy.array(A, numpy.int8))

        B = csarray(A, dtype=numpy.int8, storagetype="row")
        nptst.assert_array_equal(B.toarray(), numpy.array(A, numpy.int8))

        # Assignment to other csarray
        B = csarray(self.B, numpy.int64)

        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                self.assertEqual(B[i, j], int(self.B[i, j]))

        B = csarray(self.B, numpy.int64, storagetype="row")

        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                self.assertEqual(B[i, j], int(self.B[i, j]))

        F = csarray(self.F, numpy.float64)
        F[0, 0] += 0.1

        self.assertEqual(F[0, 0], 23.1)

        # This doesn't work as we can't instantiate using an array
        # b = csarray(self.b, numpy.int64)

        # for i in range(b.shape[0]):
        #    self.assertEqual(b[i], int(self.b[i]))

        # Test creating a matrix using a scipy.sparse
        try:
            import scipy.sparse

            A = scipy.sparse.csr_matrix((10, 5))
            A[0, 1] = 3
            A[5, 4] = -0.1

            B = csarray(A)

            self.assertEqual(A.shape, B.shape)
            self.assertEqual(A.nnz, B.nnz)
            self.assertEqual(A[0, 1], B[0, 1])
            self.assertEqual(A[5, 4], B[5, 4])

        except ImportError:
            pass

    def testNDim(self):
        A = csarray((5, 7))
        self.assertEqual(A.ndim, 2)

        A = csarray((5, 7), storagetype="row")
        self.assertEqual(A.ndim, 2)

        A = csarray((0, 0))
        self.assertEqual(A.ndim, 2)

        self.assertEqual(self.a.ndim, 1)
        self.assertEqual(self.b.ndim, 1)

    def testSize(self):
        self.assertEqual(self.A.size, 25)
        self.assertEqual(self.B.size, 35)
        self.assertEqual(self.C.size, 10000)
        self.assertEqual(self.F.size, 36)
        self.assertEqual(self.G.size, 36)

        self.assertEqual(self.a.size, 10)
        self.assertEqual(self.b.size, 10)
        self.assertEqual(self.c.size, 3)

    def testGetnnz(self):
        A = csarray((5, 7))
        self.assertEqual(A.getnnz(), 0)
        A[0, 0] = 1.0

        self.assertEqual(A.getnnz(), 1)

        A[2, 1] = 1.0
        self.assertEqual(A.getnnz(), 2)

        A[2, 5] = 1.0
        A[3, 5] = 1.0
        self.assertEqual(A.getnnz(), 4)

        # If we insert a zero it is not registered as zero
        A[4, 4] = 0.0
        self.assertEqual(A.getnnz(), 4)

        # But erasing an item keeps it (can call prune)
        A[3, 5] = 0.0
        self.assertEqual(A.getnnz(), 4)

        B = csarray((5, 7))
        B[(numpy.array([1, 2, 3]), numpy.array([4, 5, 6]))] = 1
        self.assertEqual(B.getnnz(), 3)

        for i in range(5):
            for j in range(7):
                B[i, j] = 1

        self.assertEqual(B.getnnz(), 35)

        self.assertEqual(self.A.getnnz(), 0)
        self.assertEqual(self.B.getnnz(), 5)
        self.assertEqual(self.C.getnnz(), 5)
        self.assertEqual(self.F.getnnz(), 5)
        self.assertEqual(self.G.getnnz(), 5)

        self.assertEqual(self.a.getnnz(), 3)
        self.assertEqual(self.b.getnnz(), 3)
        self.assertEqual(self.c.getnnz(), 0)

    def testSetItem(self):
        nrow = 5
        ncol = 7

        storagetypes = ["col", "row"]

        for storagetype in storagetypes:
            A = csarray((nrow, ncol), storagetype=storagetype)
            A[0, 1] = 1
            A[1, 3] = 5.2
            A[3, 3] = -0.2

            self.assertEqual(A[0, 1], 1)
            self.assertAlmostEqual(A[1, 3], 5.2)
            self.assertAlmostEqual(A[3, 3], -0.2)

            a = csarray(nrow)
            a[0] = 1
            a[1] = 5.2
            a[3] = -0.2

            self.assertEqual(a[0], 1)
            self.assertAlmostEqual(a[1], 5.2)
            self.assertAlmostEqual(a[3], -0.2)

            for i in range(nrow):
                for j in range(ncol):
                    if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3):
                        self.assertEqual(A[i, j], 0)

            self.assertRaises(ValueError, A.__setitem__, (20, 1), 1)
            self.assertRaises(TypeError, A.__setitem__, (1, 1), "a")
            self.assertRaises(ValueError, A.__setitem__, (1, 100), 1)
            A[-1, 1] = 1
            self.assertEqual(A[-1, 1], 1)
            A[0, -1] = 1
            self.assertEqual(A[0, -1], 1)
            A[-1, 1] = 0
            A[0, -1] = 0
            self.assertRaises(ValueError, a.__setitem__, (0, 0), 1)

            result = A[(numpy.array([0, 1, 3]), numpy.array([1, 3, 3]))]
            self.assertEqual(result[0], 1)
            self.assertEqual(result[1], 5.2)
            self.assertEqual(result[2], -0.2)

            # Replace value of A
            A[0, 1] = 2
            self.assertEqual(A[0, 1], 2)
            self.assertAlmostEqual(A[1, 3], 5.2)
            self.assertAlmostEqual(A[3, 3], -0.2)

            for i in range(nrow):
                for j in range(ncol):
                    if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3):
                        self.assertEqual(A[i, j], 0)

            a[0] = 10
            self.assertEqual(a[0], 10)

            # Try setting items with arrays
            A = csarray((nrow, ncol), storagetype=storagetype)
            A[numpy.array([0, 1]), numpy.array([2, 3])] = numpy.array([1.2, 2.4])

            self.assertEqual(A.getnnz(), 2)
            self.assertEqual(A[0, 2], 1.2)
            self.assertEqual(A[1, 3], 2.4)

            A[numpy.array([2, 4]), numpy.array([2, 3])] = 5

            self.assertEqual(A[2, 2], 5)
            self.assertEqual(A[4, 3], 5)

            a = csarray(nrow, storagetype=storagetype)
            a[numpy.array([0, 2])] = numpy.array([1.2, 2.4])
            self.assertEqual(a.getnnz(), 2)
            self.assertEqual(a[0], 1.2)
            self.assertEqual(a[2], 2.4)

    def testStr(self):
        nrow = 5
        ncol = 7

        storagetypes = ["col", "row"]

        for storagetype in storagetypes:
            A = csarray((nrow, ncol), storagetype=storagetype)
            A[0, 1] = 1
            A[1, 3] = 5.2
            A[3, 3] = -0.2

            outputStr = (
                "csarray dtype:float64 shape:(5, 7) non-zeros:3 storage:"
                + A.storagetype
                + "\n"
            )
            outputStr += "(0, 1) 1.0\n"
            outputStr += "(1, 3) 5.2\n"
            outputStr += "(3, 3) -0.2"
            self.assertEqual(str(A), outputStr)

            B = csarray((5, 5), storagetype=storagetype)
            outputStr = (
                "csarray dtype:float64 shape:(5, 5) non-zeros:0 storage:"
                + B.storagetype
                + "\n"
            )
            self.assertEqual(str(B), outputStr)

            outputStr = "csarray dtype:float64 shape:(10,) non-zeros:3\n"
            outputStr += "(0) 23.0\n"
            outputStr += "(3) 1.2\n"
            outputStr += "(4) -8.0"
            self.assertEqual(str(self.a), outputStr)

            outputStr = "csarray dtype:float64 shape:(3,) non-zeros:0\n"
            self.assertEqual(str(self.c), outputStr)

    def testSum(self):
        nrow = 5
        ncol = 7

        storagetypes = ["col", "row"]

        for storagetype in storagetypes:
            A = csarray((nrow, ncol), storagetype=storagetype)
            A[0, 1] = 1
            A[1, 3] = 5.2
            A[3, 3] = -0.2

            self.assertEqual(A.sum(), 6.0)

            A[3, 4] = -1.2
            self.assertEqual(A.sum(), 4.8)

            A[0, 0] = 1.34
            self.assertEqual(A.sum(), 6.14)

            A[0, 0] = 0
            self.assertEqual(A.sum(), 4.8)

        self.assertEqual(self.A.sum(), 0.0)
        self.assertEqual(self.B.sum(), 16.97)
        self.assertEqual(self.C.sum(), 16.97)
        self.assertAlmostEqual(self.D.sum(), 10)

        self.assertEqual(self.F.sum(), 10)
        self.assertEqual(self.G.sum(), 10)

        self.assertEqual(self.a.sum(), 16.2)
        self.assertEqual(self.b.sum(), 16)
        self.assertEqual(self.c.sum(), 0)

        # Test sum along axes
        nptst.assert_array_equal(self.A.sum(0), numpy.zeros(5))
        nptst.assert_array_equal(
            self.B.sum(0), numpy.array([0, 1, 0, 5, 12.2, 0, -1.23])
        )
        nptst.assert_array_equal(self.D.sum(0), numpy.array([10, 0, 0, 0, 0]))
        nptst.assert_array_equal(self.F.sum(0), self.G.sum(0))

        nptst.assert_array_equal(self.A.sum(1), numpy.zeros(5))
        nptst.assert_array_almost_equal(
            self.B.sum(1), numpy.array([-0.23, 5.2, 0, -0.2, 12.2])
        )
        nptst.assert_array_equal(self.D.sum(1), numpy.array([23.1, 0, -8.1, -5, 0]))
        nptst.assert_array_equal(self.F.sum(1), self.G.sum(1))

    def testGet(self):
        self.assertEqual(self.B[0, 1], 1)
        self.assertEqual(self.B[1, 3], 5.2)
        self.assertEqual(self.B[3, 3], -0.2)
        self.assertEqual(self.B.getnnz(), 5)

        self.assertEqual(self.G[0, 0], 23)
        self.assertEqual(self.G[2, 0], -3)

        self.assertEqual(self.a[0], 23)
        self.assertEqual(self.a[3], 1.2)
        self.assertEqual(self.a[4], -8)
        self.assertEqual(self.a.getnnz(), 3)

        # Test negative indices
        self.assertEqual(self.B[-5, -6], 1)
        self.assertEqual(self.B[-1, -3], 12.2)
        self.assertEqual(self.G[-3, -5], 5)

        self.assertEqual(self.b[-2], -8)

        self.assertRaises(ValueError, self.B.__getitem__, (20, 1))
        self.assertRaises(ValueError, self.B.__getitem__, (1, 20))
        self.assertRaises(ValueError, self.B.__getitem__, (-6, 1))
        self.assertRaises(ValueError, self.B.__getitem__, (1, -8))
        nptst.assert_array_equal(self.B[1].toarray(), self.B.toarray()[1])
        self.assertRaises(ValueError, self.B.__getitem__, "a")
        self.assertRaises(Exception, self.B.__getitem__, ("a", "c"))

        # Test array indexing using arrays
        C = self.B[numpy.array([0, 1, 3]), numpy.array([1, 3, 3])]
        self.assertEqual(C.shape[0], 3)
        self.assertEqual(C[0], 1)
        self.assertEqual(C[1], 5.2)
        self.assertEqual(C[2], -0.2)

        C = self.A[numpy.array([0, 1, 3]), numpy.array([1, 3, 3])]
        self.assertEqual(C[0], 0)
        self.assertEqual(C[1], 0)
        self.assertEqual(C[2], 0)

        C = self.G[numpy.array([0, 2, 3]), numpy.array([0, 0, 0])]
        self.assertEqual(C[0], 23)
        self.assertEqual(C[1], -3)
        self.assertEqual(C[2], -10)

        d = self.a[numpy.array([0, 3, 4])]
        self.assertEqual(d[0], 23)
        self.assertEqual(d[1], 1.2)
        self.assertEqual(d[2], -8)

        A = csarray((2, 2))
        self.assertRaises(
            ValueError, A.__getitem__, (numpy.array([0, 1]), numpy.array([1, 3]))
        )

        A = csarray((2, 2))
        self.assertRaises(
            ValueError, A.__getitem__, (numpy.array([0, 2]), numpy.array([1, 1]))
        )

        A = csarray((0, 0))
        self.assertRaises(
            ValueError, A.__getitem__, (numpy.array([0, 1, 3]), numpy.array([1, 3, 3]))
        )

        # Test oversampling
        B2 = self.B[0:1000, :]
        nptst.assert_array_equal(B2.toarray(), self.B.toarray())

        B2 = self.B[:, 0:230]
        nptst.assert_array_equal(B2.toarray(), self.B.toarray())

        # Test submatrix indexing
        C = self.B[:, :]

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i, j] = self.B[i, j]

        C = self.B[0:5, 0:7]

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i, j] = self.B[i, j]

        C = self.G[0:3, 0:4]

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i, j] = self.G[i, j]

        C = self.B[numpy.array([0, 1, 3]), :]
        self.assertEqual(C.shape, (3, 7))
        self.assertEqual(C.getnnz(), 4)
        self.assertEqual(C[0, 1], 1)
        self.assertEqual(C[1, 3], 5.2)
        self.assertEqual(C[2, 3], -0.2)
        self.assertEqual(C[0, 6], -1.23)

        C = self.H[numpy.array([0, 1, 3]), :]
        self.assertEqual(C.shape, (3, 7))
        self.assertEqual(C.getnnz(), 4)
        self.assertEqual(C[0, 1], 1)
        self.assertEqual(C[1, 3], 5.2)
        self.assertEqual(C[2, 3], -0.2)
        self.assertEqual(C[0, 6], -1.23)

        C = self.B[numpy.array([0, 1, 3]), 0:7]
        self.assertEqual(C.shape, (3, 7))
        self.assertEqual(C.getnnz(), 4)
        self.assertEqual(C[0, 1], 1)
        self.assertEqual(C[1, 3], 5.2)
        self.assertEqual(C[2, 3], -0.2)
        self.assertEqual(C[0, 6], -1.23)

        C = self.H[numpy.array([0, 1, 3]), :]
        self.assertEqual(C.shape, (3, 7))
        self.assertEqual(C.getnnz(), 4)
        self.assertEqual(C[0, 1], 1)
        self.assertEqual(C[1, 3], 5.2)
        self.assertEqual(C[2, 3], -0.2)
        self.assertEqual(C[0, 6], -1.23)

        C = self.B[:, numpy.array([3])]
        self.assertEqual(C.shape, (5, 1))
        self.assertEqual(C.getnnz(), 2)
        self.assertEqual(C[1, 0], 5.2)
        self.assertEqual(C[3, 0], -0.2)

        C = self.H[:, numpy.array([3])]
        self.assertEqual(C.shape, (5, 1))
        self.assertEqual(C.getnnz(), 2)
        self.assertEqual(C[1, 0], 5.2)
        self.assertEqual(C[3, 0], -0.2)

        self.assertEqual(self.F[0, 0], 23)

        d = self.a[0:4]
        self.assertEqual(d.shape, (4,))
        self.assertEqual(d[0], 23)
        self.assertEqual(d[3], 1.2)

        # Test for particular bug
        B = sppy.rand((10, 5), 0.5)
        nptst.assert_array_equal(B[:, 1].toarray().ravel(), B.toarray()[:, 1])

    def testSubArray(self):
        rowInds = numpy.array([0, 1], numpy.int64)
        colInds = numpy.array([1, 3, 6], numpy.int64)
        A = self.B.subArray(rowInds, colInds)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(A[i, j], self.B[rowInds[i], colInds[j]])

        A = self.H.subArray(rowInds, colInds)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(A[i, j], self.B[rowInds[i], colInds[j]])

        # Try all rows/cols
        rowInds = numpy.arange(5)
        colInds = numpy.arange(7)

        A = self.B.subArray(rowInds, colInds)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(A[i, j], self.B[rowInds[i], colInds[j]])

        A = self.H.subArray(rowInds, colInds)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(A[i, j], self.B[rowInds[i], colInds[j]])

        # No rows/cols
        rowInds = numpy.array([], numpy.int64)
        colInds = numpy.array([], numpy.int64)

        A = self.B.subArray(rowInds, colInds)
        self.assertEqual(A.shape, (0, 0))

        A = self.A.subArray(rowInds, colInds)
        self.assertEqual(A.shape, (0, 0))

    def testSlice(self):

        numRuns = 100
        dtypes = [numpy.float64, numpy.float32, numpy.float64]
        storagetypes = ["row", "col"]
        size = 5

        for i in range(numRuns):
            for dtype in dtypes:
                for storagetype in storagetypes:
                    m = numpy.random.randint(5, 20)
                    n = numpy.random.randint(5, 20)
                    density = numpy.random.rand()

                    # print(i, m, n, density, dtype, storagetype)
                    A = sppy.rand((m, n), density, dtype=dtype, storagetype=storagetype)
                    D = A.toarray()

                    rowInds = numpy.random.choice(m, size, replace=False)
                    colInds = numpy.random.choice(n, size, replace=False)

                    rowInds = numpy.sort(rowInds)
                    colInds = numpy.sort(colInds)

                    B = A[rowInds, :]
                    nptst.assert_array_equal(B.toarray(), D[rowInds, :])

                    C = A[:, colInds]
                    nptst.assert_array_equal(C.toarray(), D[:, colInds])

    # @unittest.skip("")
    def testNonZeroInds(self):

        (rowInds, colInds) = self.B.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertNotEqual(self.B[rowInds[i], colInds[i]], 0)

        self.assertEqual(self.B.getnnz(), rowInds.shape[0])
        self.assertEqual(self.B.sum(), self.B[rowInds, colInds].sum())

        (rowInds, colInds) = self.C.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertNotEqual(self.C[rowInds[i], colInds[i]], 0)

        self.assertEqual(self.C.getnnz(), rowInds.shape[0])
        self.assertEqual(self.C.sum(), self.C[rowInds, colInds].sum())

        (rowInds, colInds) = self.F.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertNotEqual(self.F[rowInds[i], colInds[i]], 0)

        self.assertEqual(self.F.getnnz(), rowInds.shape[0])
        self.assertEqual(self.F.sum(), self.F[rowInds, colInds].sum())

        (rowInds, colInds) = self.G.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertNotEqual(self.G[rowInds[i], colInds[i]], 0)

        self.assertEqual(self.G.getnnz(), rowInds.shape[0])
        self.assertEqual(self.G.sum(), self.G[rowInds, colInds].sum())

        (inds,) = self.a.nonzero()
        for i in range(inds.shape[0]):
            self.assertNotEqual(self.a[inds[i]], 0)

        # Try an array with no non zeros
        nrow = 5
        ncol = 7
        storagetypes = ["col", "row"]
        for storagetype in storagetypes:
            A = csarray((nrow, ncol), storagetype=storagetype)
            (rowInds, colInds) = A.nonzero()

            self.assertEqual(A.getnnz(), rowInds.shape[0])
            self.assertEqual(rowInds.shape[0], 0)
            self.assertEqual(colInds.shape[0], 0)

        (inds,) = self.c.nonzero()
        self.assertEqual(inds.shape[0], 0)

        # Zero size array
        nrow = 0
        ncol = 0
        A = csarray((nrow, ncol))
        (rowInds, colInds) = A.nonzero()
        self.assertEqual(A.getnnz(), rowInds.shape[0])
        self.assertEqual(rowInds.shape[0], 0)
        self.assertEqual(colInds.shape[0], 0)

        (inds,) = self.d.nonzero()
        self.assertEqual(inds.shape[0], 0)

    def testDiag(self):
        nptst.assert_array_equal(self.A.diag(), numpy.zeros(5))
        nptst.assert_array_equal(self.B.diag(), numpy.array([0, 0, 0, -0.2, 12.2]))
        nptst.assert_array_equal(self.C.diag(), numpy.zeros(100))
        nptst.assert_array_equal(self.H.diag(), numpy.array([0, 0, 0, -0.2, 12.2]))

        D = csarray((3, 3))
        D[0, 0] = -1
        D[1, 1] = 3.2
        D[2, 2] = 34

        nptst.assert_array_equal(D.diag(), numpy.array([-1, 3.2, 34]))

        E = csarray((0, 0))
        nptst.assert_array_equal(E.diag(), numpy.array([]))

        nptst.assert_array_equal(self.F.diag(), numpy.array([23, 0, 0, 0, 0, 0]))

    # @unittest.skip("")
    def testMean(self):
        self.assertEqual(self.A.mean(), 0)

        self.assertAlmostEqual(self.B.mean(), 0.4848571428571428)
        self.assertAlmostEqual(self.C.mean(), 0.001697)
        self.assertAlmostEqual(self.H.mean(), 0.4848571428571428)

        D = csarray((0, 0))
        self.assertTrue(math.isnan(D.mean()))

        self.assertEqual(self.F.mean(), 10 / float(36))

        nptst.assert_array_equal(self.A.mean(0), self.A.sum(0) / self.A.shape[0])
        nptst.assert_array_equal(self.B.mean(0), self.B.sum(0) / self.B.shape[0])
        nptst.assert_array_equal(self.C.mean(0), self.C.sum(0) / self.C.shape[0])
        nptst.assert_array_equal(self.D.mean(0), self.D.sum(0) / self.D.shape[0])
        nptst.assert_array_equal(self.F.mean(0), self.F.sum(0) / float(self.F.shape[0]))
        nptst.assert_array_equal(self.G.mean(0), self.G.sum(0) / self.G.shape[0])
        nptst.assert_array_equal(self.H.mean(0), self.H.sum(0) / self.H.shape[0])

        nptst.assert_array_equal(self.A.mean(1), self.A.sum(1) / self.A.shape[1])
        nptst.assert_array_equal(self.B.mean(1), self.B.sum(1) / self.B.shape[1])
        nptst.assert_array_equal(self.C.mean(1), self.C.sum(1) / self.C.shape[1])
        nptst.assert_array_equal(self.D.mean(1), self.D.sum(1) / self.D.shape[1])
        nptst.assert_array_equal(self.F.mean(1), self.F.sum(1) / float(self.F.shape[1]))
        nptst.assert_array_equal(self.G.mean(1), self.G.sum(1) / self.G.shape[1])
        nptst.assert_array_equal(self.H.mean(1), self.H.sum(1) / self.H.shape[1])

        self.assertEqual(self.a.mean(), 1.6199999999999999)
        self.assertEqual(self.b.mean(), 1.6)
        self.assertEqual(self.c.mean(), 0.0)
        self.assertTrue(math.isnan(self.d.mean()))

    def testCopy(self):
        A = csarray((5, 5))
        A[0, 0] = 1
        A[1, 0] = 2
        A[4, 2] = 3
        self.assertEqual(A[0, 0], 1)
        self.assertEqual(A[1, 0], 2)
        self.assertEqual(A[4, 2], 3)

        B = A.copy()
        A[0, 0] = 2
        A[1, 0] = 3
        A[4, 2] = 4
        A[4, 4] = 5

        self.assertEqual(A[0, 0], 2)
        self.assertEqual(A[1, 0], 3)
        self.assertEqual(A[4, 2], 4)
        self.assertEqual(A[4, 4], 5)
        self.assertEqual(A.getnnz(), 4)

        self.assertEqual(B[0, 0], 1)
        self.assertEqual(B[1, 0], 2)
        self.assertEqual(B[4, 2], 3)
        self.assertEqual(B.getnnz(), 3)

        F = self.F.copy()
        F[0, 0] = -15
        self.assertEqual(F[0, 0], -15)
        self.assertEqual(self.F[0, 0], 23)

        G = self.G.copy()
        G[0, 0] = -15
        self.assertEqual(G[0, 0], -15)
        self.assertEqual(self.G[0, 0], 23)

        # Now try with 1d arrays
        a2 = self.a.copy()
        self.a[0] = 10
        self.a[3] = 1
        self.a[4] = 2

        self.assertEqual(a2[0], 23)
        self.assertEqual(a2[3], 1.2)
        self.assertEqual(a2[4], -8)

    def testMultiply(self):
        val = 2.0
        C = self.B * val

        self.assertEqual(self.B[0, 1], 1)
        self.assertEqual(self.B[1, 3], 5.2)
        self.assertEqual(self.B[3, 3], -0.2)
        self.assertEqual(self.B[0, 6], -1.23)
        self.assertEqual(self.B[4, 4], 12.2)

        self.assertEqual(C[0, 1], self.B[0, 1] * val)
        self.assertEqual(C[1, 3], self.B[1, 3] * val)
        self.assertEqual(C[3, 3], self.B[3, 3] * val)
        self.assertEqual(C[0, 6], self.B[0, 6] * val)
        self.assertEqual(C[4, 4], self.B[4, 4] * val)

        C = self.H * val

        self.assertEqual(self.H[0, 1], 1)
        self.assertEqual(self.H[1, 3], 5.2)
        self.assertEqual(self.H[3, 3], -0.2)
        self.assertEqual(self.H[0, 6], -1.23)
        self.assertEqual(self.H[4, 4], 12.2)

        self.assertEqual(C[0, 1], self.H[0, 1] * val)
        self.assertEqual(C[1, 3], self.H[1, 3] * val)
        self.assertEqual(C[3, 3], self.H[3, 3] * val)
        self.assertEqual(C[0, 6], self.H[0, 6] * val)
        self.assertEqual(C[4, 4], self.H[4, 4] * val)

        G = self.F * val
        self.assertEqual(self.F[0, 0], 23)
        self.assertEqual(G[0, 0], 46)

        # Now with vectors
        a2 = self.a * val
        self.assertEqual(a2[0], self.a[0] * val)
        self.assertEqual(a2[3], self.a[3] * val)
        self.assertEqual(a2[4], self.a[4] * val)

    def testTrace(self):
        self.assertEqual(self.A.trace(), 0)
        self.assertEqual(self.B.trace(), 12)
        self.assertEqual(self.C.trace(), 0)
        self.assertEqual(self.D.trace(), 23.1)
        self.assertEqual(self.F.trace(), 23)
        self.assertEqual(self.G.trace(), 23)
        self.assertEqual(self.H.trace(), 12)

    # @unittest.skip("")
    def testToarray(self):
        A = self.A.toarray()
        self.assertEqual(type(A), numpy.ndarray)
        self.assertEqual(A.shape, self.A.shape)
        self.assertEqual(A.sum(), 0)

        B = self.B.toarray()
        self.assertEqual(type(B), numpy.ndarray)
        self.assertEqual(B.shape, self.B.shape)
        self.assertEqual(B[0, 1], 1)
        self.assertEqual(B[1, 3], 5.2)
        self.assertEqual(B[3, 3], -0.2)
        self.assertEqual(B[0, 6], -1.23)
        self.assertEqual(B[4, 4], 12.2)
        self.assertEqual(B.sum(), self.B.sum())

        D = self.D.toarray()
        self.assertEqual(type(D), numpy.ndarray)
        self.assertEqual(D.shape, self.D.shape)
        self.assertEqual(D[0, 0], 23.1)
        self.assertEqual(D[2, 0], -3.1)
        self.assertEqual(D[3, 0], -10.0)
        self.assertEqual(D[2, 1], -5)
        self.assertEqual(D[3, 1], 5)
        self.assertAlmostEqual(D.sum(), self.D.sum())

        F = self.F.toarray()
        self.assertEqual(type(F), numpy.ndarray)
        self.assertEqual(F.shape, self.F.shape)
        self.assertEqual(F[0, 0], 23)
        self.assertEqual(F[2, 0], -3)
        self.assertEqual(F[3, 0], -10)
        self.assertEqual(F[2, 1], -5)
        self.assertEqual(F[3, 1], 5)
        self.assertAlmostEqual(F.sum(), self.F.sum())

        G = self.G.toarray()
        self.assertEqual(type(G), numpy.ndarray)
        self.assertEqual(G.shape, self.G.shape)
        self.assertEqual(G[0, 0], 23)
        self.assertEqual(G[2, 0], -3)
        self.assertEqual(G[3, 0], -10)
        self.assertEqual(G[2, 1], -5)
        self.assertEqual(G[3, 1], 5)
        self.assertAlmostEqual(G.sum(), self.G.sum())

        # Vectors
        a2 = self.a.toarray()
        self.assertEqual(type(a2), numpy.ndarray)
        self.assertEqual(a2.shape, self.a.shape)
        self.assertEqual(a2[0], 23)
        self.assertEqual(a2[3], 1.2)
        self.assertEqual(a2[4], -8)
        self.assertAlmostEqual(a2.sum(), self.a.sum())

        b2 = self.b.toarray()
        self.assertEqual(type(b2), numpy.ndarray)
        self.assertEqual(b2.shape, self.b.shape)
        self.assertEqual(b2[0], 23)
        self.assertEqual(b2[5], 1)
        self.assertEqual(b2[8], -8)
        self.assertAlmostEqual(b2.sum(), self.b.sum())

    def testMin(self):
        self.assertEqual(self.A.min(), 0)
        self.assertEqual(self.B.min(), -1.23)
        self.assertEqual(self.C.min(), -1.23)
        self.assertEqual(self.D.min(), -10)
        self.assertTrue(math.isnan(self.E.min()))
        self.assertEqual(self.F.min(), -10)
        self.assertEqual(self.G.min(), -10)
        self.assertEqual(self.H.min(), -1.23)

        self.assertEqual(self.a.min(), -8)
        self.assertEqual(self.b.min(), -8)
        self.assertEqual(self.c.min(), 0)
        self.assertTrue(math.isnan(self.d.min()))

    def testMax(self):
        self.assertEqual(self.A.max(), 0)
        self.assertEqual(self.B.max(), 12.2)
        self.assertEqual(self.C.max(), 12.2)
        self.assertEqual(self.D.max(), 23.1)
        self.assertTrue(math.isnan(self.E.max()))
        self.assertEqual(self.F.max(), 23)
        self.assertEqual(self.G.max(), 23)
        self.assertEqual(self.H.max(), 12.2)

        self.assertEqual(self.a.max(), 23)
        self.assertEqual(self.b.max(), 23)
        self.assertEqual(self.c.max(), 0)
        self.assertTrue(math.isnan(self.d.max()))

    # @unittest.skip("")
    def testVar(self):
        self.assertEqual(self.A.var(), self.A.toarray().var())
        self.assertAlmostEqual(self.B.var(), self.B.toarray().var())
        self.assertAlmostEqual(self.C.var(), self.C.toarray().var())
        self.assertAlmostEqual(self.D.var(), self.D.toarray().var())

        self.assertAlmostEqual(self.F.var(), self.F.toarray().var())
        self.assertAlmostEqual(self.G.var(), self.G.toarray().var())
        self.assertAlmostEqual(self.H.var(), self.H.toarray().var())

        self.assertAlmostEqual(self.a.var(), self.a.toarray().var())
        self.assertAlmostEqual(self.b.var(), self.b.toarray().var())

    # @unittest.skip("")
    def testStd(self):
        self.assertEqual(self.A.std(), self.A.toarray().std())
        self.assertAlmostEqual(self.B.std(), self.B.toarray().std())
        self.assertAlmostEqual(self.C.std(), self.C.toarray().std())
        self.assertAlmostEqual(self.D.std(), self.D.toarray().std())

        self.assertAlmostEqual(self.F.std(), self.F.toarray().std())
        self.assertAlmostEqual(self.G.std(), self.G.toarray().std())
        self.assertAlmostEqual(self.H.std(), self.H.toarray().std())

        self.assertAlmostEqual(self.a.std(), self.a.toarray().std())
        self.assertAlmostEqual(self.b.std(), self.b.toarray().std())

    def testAbs(self):
        nptst.assert_array_equal(abs(self.A).toarray(), abs(self.A.toarray()))
        nptst.assert_array_equal(abs(self.B).toarray(), abs(self.B.toarray()))
        nptst.assert_array_equal(abs(self.C).toarray(), abs(self.C.toarray()))
        nptst.assert_array_equal(abs(self.D).toarray(), abs(self.D.toarray()))
        nptst.assert_array_equal(abs(self.F).toarray(), abs(self.F.toarray()))
        nptst.assert_array_equal(abs(self.G).toarray(), abs(self.G.toarray()))
        nptst.assert_array_equal(abs(self.H).toarray(), abs(self.H.toarray()))

        nptst.assert_array_equal(abs(self.a).toarray(), abs(self.a.toarray()))
        nptst.assert_array_equal(abs(self.b).toarray(), abs(self.b.toarray()))

    def testNeg(self):
        nptst.assert_array_equal((-self.A).toarray(), -self.A.toarray())
        nptst.assert_array_equal((-self.B).toarray(), -self.B.toarray())
        nptst.assert_array_equal((-self.C).toarray(), -self.C.toarray())
        nptst.assert_array_equal((-self.D).toarray(), -self.D.toarray())

        nptst.assert_array_equal((-self.F).toarray(), -self.F.toarray())
        nptst.assert_array_equal((-self.G).toarray(), -self.G.toarray())
        nptst.assert_array_equal((-self.H).toarray(), -self.H.toarray())

        nptst.assert_array_equal((-self.a).toarray(), -self.a.toarray())
        nptst.assert_array_equal((-self.b).toarray(), -self.b.toarray())

    def testAdd(self):
        # print(self.A.__add__(self.A._array))
        nptst.assert_array_equal((self.A + self.A).toarray(), self.A.toarray() * 2)
        nptst.assert_array_equal((self.B + self.B).toarray(), self.B.toarray() * 2)
        nptst.assert_array_equal((self.C + self.C).toarray(), self.C.toarray() * 2)
        nptst.assert_array_equal((self.D + self.D).toarray(), self.D.toarray() * 2)

        nptst.assert_array_equal((self.F + self.F).toarray(), self.F.toarray() * 2)
        nptst.assert_array_equal((self.G + self.G).toarray(), self.G.toarray() * 2)
        nptst.assert_array_equal((self.H + self.H).toarray(), self.H.toarray() * 2)

        A = csarray((5, 5))
        A[0, 1] = 4
        A[1, 3] = 2
        A[3, 3] = 1

        B = csarray((5, 5))
        B[0, 2] = 9.2
        B[2, 3] = -5
        B[3, 4] = 12

        nptst.assert_array_equal((A + B).toarray(), A.toarray() + B.toarray())

        nptst.assert_array_equal((self.a + self.a).toarray(), self.a.toarray() * 2)
        nptst.assert_array_equal((self.b + self.b).toarray(), self.b.toarray() * 2)
        nptst.assert_array_equal((self.c + self.c).toarray(), self.c.toarray() * 2)

    def testSub(self):
        nptst.assert_array_equal((self.A - self.A).toarray(), self.A.toarray() * 0)
        nptst.assert_array_equal((self.B - self.B).toarray(), self.B.toarray() * 0)
        nptst.assert_array_equal((self.C - self.C).toarray(), self.C.toarray() * 0)
        nptst.assert_array_equal((self.D - self.D).toarray(), self.D.toarray() * 0)
        nptst.assert_array_equal((self.F - self.F).toarray(), self.F.toarray() * 0)
        nptst.assert_array_equal((self.G - self.G).toarray(), self.G.toarray() * 0)
        nptst.assert_array_equal((self.H - self.H).toarray(), self.H.toarray() * 0)

        nptst.assert_array_equal((self.B * 2 - self.B).toarray(), self.B.toarray())

        A = csarray((5, 5))
        A[0, 1] = 4
        A[1, 3] = 2
        A[3, 3] = 1

        B = csarray((5, 5))
        B[0, 2] = 9.2
        B[2, 3] = -5
        B[3, 4] = 12

        nptst.assert_array_equal((A - B).toarray(), A.toarray() - B.toarray())

        nptst.assert_array_equal((self.a - self.a).toarray(), self.a.toarray() * 0)
        nptst.assert_array_equal((self.b - self.b).toarray(), self.b.toarray() * 0)
        nptst.assert_array_equal((self.c - self.c).toarray(), self.c.toarray() * 0)

    def testHadamard(self):
        nptst.assert_array_equal(
            (self.A.hadamard(self.A)).toarray(), (self.A.toarray()) ** 2
        )
        nptst.assert_array_equal(
            (self.B.hadamard(self.B)).toarray(), self.B.toarray() ** 2
        )
        nptst.assert_array_equal(
            (self.C.hadamard(self.C)).toarray(), self.C.toarray() ** 2
        )
        nptst.assert_array_equal(
            (self.D.hadamard(self.D)).toarray(), self.D.toarray() ** 2
        )

        nptst.assert_array_equal(
            (self.F.hadamard(self.F)).toarray(), self.F.toarray() ** 2
        )
        nptst.assert_array_equal(
            (self.G.hadamard(self.G)).toarray(), self.G.toarray() ** 2
        )
        nptst.assert_array_equal(
            (self.H.hadamard(self.H)).toarray(), self.H.toarray() ** 2
        )

        for storagetype in self.storagetypes:
            A = csarray((5, 5), storagetype=storagetype)
            A[0, 1] = 4
            A[2, 3] = -1.2
            A[1, 3] = 2
            A[3, 3] = 1

            B = csarray((5, 5), storagetype=storagetype)
            B[0, 2] = 9.2
            B[2, 3] = -5
            B[3, 4] = 12
            B[3, 3] = 12

            C = csarray((5, 5), storagetype=storagetype)

            nptst.assert_array_equal(
                (A.hadamard(B)).toarray(), A.toarray() * B.toarray()
            )
            nptst.assert_array_equal((A.hadamard(C)).toarray(), C.toarray())

        nptst.assert_array_equal(
            (self.a.hadamard(self.a)).toarray(), (self.a.toarray()) ** 2
        )
        nptst.assert_array_equal(
            (self.b.hadamard(self.b)).toarray(), (self.b.toarray()) ** 2
        )
        nptst.assert_array_equal(
            (self.c.hadamard(self.c)).toarray(), (self.c.toarray()) ** 2
        )

    def testReserve(self):
        for storagetype in self.storagetypes:
            A = csarray((5, 5), storagetype=storagetype)
            A.reserve(5)
            A[0, 1] = 4
            A[2, 3] = -1.2
            A[1, 3] = 2
            A[3, 3] = 1

    def testCompress(self):
        for storagetype in self.storagetypes:
            A = csarray((5, 5), storagetype=storagetype)
            A[0, 1] = 4
            A[2, 3] = -1.2
            A[1, 3] = 2
            A[3, 3] = 1
            A.compress()

    def testDot(self):
        for storagetype in self.storagetypes:
            A = csarray((5, 5), storagetype=storagetype)
            A[0, 1] = 4
            A[2, 3] = -1.2
            A[1, 3] = 2
            A[3, 3] = 1

            B = A.dot(A)
            nptst.assert_array_equal(B.toarray(), A.toarray().dot(A.toarray()))

        B = self.D.dot(self.D)
        nptst.assert_array_equal(B.toarray(), self.D.toarray().dot(self.D.toarray()))

        C = csarray((5, 2))
        for i in range(5):
            for j in range(2):
                C[i, j] = 1

        self.assertRaises(ValueError, C.dot, C)
        B = A.dot(C)
        nptst.assert_array_equal(B.toarray(), A.toarray().dot(C.toarray()))

        self.assertEqual((self.a.dot(self.a)), (self.a.dot(self.a)))
        self.assertEqual((self.b.dot(self.b)), (self.b.dot(self.b)))
        self.assertEqual((self.c.dot(self.c)), (self.c.dot(self.c)))

        # Now test dot product with ndarray
        D = self.D.toarray()
        B = self.D.dot(D)

        nptst.assert_array_equal(B, D.dot(D))

        A = numpy.random.rand(10, 5)
        B = numpy.random.rand(5, 6)

        C = A.dot(B)

        Ahat = csarray(A)
        Chat = Ahat.dot(B)

        nptst.assert_allclose(C, Chat, rtol=1e-14, atol=1e-14)

        # Try some random matrices
        numRuns = 10
        for i in range(numRuns):
            m = numpy.random.randint(1, 50)
            n = numpy.random.randint(1, 50)
            p = numpy.random.randint(1, 50)

            A = numpy.random.rand(m, n)
            B = numpy.random.rand(n, p)
            v = numpy.random.rand(n, p)

            C = A.dot(B)

            Ahat = csarray(A)
            Chat = Ahat.dot(B)

            nptst.assert_array_almost_equal(C, Chat)

            u = A.dot(v)
            uHat = Ahat.dot(v)
            nptst.assert_array_almost_equal(u, uHat)

    def testPdot(self):
        """
        D = numpy.ascontiguousarray(self.D.toarray())
        B = self.D.pdot(D)

        nptst.assert_array_equal(B, D.dot(D))

        A = numpy.random.rand(10, 5)
        B = numpy.random.rand(5, 6)

        C = A.dot(B)

        Ahat = csarray(A)
        Chat = Ahat.pdot(B)

        nptst.assert_array_almost_equal(C, Chat)
        """

        # Try some random matrices
        numRuns = 10
        for i in range(numRuns):
            m = numpy.random.randint(1, 50)
            n = numpy.random.randint(1, 50)
            p = numpy.random.randint(1, 50)

            A = numpy.random.rand(m, n)
            B = numpy.random.rand(n, p)
            v = numpy.random.rand(n, p)

            C = A.dot(B)

            Ahat = csarray(A, storagetype="row")
            Chat = Ahat.pdot(B)

            nptst.assert_array_almost_equal(C, Chat, 3)

            u = A.dot(v)
            uHat = Ahat.pdot(v)
            nptst.assert_array_almost_equal(u, uHat)

    def testTranspose(self):

        for storagetype in self.storagetypes:
            A = csarray((5, 5), storagetype=storagetype)
            A[0, 1] = 4
            A[2, 3] = -1.2
            A[1, 3] = 2
            A[3, 3] = 1

            self.assertEqual(type(A.T), csarray)

            nptst.assert_array_equal(A.transpose().toarray(), A.toarray().T)

        nptst.assert_array_equal(self.A.T.toarray(), self.A.toarray().T)
        nptst.assert_array_equal(self.B.T.toarray(), self.B.toarray().T)
        nptst.assert_array_equal(self.C.transpose().toarray(), self.C.toarray().T)
        nptst.assert_array_equal(self.D.transpose().toarray(), self.D.toarray().T)
        nptst.assert_array_equal(self.E.transpose().toarray(), self.E.toarray().T)
        nptst.assert_array_equal(self.F.transpose().toarray(), self.F.toarray().T)
        nptst.assert_array_equal(self.G.transpose().toarray(), self.G.toarray().T)
        nptst.assert_array_equal(self.H.transpose().toarray(), self.H.toarray().T)

    def testOnes(self):
        self.a.ones()
        nptst.assert_array_equal(self.a.toarray(), numpy.ones(self.a.shape[0]))

        self.A.ones()
        nptst.assert_array_equal(self.A.toarray(), numpy.ones(self.A.shape))

        self.G.ones()
        nptst.assert_array_equal(self.G.toarray(), numpy.ones(self.G.shape))

    def testRowInds(self):
        nptst.assert_array_equal(self.B.rowInds(0), numpy.array([1, 6]))
        nptst.assert_array_equal(self.B.rowInds(1), numpy.array([3]))

        nptst.assert_array_equal(self.H.rowInds(0), numpy.array([1, 6]))
        nptst.assert_array_equal(self.H.rowInds(1), numpy.array([3]))

        nptst.assert_array_equal(self.C.rowInds(0), numpy.array([1, 62]))
        nptst.assert_array_equal(self.C.rowInds(1), numpy.array([]))

    def testColInds(self):
        nptst.assert_array_equal(self.B.colInds(3), numpy.array([1, 3]))
        nptst.assert_array_equal(self.B.colInds(1), numpy.array([0]))

        nptst.assert_array_equal(self.H.colInds(3), numpy.array([1, 3]))
        nptst.assert_array_equal(self.H.colInds(1), numpy.array([0]))

        nptst.assert_array_equal(self.D.colInds(0), numpy.array([0, 2, 3]))
        nptst.assert_array_equal(self.D.colInds(2), numpy.array([]))

    def testValues(self):
        nptst.assert_array_equal(self.A.values(), self.A[self.A.nonzero()])
        nptst.assert_array_equal(self.B.values(), self.B[self.B.nonzero()])
        nptst.assert_array_equal(self.C.values(), self.C[self.C.nonzero()])
        nptst.assert_array_equal(self.D.values(), self.D[self.D.nonzero()])
        nptst.assert_array_equal(self.E.values(), self.E[self.E.nonzero()])
        nptst.assert_array_equal(self.F.values(), self.F[self.F.nonzero()])
        nptst.assert_array_equal(self.G.values(), self.G[self.G.nonzero()])
        nptst.assert_array_equal(self.H.values(), self.H[self.H.nonzero()])

    def testToScipyCsc(self):
        try:
            import scipy.sparse

            A = self.A.toScipyCsc()
            B = self.B.toScipyCsc()
            C = self.C.toScipyCsc()
            D = self.D.toScipyCsc()
            F = self.F.toScipyCsc()

            self.assertEqual(A.getnnz(), self.A.getnnz())
            self.assertEqual(B.getnnz(), self.B.getnnz())
            self.assertEqual(C.getnnz(), self.C.getnnz())
            self.assertEqual(D.getnnz(), self.D.getnnz())
            self.assertEqual(F.getnnz(), self.F.getnnz())

            # Now check elements are correct
            (rowInds, colInds) = self.B.nonzero()
            for i in range(rowInds.shape[0]):
                self.assertEqual(
                    B[rowInds[i], colInds[i]], self.B[rowInds[i], colInds[i]]
                )

            (rowInds, colInds) = self.C.nonzero()
            for i in range(rowInds.shape[0]):
                self.assertEqual(
                    C[rowInds[i], colInds[i]], self.C[rowInds[i], colInds[i]]
                )

            (rowInds, colInds) = self.D.nonzero()
            for i in range(rowInds.shape[0]):
                self.assertEqual(
                    D[rowInds[i], colInds[i]], self.D[rowInds[i], colInds[i]]
                )

            (rowInds, colInds) = self.F.nonzero()
            for i in range(rowInds.shape[0]):
                self.assertEqual(
                    F[rowInds[i], colInds[i]], self.F[rowInds[i], colInds[i]]
                )

        except ImportError as err:
            print(err)

    def testToScipyCsr(self):
        try:
            import scipy.sparse

            G = self.G.toScipyCsr()
            H = self.H.toScipyCsr()
            I = self.I.toScipyCsr()

            self.assertEqual(G.getnnz(), self.G.getnnz())
            self.assertEqual(H.getnnz(), self.H.getnnz())
            self.assertEqual(I.getnnz(), self.I.getnnz())

            # Now check elements are correct
            (rowInds, colInds) = self.G.nonzero()
            for i in range(rowInds.shape[0]):
                self.assertEqual(
                    G[rowInds[i], colInds[i]], self.G[rowInds[i], colInds[i]]
                )

            (rowInds, colInds) = self.H.nonzero()
            for i in range(rowInds.shape[0]):
                self.assertEqual(
                    H[rowInds[i], colInds[i]], self.H[rowInds[i], colInds[i]]
                )
        except ImportError as err:
            print(err)

    def testPut(self):
        A = csarray((10, 10))

        rowInds = numpy.array([1, 2, 5, 7], numpy.int32)
        colInds = numpy.array([4, 1, 9, 0], numpy.int32)
        vals = numpy.random.randn(rowInds.shape[0])

        A.put(vals, rowInds, colInds)

        for i in range(rowInds.shape[0]):
            self.assertEqual(A[rowInds[i], colInds[i]], vals[i])

        self.assertEqual(A.nnz, rowInds.shape[0])

    def testPutInit(self):
        A = csarray((10, 10), storagetype="col")

        rowInds = numpy.array([1, 2, 5, 7, 8, 1], numpy.int32)
        colInds = numpy.array([0, 0, 0, 1, 1, 2], numpy.int32)
        vals = numpy.random.randn(rowInds.shape[0])

        A.put(vals, rowInds, colInds, True)

        for i in range(rowInds.shape[0]):
            self.assertEqual(A[rowInds[i], colInds[i]], vals[i])

        self.assertEqual(A.nnz, rowInds.shape[0])

        # Test row format
        A = csarray((10, 10), storagetype="row")
        rowInds = numpy.array([1, 1, 2, 5, 7, 8], numpy.int32)
        colInds = numpy.array([0, 2, 0, 0, 1, 1], numpy.int32)

        A.put(vals, rowInds, colInds, True)

        for i in range(rowInds.shape[0]):
            self.assertEqual(A[rowInds[i], colInds[i]], vals[i])

        self.assertEqual(A.nnz, rowInds.shape[0])

        # Try a larger matrix
        numpy.random.seed(21)
        m = 1000000
        n = 1000000
        numInds = 1000

        inds = numpy.random.randint(0, m * n, numInds)
        inds = numpy.unique(inds)
        vals = numpy.random.randn(inds.shape[0])

        rowInds, colInds = numpy.unravel_index(inds, (m, n), order="F")
        rowInds = numpy.array(rowInds, numpy.int32)
        colInds = numpy.array(colInds, numpy.int32)
        A = csarray((m, n), storagetype="col")
        A.put(vals, rowInds, colInds)

        for i in range(vals.shape[0]):
            self.assertEqual(A[rowInds[i], colInds[i]], vals[i])

        self.assertEqual(A.nnz, vals.shape[0])

    def testBiCGSTAB(self):
        # This doesn't always converge
        numRuns = 10

        for i in range(numRuns):
            n = numpy.random.randint(5, 20)
            A = numpy.random.rand(n, n)
            x = numpy.random.rand(n)

            b = A.dot(x)

            A = csarray(A)

            x2, output = A.biCGSTAB(b, tol=10**-6, maxIter=n)

            if output == 0:
                nptst.assert_array_almost_equal(x, x2, 3)

        # Try with bad input
        m = 3
        n = 5
        A = numpy.random.rand(n, m)
        A = csarray(A)
        x = numpy.random.rand(m)
        b = A.dot(x)

        self.assertRaises(ValueError, A.biCGSTAB, b)

        A = numpy.random.rand(n, n)
        A = csarray(A)
        b = numpy.array(n + 1)
        self.assertRaises(ValueError, A.biCGSTAB, b)

    def testPrune(self):
        A = csarray((10, 10), storagetype="row")
        A[0, 1] = 10
        A[2, 3] = -0.4
        A[3, 3] = 0
        A[4, 8] = 0
        A.prune(10**-6)

        self.assertEqual(A.nnz, 2)
        A[0, 1] = 0
        self.assertEqual(A.nnz, 2)
        A.prune()
        self.assertEqual(A.nnz, 1)

    def testClip(self):
        B = self.B.clip(0, 12.2)

        self.assertEqual(B.nnz, 3)
        self.assertEqual(B[0, 1], 1)
        self.assertEqual(B[1, 3], 5.2)
        self.assertEqual(B[4, 4], 12.2)

        B = self.B.clip(2, 12.2)
        nptst.assert_array_equal(B.toarray(), numpy.clip(self.B.toarray(), 2, 12.2))

    def testPower(self):
        B2 = self.B.toarray() ** 2
        C = self.B.power(2)

        nptst.assert_array_almost_equal(C.toarray(), B2)

    def testSubmatrix(self):
        C = self.B.submatrix(0, 0, 4, 4)

        nptst.assert_array_equal(C.toarray(), self.B.toarray()[0:4, 0:4])

        self.assertRaises(ValueError, self.B.submatrix, -1, 0, 4, 4)
        self.assertRaises(ValueError, self.B.submatrix, 0, -1, 4, 4)
        self.assertRaises(ValueError, self.B.submatrix, 0, 0, 40, 4)
        self.assertRaises(ValueError, self.B.submatrix, 0, 0, 4, 40)

        C = self.B.submatrix(0, 0, 5, 7)
        nptst.assert_array_equal(C.toarray(), self.B.toarray())

        C = self.F.submatrix(1, 1, 4, 4)
        nptst.assert_array_equal(C.toarray(), self.F.toarray()[1:5, 1:5])

        C = self.G.submatrix(2, 3, 1, 2)
        nptst.assert_array_equal(C.toarray(), self.G.toarray()[2:3, 3:5])

    def testPickle(self):
        s = pickle.dumps(self.B)
        B = pickle.loads(s)

        for i in range(self.B.shape[0]):
            for j in range(self.B.shape[1]):
                self.assertEqual(B[i, j], self.B[i, j])

        self.assertEqual(self.B.dtype, B.dtype)
        self.assertEqual(self.B.storagetype, B.storagetype)

        s = pickle.dumps(self.A)
        A = pickle.loads(s)

        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                self.assertEqual(A[i, j], self.A[i, j])

        self.assertEqual(self.A.dtype, A.dtype)
        self.assertEqual(self.A.storagetype, A.storagetype)

        s = pickle.dumps(self.F)
        F = pickle.loads(s)

        for i in range(self.F.shape[0]):
            for j in range(self.F.shape[1]):
                self.assertEqual(F[i, j], self.F[i, j])

        self.assertEqual(self.F.dtype, F.dtype)
        self.assertEqual(self.F.storagetype, F.storagetype)

        # Test vector
        s = pickle.dumps(self.a)
        a = pickle.loads(s)

        for i in range(self.a.shape[0]):
            self.assertEqual(a[i], self.a[i])

        self.assertEqual(self.a.dtype, a.dtype)
        self.assertEqual(self.a.storagetype, a.storagetype)

        # Different dtype
        s = pickle.dumps(self.b)
        b = pickle.loads(s)

        for i in range(self.b.shape[0]):
            self.assertEqual(b[i], self.b[i])

        self.assertEqual(self.b.dtype, b.dtype)
        self.assertEqual(self.b.storagetype, b.storagetype)

    def testNonzeroRowsList(self):
        omegaList = self.B.nonzeroRowsList()

        for i in range(self.B.shape[0]):
            nptst.assert_array_equal(omegaList[i], self.B.toarray()[i, :].nonzero()[0])

        omegaList = self.C.nonzeroRowsList()

        for i in range(self.C.shape[0]):
            nptst.assert_array_equal(omegaList[i], self.C.toarray()[i, :].nonzero()[0])

        omegaList = self.D.nonzeroRowsList()

        for i in range(self.D.shape[0]):
            nptst.assert_array_equal(omegaList[i], self.D.toarray()[i, :].nonzero()[0])

        omegaList = self.F.nonzeroRowsList()

        for i in range(self.F.shape[0]):
            nptst.assert_array_equal(omegaList[i], self.F.toarray()[i, :].nonzero()[0])

    def testNonzeroRowsPtr(self):
        indPtr, colInds = self.B.nonzeroRowsPtr()

        for i in range(self.B.shape[0]):
            nptst.assert_array_equal(
                colInds[indPtr[i] : indPtr[i + 1]], self.B.toarray()[i, :].nonzero()[0]
            )

        indPtr, colInds = self.C.nonzeroRowsPtr()

        for i in range(self.C.shape[0]):
            nptst.assert_array_equal(
                colInds[indPtr[i] : indPtr[i + 1]], self.C.toarray()[i, :].nonzero()[0]
            )

        indPtr, colInds = self.D.nonzeroRowsPtr()

        for i in range(self.D.shape[0]):
            nptst.assert_array_equal(
                colInds[indPtr[i] : indPtr[i + 1]], self.D.toarray()[i, :].nonzero()[0]
            )

        indPtr, colInds = self.F.nonzeroRowsPtr()

        for i in range(self.F.shape[0]):
            nptst.assert_array_equal(
                colInds[indPtr[i] : indPtr[i + 1]], self.F.toarray()[i, :].nonzero()[0]
            )

    def testSin(self):
        # B, C, F, G
        B = self.B.sin()
        rowInds, colInds = self.B.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                B[rowInds[i], colInds[i]], numpy.sin(self.B[rowInds[i], colInds[i]])
            )

        C = self.C.sin()
        rowInds, colInds = self.C.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                C[rowInds[i], colInds[i]], numpy.sin(self.C[rowInds[i], colInds[i]])
            )

        F = self.F.sin()
        rowInds, colInds = self.F.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                F[rowInds[i], colInds[i]], numpy.sin(self.F[rowInds[i], colInds[i]])
            )

        G = self.G.sin()
        rowInds, colInds = self.G.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                G[rowInds[i], colInds[i]], numpy.sin(self.G[rowInds[i], colInds[i]])
            )

    def testCos(self):
        # B, C, F, G
        B = self.B.cos()
        rowInds, colInds = self.B.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                B[rowInds[i], colInds[i]], numpy.cos(self.B[rowInds[i], colInds[i]])
            )

        C = self.C.cos()
        rowInds, colInds = self.C.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                C[rowInds[i], colInds[i]], numpy.cos(self.C[rowInds[i], colInds[i]])
            )

        F = self.F.cos()
        rowInds, colInds = self.F.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                F[rowInds[i], colInds[i]], numpy.cos(self.F[rowInds[i], colInds[i]])
            )

        G = self.G.cos()
        rowInds, colInds = self.G.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                G[rowInds[i], colInds[i]], numpy.cos(self.G[rowInds[i], colInds[i]])
            )

    def testSign(self):
        # B, C, F, G
        B = self.B.sign()
        rowInds, colInds = self.B.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                B[rowInds[i], colInds[i]], numpy.sign(self.B[rowInds[i], colInds[i]])
            )

        C = self.C.sign()
        rowInds, colInds = self.C.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                C[rowInds[i], colInds[i]], numpy.sign(self.C[rowInds[i], colInds[i]])
            )

        F = self.F.sign()
        rowInds, colInds = self.F.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                F[rowInds[i], colInds[i]], numpy.sign(self.F[rowInds[i], colInds[i]])
            )

        G = self.G.sign()
        rowInds, colInds = self.G.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                G[rowInds[i], colInds[i]], numpy.sign(self.G[rowInds[i], colInds[i]])
            )

    def testCeil(self):
        # B, C, F, G
        B = self.B.ceil()
        rowInds, colInds = self.B.nonzero()
        self.assertEqual(B.dtype, numpy.float64)

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                B[rowInds[i], colInds[i]], numpy.ceil(self.B[rowInds[i], colInds[i]])
            )

        C = self.C.ceil()
        rowInds, colInds = self.C.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                C[rowInds[i], colInds[i]], numpy.ceil(self.C[rowInds[i], colInds[i]])
            )

        F = self.F.ceil()
        rowInds, colInds = self.F.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                F[rowInds[i], colInds[i]], numpy.ceil(self.F[rowInds[i], colInds[i]])
            )

        G = self.G.ceil()
        rowInds, colInds = self.G.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                G[rowInds[i], colInds[i]], numpy.ceil(self.G[rowInds[i], colInds[i]])
            )

    def testFloor(self):
        # B, C, F, G
        B = self.B.floor()
        rowInds, colInds = self.B.nonzero()
        self.assertEqual(B.dtype, numpy.float64)

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                B[rowInds[i], colInds[i]], numpy.floor(self.B[rowInds[i], colInds[i]])
            )

        C = self.C.floor()
        rowInds, colInds = self.C.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                C[rowInds[i], colInds[i]], numpy.floor(self.C[rowInds[i], colInds[i]])
            )

        F = self.F.floor()
        rowInds, colInds = self.F.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                F[rowInds[i], colInds[i]], numpy.floor(self.F[rowInds[i], colInds[i]])
            )

        G = self.G.floor()
        rowInds, colInds = self.G.nonzero()

        for i in range(rowInds.shape[0]):
            self.assertEqual(
                G[rowInds[i], colInds[i]], numpy.floor(self.G[rowInds[i], colInds[i]])
            )


if __name__ == "__main__":
    unittest.main()
