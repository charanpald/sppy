import logging
import sys
import unittest
import numpy
import numpy.testing as nptst
import sppy


class utilTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testDiag(self):

        a = numpy.array([1, 2, 3])

        X = sppy.diag(a)
        self.assertEqual(X.getnnz(), 3)
        self.assertEqual(X[0, 0], 1)
        self.assertEqual(X[1, 1], 2)
        self.assertEqual(X[2, 2], 3)

    def testEye(self):
        X = sppy.eye(10)

        for i in range(10):
            self.assertEqual(X[i, i], 1)

        self.assertEqual(X.dtype, numpy.float64)

        X = sppy.eye(10, numpy.int64)
        for i in range(10):
            self.assertEqual(X[i, i], 1)

        self.assertEqual(X.dtype, numpy.int64)

    def testRand(self):
        X = sppy.rand((10, 10), 0.1)

        self.assertEqual(X.shape, (10, 10))
        self.assertAlmostEqual(X.getnnz() / float(100), 0.1, 1)

        X = sppy.rand(100, 0.1)

        self.assertEqual(X.shape, (100,))
        self.assertAlmostEqual(X.getnnz() / float(100), 0.1, 1)

    def testZeros(self):
        X = sppy.zeros((5, 5))
        self.assertEqual(X.shape, (5, 5))
        nptst.assert_array_equal(X.toarray(), numpy.zeros((5, 5)))

        x = sppy.zeros(5)
        self.assertEqual(x.shape, (5,))
        nptst.assert_array_equal(x.toarray(), numpy.zeros((5,)))

    def testOnes(self):
        X = sppy.ones((5, 5))
        self.assertEqual(X.shape, (5, 5))
        nptst.assert_array_equal(X.toarray(), numpy.ones((5, 5)))

        x = sppy.ones((5,))
        self.assertEqual(x.shape, (5,))
        nptst.assert_array_equal(x.toarray(), numpy.ones((5,)))


if __name__ == "__main__":
    unittest.main()
