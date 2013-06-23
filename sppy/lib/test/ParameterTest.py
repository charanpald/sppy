

import unittest
import numpy
import logging
from sppy.lib.Parameter import Parameter

class  ParameterTest(unittest.TestCase):
    def testCheckInt(self):
        min = 0
        max = 5
        i = 2

        Parameter.checkInt(i, min, max)
        Parameter.checkInt(min, min, max)
        Parameter.checkInt(max, min, max)
        Parameter.checkInt(i, i, i)

        self.assertRaises(ValueError, Parameter.checkInt, i, max, min)
        self.assertRaises(ValueError, Parameter.checkInt, i, float(min), max)
        self.assertRaises(ValueError, Parameter.checkInt, i, min, float(max))
        #self.assertRaises(ValueError, Parameter.checkInt, 2.0, min, max)
        self.assertRaises(ValueError, Parameter.checkInt, -1, min, max)
        self.assertRaises(ValueError, Parameter.checkInt, 6, min, max)

        #Check half ranges such as [0, inf]
        Parameter.checkInt(i, min, float("inf"))
        Parameter.checkInt(i, float("-inf"), max)

        #Check use of numpy int32
        min = numpy.int32(0)
        max = numpy.int32(5)
        i = numpy.int32(2)

        Parameter.checkInt(i, min, max)
        Parameter.checkInt(min, min, max)
        Parameter.checkInt(max, min, max)
        Parameter.checkInt(i, i, i)

        #Test using an array with 1 int
        i = numpy.array([1], numpy.int)
        logging.debug((type(i)))
        self.assertRaises(ValueError, Parameter.checkInt, i, min, max)

    def testCheckFloat(self):
        min = 0.0
        max = 5.0
        i = 2.0

        Parameter.checkFloat(i, min, max)
        Parameter.checkFloat(min, min, max)
        Parameter.checkFloat(max, min, max)
        Parameter.checkFloat(i, i, i)

        self.assertRaises(ValueError, Parameter.checkFloat, i, max, min)
        self.assertRaises(ValueError, Parameter.checkFloat, i, int(min), max)
        self.assertRaises(ValueError, Parameter.checkFloat, i, min, int(max))
        self.assertRaises(ValueError, Parameter.checkFloat, 2, min, max)
        self.assertRaises(ValueError, Parameter.checkFloat, -1, min, max)
        self.assertRaises(ValueError, Parameter.checkFloat, 6, min, max)

        #Check half ranges such as [0, inf]
        Parameter.checkFloat(i, min, float("inf"))
        Parameter.checkFloat(i, float("-inf"), max)

        #Check use of numpy float64
        min = numpy.float64(0.0)
        max = numpy.float64(5.0)
        i = numpy.float64(2.0)

        Parameter.checkFloat(i, min, max)
        Parameter.checkFloat(min, min, max)
        Parameter.checkFloat(max, min, max)
        Parameter.checkFloat(i, i, i)

    def testCheckString(self):
        s = "a"
        lst = ["a", "b", "c"]

        Parameter.checkString("a", lst)
        Parameter.checkString("b", lst)
        Parameter.checkString("c", lst)

        self.assertRaises(ValueError, Parameter.checkString, "d", lst)
        self.assertRaises(ValueError, Parameter.checkString, 5, lst)
        self.assertRaises(ValueError, Parameter.checkString, "a", s)

    def testCheckList(self):
        lst = [1, 2, 3, 2, 2]
        Parameter.checkList(lst, Parameter.checkInt, [1, 3])

        lst = [1, 2, 3, 2, 4]
        self.assertRaises(ValueError, Parameter.checkList, lst, Parameter.checkInt, [1, 3])

        lst = [1, 2, 3, 2, 0]
        self.assertRaises(ValueError, Parameter.checkList, lst, Parameter.checkInt, [1, 3])

        lst = [1, 2, 3, 2, 1.2]
        self.assertRaises(ValueError, Parameter.checkList, lst, Parameter.checkInt, [1, 3])

        lst = "a"
        self.assertRaises(ValueError, Parameter.checkList, lst, Parameter.checkInt, [1, 3])

        lst = [0.1, 0.6, 1.4]
        Parameter.checkList(lst, Parameter.checkFloat, [0.1, 3.0])

        #Test use of array 
        lst = numpy.array([0.1, 0.6, 1.4])
        Parameter.checkList(lst, Parameter.checkFloat, [0.1, 3.0])

        lst = numpy.array([[0.1, 0.6, 1.4]])
        self.assertRaises(ValueError, Parameter.checkList, lst, Parameter.checkFloat, [0.1, 3.0])


    def checkBoolean(self):
        a = True
        b = False
        c = 0
        d = 1
        e = "s"

        Parameter.checkBoolean(a)
        Parameter.checkBoolean(b)

        self.assertRaises(ValueError, Parameter.checkBoolean, c)
        self.assertRaises(ValueError, Parameter.checkBoolean, d)
        self.assertRaises(ValueError, Parameter.checkBoolean, e)

    def checkClass(self):
        a = VertexList(10, 1)
        b = 2
        c = True
        d = SparseGraph(a)

        Parameter.checkClass(a, VertexList)
        Parameter.checkClass(b, int)
        Parameter.checkClass(c, bool)
        Parameter.checkClass(d, SparseGraph)

        self.assertRaises(ValueError, Parameter.checkClass, a, SparseGraph)
        self.assertRaises(ValueError, Parameter.checkClass, b, VertexList)

    def testCheckSymmetric(self):

        W = numpy.random.rand(5, 5)
        W = W.T + W 
        
        Parameter.checkSymmetric(W)

        W[0, 1] += 0.1
        self.assertRaises(ValueError, Parameter.checkSymmetric, W)
        self.assertRaises(ValueError, Parameter.checkSymmetric, W, 0.1)

        Parameter.checkSymmetric(W, 0.2)



if __name__ == '__main__':
    unittest.main()

