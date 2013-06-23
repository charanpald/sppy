'''
Created on 31 Jul 2009

@author: charanpal
'''

import numpy
import logging

class Parameter(object):
    """
    A class of static methods which are useful for type checking. 
    """
    def __init__(self):
        pass

    @staticmethod
    def checkClass(obj, objectType, softCheck = False):
        """
        Check if an object belongs to a particular class and raise a ValueError
        if it does not.

        :param objectType: The object to test. `
        """
        if not isinstance(obj, objectType):
            return Parameter.whatToDo("Expecting object of type " + str(objectType) + " but received " + str(obj.__class__ ), softCheck)
        return True


    @staticmethod
    def checkBoolean(val, softCheck = False):
        """
        Check if val is a boolean and raise a ValueError if it is not.

        :param val: The value to test. 
        :type val: :class:`bool`
        """
        if type(val) != bool:
            return Parameter.whatToDo("Expecting boolean but received " + str(type(val)), softCheck)
        return True

    @staticmethod
    def checkIndex(i, min, max, softCheck = False):
        """
        Check if i is an integer value between min inclusive and max exclusive and raise a
        ValueError if it is not. If one requires an open ended range, then min
        or max can be float('inf') for example.

        :param i: The value to test, such that min <= i < max.
        :type i: :class:`int`

        :param min: The minimum value of i.
        :type min: :class:`int`

        :param max: The maximum+1 value of i.
        :type max: :class:`int`
        """
        if not (type(min) in Parameter.intTypes or min == float("-inf")):
            return Parameter.whatToDo("Minimum value is not an integer: " + str(min), softCheck)
        if not (type(max) in Parameter.intTypes or max == float("inf")):
            return Parameter.whatToDo("Maximum value is not an integer: " + str(max), softCheck)
        if (type(i) not in Parameter.intTypes):
            return Parameter.whatToDo("Invalid parameter value (not int): " + str(i) + " is " + str(type(i)), softCheck)

        if i < min or i >= max:
            return Parameter.whatToDo("Invalid parameter value: " + str(i) + " not in [" + str(min)+ ", " + str(max-1) + "]", softCheck)
        return True

    @staticmethod
    def checkInt(i, min, max, softCheck = False):
        """
        Check if i is an integer value between min and max inclusive and raise a
        ValueError if it is not. If one requires an open ended range, then min
        or max can be float('inf') for example.

        :param i: The value to test, such that min <= i <= max.
        :type i: :class:`int`

        :param min: The minimum value of i.
        :type min: :class:`int`

        :param max: The maximum value of i.
        :type max: :class:`int`
        """
        if not (type(min) in Parameter.intTypes or min == float("-inf")):
            return Parameter.whatToDo("Minimum value is not an integer: " + str(min), softCheck)
        if not (type(max) in Parameter.intTypes or max == float("inf")):
            return Parameter.whatToDo("Maximum value is not an integer: " + str(max), softCheck)
        if (type(i) not in Parameter.intTypes):
            return Parameter.whatToDo("Invalid parameter value (not int): " + str(i) + " is " + str(type(i)), softCheck)
 
        if i < min or i > max:
            return Parameter.whatToDo("Invalid parameter value: " + str(i) + " not in [" + str(min)+ ", " + str(max) + "]" , softCheck)
        return True
        
    @staticmethod
    def checkFloat(i, min, max, softCheck = False):
        """
        Check if i is an floating point value between min and max inclusive and raise a
        ValueError if it is not. If one requires an open ended range, then min
        or max can be float('inf') for example.

        :param i: The value to test, such that min <= i <= max.
        :type i: :class:`float`

        :param min: The minimum value of i.
        :type min: :class:`float`

        :param max: The maximum value of i.
        :type max: :class:`float`
        """
        if type(min) not in Parameter.floatTypes:
            return Parameter.whatToDo("Minimum value is not a float: " + str(min), softCheck)
        if type(max) not in Parameter.floatTypes:
            return Parameter.whatToDo("Maximum value is not a float: " + str(max), softCheck)
        if type(i) not in Parameter.floatTypes:
            return Parameter.whatToDo("Invalid parameter value (not float): " + str(i) + " is " + str(type(i)), softCheck)

        if i < min or i > max:
            return Parameter.whatToDo("Invalid parameter value: " + str(i) + " not in [" + str(min)+ ", " + str(max) + "]" , softCheck)
        return True

    @staticmethod
    def checkString(s, strList, softCheck = False):
        """
        Check if s is an string in strList and raise a ValueError if it is not.

        :param s: The string to test. 
        :type s: :class:`str`

        :param strList: A list of valid strings.
        :type strList: :class:`list`
        """
        if type(strList) != list:
            return Parameter.whatToDo("Second parameter is required to be a list: " + str(strList), softCheck)

        if type(s) != str or s not in strList:
            return Parameter.whatToDo("Invalid string: " + str(s), softCheck)
        return True

    @staticmethod
    def checkList(lst, func, params, softCheck = False):
        """
        Check if a list/ndarray obeys the constaints given by func. For example, in order
        to check if a list/ndarray lst contains integers between 0 and 10, one can use
        Parameter.checkList(lst, Parameter.checkInt, [0, 10]). The first argument
        of checkInt is taken from the input list and the remaining ones correspond to 
        the 3rd parameter of checkList. If an array is used as input, then it must be 1D. 

        :param lst: The list to test.
        :type lst: :class:`list`

        :param func: A function which raises an exception when an invalid list entry is encountered, otherwise does nothing.

        :param params: A list of parameter to the checking function. 
        :type lst: :class:`list`
        """
        if type(lst) != list and type(lst) != numpy.ndarray:
            return Parameter.whatToDo("First parameter is required to be a list or array, not " + str(type(lst)), softCheck)
        if type(lst) == numpy.ndarray and lst.ndim != 1:
            return Parameter.whatToDo("First parameter must be 1 dimensional: " + str(lst.shape), softCheck)

        for i in lst:
            func(i, *params)
        return True
            
    @staticmethod
    def checkArray(array, softCheck = False, arrayInfo = ""):
        """
        Check that an array contains no nan or inf values
        """
        if numpy.isinf(array).any():
            return Parameter.whatToDo("The array " + arrayInfo + " contains a 'inf' value", softCheck)
        if numpy.isnan(array).any():
            return Parameter.whatToDo("The array " + arrayInfo + " contains a 'NaN' value", softCheck)
        if not numpy.isrealobj(array):
            return Parameter.whatToDo("The array " + arrayInfo + " has an imaginary part", softCheck)
        return True

    @staticmethod
    def checkSymmetric(A, tol=10**-6, softCheck = False, arrayInfo = "?"):
        """
        Takes as input a matrix A and checks if it is symmetric by verifying whether
        ||A - A.T||_F < tol.
        """
        if type(A)==numpy.ndarray and ((A.T - A)**2).sum() > tol**2:
            return Parameter.whatToDo("A=" + arrayInfo + " is not a symmetric matrix, ||A.T - A||_F^2 = " + str(((A.T - A)**2).sum()), softCheck)
        return True
        
    @staticmethod
    def checkOrthogonal(A, tol=10**-6, softCheck = False, investigate = False, arrayInfo = "?"):
        """
        Takes as input a matrix A and checks if it is orthogonal by verifying whether
        ||A*A.T - Id||_F < tol.
        """
        diff = numpy.linalg.norm(A.conj().T.dot(A) - numpy.eye(A.shape[1]))
        if diff > tol:
            try :
                return Parameter.whatToDo("Non-orthogonal matrix A=" + arrayInfo + ": ||A*A.T - Id||_F = " + str(diff), softCheck)
            finally : # s.t. when raising an error, the investigation appears after
                if investigate:
                    Diff = A.conj().T.dot(A) - numpy.eye(A.shape[1])
                    print("indexes:\n", (abs(Diff)>tol).nonzero())
                    print("values\n", Diff[abs(Diff)>tol])
        return True
        
    @staticmethod
    def whatToDo(msg, softCheck = False):
        logging.warn(msg)
        if softCheck:
            return False
        else:
            raise ValueError("Check Error: " + msg)
        
    #Note that there are two types of int32 for some weird reason. 
    a = numpy.int8(1)
    b = numpy.int32(2)
    intTypes = [int, numpy.int8, numpy.int16, type(a+b), type(b+a), numpy.int64]
    floatTypes = [float, numpy.float32, numpy.float64]
