
import numpy

class Util(object):
    '''
    A class with some general useful function that don't fit in anywhere else.
    '''
    def __init__(self):
        pass

    @staticmethod 
    def indSvd(P, s, Q, inds):
        """
        Take the output of numpy.linalg.svd and return the eigenvalue and vectors
        sorted in order indexed by ind.
        """
        if inds.shape[0] != 0:
            P = P[:, inds]
            s = s[inds]
            Q = Q.conj().T
            Q = Q[:, inds]
        else:
            P = numpy.zeros((P.shape[0], 0))
            s = numpy.zeros(0)
            Q = Q.conj().T
            Q = numpy.zeros((Q.shape[0], 0))

        return P, s, Q
    
 