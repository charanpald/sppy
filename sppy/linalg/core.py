import gc 
import numpy 
from sppy.linalg.GeneralLinearOperator import GeneralLinearOperator
from sppy.lib.Parameter import Parameter 
import sppy.csarray as csarray 

def biCGSTAB(A, b, maxIter=1000, tol=10**-6): 
    """
    Solve the linear system of equations given by A x = b where A is a csarray 
    and b is a numpy array of the same dtype. Uses the Iterative stabilized 
    bi-conjugate gradient method. 
    
    :param A: A csarray object of size n x n 
    
    :param b: A numpy array of length n of the same dtype as A. 
    
    :param maxIter: The maximum number of iteartions of the method 
    
    :param tol: The error tolerance
    
    :return x: A numpy array corresponding to the solution vector. 
    
    :return i: The output code: 0 = success, 1 = numerical Issue, 2 = no convergence, 3 = invalid input
    """
    return A._array.biCGSTAB(b, maxIter, tol)

def norm(A, ord=None): 
    """
    This function returns the Frobenius norm of the input A, which is defined as 
    sqrt(sum A_ij^2).  

    :param A: A csarray object.   
    
    :param ord: The type of norm required, currently ignored. 
    
    :return: The Frobenius norm of A. 
    """
    return A._array.norm()

def rsvd(A, k, p=10, q=2, omega=None): 
    """
    Compute the randomised SVD using the algorithm on page 9 of Halko et al., 
    Finding Structure with randomness: stochastic algorithms for constructing 
    approximate matrix decompositions, 2009.         
    
    Finds the partial SVD of a sparse or dense matrix A, resolving the largest k 
    singular vectors/values, using exponent q and k+p projections. Returns the 
    left and right singular vectors, and the singular values. The resulting matrix 
    can be approximated using A ~ U s V.T. To improve the approximation quality 
    for a fixed k, increase p or q.
    
    :param A: A sparse or dense matrix or GeneralLinearOperator 
    
    :param k: The number of singular values and random projections
    
    :param p: The oversampling parameter 
    
    :param q: The exponent for the projections.
    
    :param omega: An initial matrix to perform random projections onto with at least k columns 
    
    :return U: The left singular vectors 
    
    :return s: The singular values 
    
    :return V: The right singular vectors
    """
    Parameter.checkInt(k, 1, float("inf"))
    Parameter.checkInt(p, 0, float("inf"))
    Parameter.checkInt(q, 0, float("inf"))        

    if isinstance(A, GeneralLinearOperator): 
        L = A 
    else: 
        L = GeneralLinearOperator.asLinearOperator(A) 
    
    n = L.shape[1]
    if omega == None: 
        omega = numpy.random.randn(n, k+p)
    else: 
        omega = numpy.c_[omega, numpy.random.randn(n, p+k - omega.shape[1])]
    
    Y = L.matmat(omega)
    Q, R = numpy.linalg.qr(Y)
    del omega 

    for i in range(q):
        Y = L.rmatmat(Q)
        Q, R = numpy.linalg.qr(Y)
        gc.collect() 
        
        Y = L.matmat(Q)
        Q, R = numpy.linalg.qr(Y)
        gc.collect() 
    
    del Y 
    del R 
    gc.collect() 
    
    B = L.rmatmat(Q).T
    U, s, V = numpy.linalg.svd(B, full_matrices=False)
    del B 
    V = V.T
    U = Q.dot(U)

    U = U[:, 0:k]
    s = s[0:k]
    V = V[:, 0:k]        
    
    return U, s, V 
        
        
