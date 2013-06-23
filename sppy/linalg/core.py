import gc 
import numpy 
from sppy.linalg.GeneralLinearOperator import GeneralLinearOperator
from sppy.lib.Parameter import Parameter 

def rsvd(X, k, p=10, q=2, omega=None): 
    """
    Compute the randomised SVD using the algorithm on page 9 of Halko et al., 
    Finding Structure with randomness: stochastic algorithms for constructing 
    approximate matrix decompositions, 2009.         
    
    Finds the partial SVD of a sparse or dense matrix X, resolving the largest k 
    singular vectors/values, using exponent q and k+p projections. Returns the 
    left and right singular vectors, and the singular values. The resulting matrix 
    can be approximated using X ~ U s V.T. To improve the approximation quality 
    for a fixed k, increase p or q.
    
    :param X: A sparse or dense matrix or GeneralLinearOperator 
    
    :param k: The number of singular values and random projections
    
    :param p: The oversampling parameter 
    
    :param q: The exponent for the projections.
    
    :param omega: An initial matrix to perform random projections onto with at least k columns 
    """
    Parameter.checkInt(k, 1, float("inf"))
    Parameter.checkInt(p, 0, float("inf"))
    Parameter.checkInt(q, 0, float("inf"))        

    if isinstance(X, GeneralLinearOperator): 
        L = X 
    else: 
        L = GeneralLinearOperator.asLinearOperator(X) 
    
    n = L.shape[1]
    if omega == None: 
        omega = numpy.random.randn(n, k+p)
    else: 
        omega = numpy.c_[omega, numpy.random.randn(n, p+k - omega.shape[1])]
    
    Y = L.matmat(omega)
    del omega 

    for i in range(q):
        Y = L.rmatmat(Y)
        gc.collect() 
        Y = L.matmat(Y)
        gc.collect() 
    
    Q, R = numpy.linalg.qr(Y)
    
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
        
        
def norm(X, ord=None): 
    """
    This function returns the Frobenius norm of the input X, which is defined as 
    sqrt(sum X_ij^2).  

    :param X: A csarray object.   
    
    :param ord: The type of norm required, currently ignored. 
    """
    return X._array.norm()