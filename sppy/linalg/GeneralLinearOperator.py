 


class GeneralLinearOperator(object): 
    """
    A more general form of LinearOperator in scipy.linalg. Can be used 
    with many of the scipy functions. 
    
    The new operation is rmatmat which is X.T V. 
    """
    def __init__(self, shape, matvec, rmatvec=None, matmat=None, rmatmat=None, dtype=None): 
        
        self.shape = shape 
        self.matvec = matvec 
        self.rmatvec = rmatvec 
        self.matmat = matmat 
        self.rmatmat = rmatmat
        self.dtype = dtype 
        
    @staticmethod 
    def asLinearOperator(X, parallel=False): 
        """
        Make a general linear operator from csarray X. 
        """
        if not parallel: 
            def matvec(v): 
                return X.dot(v)
                
            def rmatvec(v): 
                return X.T.dot(v)
                
            def matmat(V): 
                return X.dot(V)
                
            def rmatmat(V): 
                return X.T.dot(V)
        else:
            def matvec(v): 
                return X.pdot(v)
                
            def rmatvec(v): 
                return X.T.pdot(v)
                
            def matmat(V): 
                return X.pdot(V)
                
            def rmatmat(V): 
                return X.T.pdot(V)
            
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, X.dtype)
