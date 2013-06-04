


class GeneralLinearOperator(object): 
    """
    A slightly more general form of LinearOperator 
    
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
    def asLinearOperator(X): 
        """
        Make a general linear operator from matrix X. 
        """
        
        def matvec(v): 
            return X.dot(v)
            
        def rmatvec(v): 
            return X.T.dot(v)
            
        def matmat(V): 
            return X.dot(V)
            
        def rmatmat(V): 
            return X.T.dot(V)
            
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, X.dtype)
