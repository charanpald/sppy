 


class GeneralLinearOperator(object): 
    """
    A more general form of LinearOperator in scipy.linalg. Can be used 
    with many of the scipy functions. 
    
    The new operation is rmatmat which is X.T V. 
    """
    def __init__(self, shape, matvec, rmatvec=None, matmat=None, rmatmat=None, add=None, dtype=None): 
        
        self.shape = shape 
        self.matvec = matvec 
        self.rmatvec = rmatvec 
        self.matmat = matmat 
        self.rmatmat = rmatmat
        self.add = add
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
                
            def add(V): 
                return X + V 
        else:
            def matvec(v): 
                return X.pdot(v)
                
            def rmatvec(v): 
                return X.T.pdot(v)
                
            def matmat(V): 
                return X.pdot(V)
                
            def rmatmat(V): 
                return X.T.pdot(V)
            
            def add(V): 
                return X + V 
            
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, add, X.dtype)
        
    @staticmethod
    def asLinearOperatorSum(X, Y): 
        """
        Take two linear operators X and Y, and operate on their sum, using lazy 
        evaluation. 
        """
        if X.shape != Y.shape: 
            raise ValueError("Shapes of X and Y do not match: " + str(X.shape) + " " + str(Y.shape))
        
        def matvec(v): 
            return X.matvec(v) + Y.matvec(v)
            
        def rmatvec(v): 
            return X.rmatvec(v) + Y.rmatvec(v)
            
        def matmat(V): 
            return X.matmat(V) + Y.matmat(V)
            
        def rmatmat(V): 
            return X.rmatmat(V) + Y.rmatmat(V)
            
        def add(V): 
            return Y.add(X.add(V)) 
            
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, add, X.dtype)
