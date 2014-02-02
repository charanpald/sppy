
"""
Some functions to read and write matrix market files. 
"""


def mmwrite(filename, A, comment='', field=None, precision=None): 
    """
    Write a csarray object in matrix market format. 
    """
    
    if field == None: 
        if A.dtype.kind == "i": 
            field = "integer"
        else: 
            field = "real"
    
    if field == "real": 
        if precision != None: 
            fmtStr = "%." + str(precision) + "f"
        else: 
            fmtStr = "%f"
    else: 
        fmtStr = "%d"
    
    fileObj = open(filename, "w")
    
    fileObj.write("%%MatrixMarket matrix coordinate " + field + " general\n")
    fileObj.write("%%" + comment + "\n")
    fileObj.write("%%\n")
    
    fileObj.write(str(A.shape[0]) + " " + str(A.shape[1]) + " " + str(A.nnz) + "\n")
    
    rowInds, colInds = A.nonzero()
    vals = A.values()
    
    for i, val in enumerate(vals):
        valStr = fmtStr % val
        fileObj.write(str(rowInds[i]+1) + " " + str(colInds[i]+1) + " " + valStr + "\n")
    
    fileObj.close()
        
    
    
    
    