
import numpy 
import string
from sppy import csarray
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
        
def mmread(filename, storagetype="col"): 
    """
    Read from a matrix market file. Note that we do not allow comments (%) in the 
    body of the elements, only in the header lines. 
    """
    fileObj = open(filename, "r")
    
    line = fileObj.readline()
    line.strip().strip("#")
    
    vals = line.split()
    
    if vals[3] == "integer": 
        dtype = numpy.int 
    elif vals[3] == "real": 
        dtype = numpy.float 
    else:
        raise ValueError("Invalid data type: " + vals[3])
    
    line = fileObj.readline()
    while string.find(line, "%") == 0: 
        line = fileObj.readline()    
    
    vals = line.split()
    m = int(vals[0])
    n = int(vals[1])
    nnz = int(vals[2])
    
    rowInds = numpy.zeros(nnz, numpy.int32)
    colInds = numpy.zeros(nnz, numpy.int32)
    values = numpy.zeros(nnz, dtype)
    
    #This reads the rest of the file reasonably quickly 
    Z = numpy.fromfile(fileObj, sep=" ")
    Z = Z.reshape((nnz, 3))

    rowInds = numpy.array(Z[:, 0], numpy.int32)-1 
    colInds = numpy.array(Z[:, 1], numpy.int32)-1 
    values = numpy.ascontiguousarray(Z[:, 2], dtype=dtype)

    fileObj.close()

    A = csarray((m, n), dtype=dtype, storagetype=storagetype)
    A.reserve(nnz)
    A.put(values, rowInds, colInds, init=True)
    
    return A     