"""
Compare the performance of sparray and scipy.sparse. 
"""
import sys 
import time 
import numpy 
import logging 
from sparray import dyn_array 
from scipy.sparse import csc_matrix, csr_matrix


#Multiply two matrices 
#Get a selection of elements 
#Col and row slice 

def setGetBenchmark(ns): 
    numVals = 5000 
    
    vals = numpy.random.rand(numVals)
    numMethods = 3
    setTimeArray = numpy.zeros((numMethods, len(ns)))
    getTimeArray = numpy.zeros((numMethods, len(ns)))
    repetitions = 5 
    
    for s in range(len(ns)):
        n = ns[s]
        logging.debug("n="+str(n))
        numpy.random.seed(21)
        rowInds = numpy.random.randint(0, n, numVals)        
        colInds = numpy.random.randint(0, n, numVals)
        
        logging.debug("Benchmarking set from csc_matrix")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csc_matrix((n, n))            
            for j in range(numVals): 
                A1[rowInds[j], colInds[j]] = vals[j]
        setTimeArray[0, s] =  time.clock() - startTime
        
        logging.debug("Benchmarking get from csc_matrix")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csc_matrix((n, n))            
            for j in range(numVals): 
                v = A1[rowInds[j], colInds[j]]
        getTimeArray[0, s] =  time.clock() - startTime
        
        logging.debug("Benchmarking set from csr_matrix")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csr_matrix((n, n))            
            for j in range(numVals): 
                A1[rowInds[j], colInds[j]] = vals[j]
        setTimeArray[1, s] =  time.clock() - startTime  
        
        logging.debug("Benchmarking get from csr_matrix")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csc_matrix((n, n))            
            for j in range(numVals): 
                v = A1[rowInds[j], colInds[j]]
        getTimeArray[1, s] =  time.clock() - startTime
        
        logging.debug("Benchmarking set from dyn_array")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = dyn_array((n, n))            
            for j in range(numVals): 
                A1[rowInds[j], colInds[j]] = vals[j]
        setTimeArray[2, s] =  time.clock() - startTime
        
        logging.debug("Benchmarking get from dyn_array")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csc_matrix((n, n))            
            for j in range(numVals): 
                v = A1[rowInds[j], colInds[j]]
        getTimeArray[2, s] =  time.clock() - startTime
        
    return setTimeArray, getTimeArray

def addMultBenchmark(ns): 
    numVals = 1000 
    vals = numpy.random.rand(numVals)
    numMethods = 3
    addTimeArray = numpy.zeros((numMethods, len(ns)))
    multTimeArray = numpy.zeros((numMethods, len(ns)))
    repetitions = 1000 
    
    for s in range(len(ns)):
        n = ns[s]
        logging.debug("n="+str(n))
        numpy.random.seed(21)
        rowInds1 = numpy.random.randint(0, n, numVals)        
        colInds1 = numpy.random.randint(0, n, numVals)
        rowInds2 = numpy.random.randint(0, n, numVals)        
        colInds2 = numpy.random.randint(0, n, numVals)
             
        print("Benchmarking csc_matrix")
        A1 = csc_matrix((n, n))       
        A2 = csc_matrix((n, n)) 
        
        for j in range(numVals): 
            A1[rowInds1[j], colInds1[j]] = vals[j]
            A2[rowInds2[j], colInds2[j]] = vals[j]
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 + A2
        addTimeArray[0, s] =  time.clock() - startTime 
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 * A2
        multTimeArray[0, s] =  time.clock() - startTime 
        
        print("Benchmarking csr_matrix")
        A1 = csr_matrix((n, n))       
        A2 = csr_matrix((n, n)) 
        
        for j in range(numVals): 
            A1[rowInds1[j], colInds1[j]] = vals[j]
            A2[rowInds2[j], colInds2[j]] = vals[j]
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 + A2
        addTimeArray[1, s] =  time.clock() - startTime 
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 * A2
        multTimeArray[1, s] =  time.clock() - startTime 
        
        print("Benchmarking dyn_array")
        A1 = dyn_array((n, n))       
        A2 = dyn_array((n, n)) 
        
        for j in range(numVals): 
            A1[rowInds1[j], colInds1[j]] = vals[j]
            A2[rowInds2[j], colInds2[j]] = vals[j]
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 + A2
        addTimeArray[2, s] =  time.clock() - startTime 
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1.hadamard(A2)
        multTimeArray[2, s] =  time.clock() - startTime 
        
    return addTimeArray * 1000.0, multTimeArray*1000.0
    
def meanSumBenchmark(ns): 
    numVals = 1000 
    vals = numpy.random.rand(numVals)
    numMethods = 3
    sumTimeArray = numpy.zeros((numMethods, len(ns)))
    repetitions = 1000 
    
    for s in range(len(ns)):
        n = ns[s]
        logging.debug("n="+str(n))
        numpy.random.seed(21)
        rowInds = numpy.random.randint(0, n, numVals)        
        colInds = numpy.random.randint(0, n, numVals)
             
        print("Benchmarking csc_matrix")
        A = csc_matrix((n, n))       
        
        for j in range(numVals): 
            A[rowInds[j], colInds[j]] = vals[j]

        startTime = time.clock()
        for i in range(repetitions): 
            b = A.sum()
        sumTimeArray[0, s] =  time.clock() - startTime 
        
        print("Benchmarking csr_matrix")
        A = csr_matrix((n, n))       
        
        for j in range(numVals): 
            A[rowInds[j], colInds[j]] = vals[j]

        startTime = time.clock()
        for i in range(repetitions): 
            b = A.sum()
        sumTimeArray[1, s] =  time.clock() - startTime 
        
        print("Benchmarking dyn_array")
        A = dyn_array((n, n))       
        
        for j in range(numVals): 
            A[rowInds[j], colInds[j]] = vals[j]

        startTime = time.clock()
        for i in range(repetitions): 
            b = A.sum()
        sumTimeArray[2, s] =  time.clock() - startTime 
        
    return sumTimeArray *  1000.0

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
numpy.set_printoptions(suppress=True)

ns = [100, 500, 1000, 5000, 10000, 50000, 100000]

setTimeArray, getTimeArray = setGetBenchmark(ns)
addTimeArray, multTimeArray = addMultBenchmark(ns)
sumTimeArray = meanSumBenchmark(ns)

print(setTimeArray)
print(getTimeArray)
print(addTimeArray)
print(multTimeArray)
print(sumTimeArray)