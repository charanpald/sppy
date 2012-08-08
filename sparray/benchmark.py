"""
Compare the performance of sparray and scipy.sparse. 
"""
import sys 
import time 
import numpy 
import logging 
from sparray import csarray 
import matplotlib.pyplot as plt 
from scipy.sparse import csc_matrix, csr_matrix


#Multiply two matrices 
#Get a selection of elements 
#Col and row slice 

def getRandomElements(numVals, shape, seed=21): 
    numpy.random.seed(seed) 
    vals = numpy.random.rand(numVals)
    
    rowInds = numpy.random.randint(0, shape[0], numVals)        
    colInds = numpy.random.randint(0, shape[1], numVals)
    
    return rowInds, colInds, vals

def setRandomElements(numVals, A, seed=21):
    rowInds, colInds, vals = getRandomElements(numVals, A.shape, seed)
    
    for j in range(numVals): 
        A[rowInds[j], colInds[j]] = vals[j]

def setGetBenchmark(ns): 
    numVals = 1000 
    
    numMethods = 3
    setTimeArray = numpy.zeros((numMethods, len(ns)))
    getTimeArray = numpy.zeros((numMethods, len(ns)))
    repetitions = 5 
    
    for s in range(len(ns)):
        n = ns[s]
        logging.debug("n="+str(n))
        
        rowInds, colInds, vals = getRandomElements(numVals, (n, n))
        
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
        
        logging.debug("Benchmarking set from csarray")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csarray((n, n))     
            for j in range(numVals): 
                A1[rowInds[j], colInds[j]] = vals[j]
        setTimeArray[2, s] =  time.clock() - startTime
        
        logging.debug("Benchmarking get from csarray")
        startTime = time.clock()
        for i in range(repetitions):
            A1 = csc_matrix((n, n))            
            for j in range(numVals): 
                v = A1[rowInds[j], colInds[j]]
        getTimeArray[2, s] =  time.clock() - startTime
        
    return setTimeArray, getTimeArray

def addMultBenchmark(ns): 
    numVals = 1000 
    numMethods = 3
    addTimeArray = numpy.zeros((numMethods, len(ns)))
    multTimeArray = numpy.zeros((numMethods, len(ns)))
    repetitions = 1000 
    
    for s in range(len(ns)):
        n = ns[s]
        logging.debug("n="+str(n))
             
        print("Benchmarking +/* of csc_matrix")
        A1 = csc_matrix((n, n))       
        A2 = csc_matrix((n, n)) 
        setRandomElements(numVals, A1, 21)
        setRandomElements(numVals, A2, 22)
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 + A2
        addTimeArray[0, s] =  time.clock() - startTime 
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 * A2
        multTimeArray[0, s] =  time.clock() - startTime 
        
        print("Benchmarking  +/* of csr_matrix")
        A1 = csr_matrix((n, n))       
        A2 = csr_matrix((n, n)) 
        setRandomElements(numVals, A1, 21)
        setRandomElements(numVals, A2, 22)
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 + A2
        addTimeArray[1, s] =  time.clock() - startTime 
        
        startTime = time.clock()
        for i in range(repetitions): 
            B = A1 * A2
        multTimeArray[1, s] =  time.clock() - startTime 
        
        print("Benchmarking  +/* of csarray")
        A1 = csarray((n, n))       
        A2 = csarray((n, n)) 
        setRandomElements(numVals, A1, 21)
        setRandomElements(numVals, A2, 22)
        A1.compress()
        A2.compress()        
        
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
    numMethods = 3
    sumTimeArray = numpy.zeros((numMethods, len(ns)))
    repetitions = 100 
    
    for s in range(len(ns)):
        n = ns[s]
        logging.debug("n="+str(n))

        print("Benchmarking sum of csc_matrix")
        A = csc_matrix((n, n))       
        setRandomElements(numVals, A, 21)

        startTime = time.clock()
        for i in range(repetitions): 
            b = A.sum()
        sumTimeArray[0, s] =  time.clock() - startTime 
        
        print("Benchmarking sum of csr_matrix")
        A = csr_matrix((n, n))       
        setRandomElements(numVals, A, 21)

        startTime = time.clock()
        for i in range(repetitions): 
            b = A.sum()
        sumTimeArray[1, s] =  time.clock() - startTime 
        
        print("Benchmarking sum of csarray")
        A = csarray((n, n))       
        setRandomElements(numVals, A, 21)
        A.compress()

        startTime = time.clock()
        for i in range(repetitions): 
            b = A.sum()
        sumTimeArray[2, s] =  time.clock() - startTime 
        
    return sumTimeArray *  1000.0

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
numpy.set_printoptions(suppress=True)

ns = [100, 500, 1000, 5000, 10000, 50000, 100000]
#ns = [100, 500, 1000, 5000]
methodNames = ["csc_matrix", "csr_matrix", "csarray"]

setTimeArray, getTimeArray = setGetBenchmark(ns)
addTimeArray, multTimeArray = addMultBenchmark(ns)
sumTimeArray = meanSumBenchmark(ns)

plt.figure(0)
for i in range(len(methodNames)): 
    plt.plot(ns, setTimeArray[i, :], label=methodNames[i])
plt.title("set")
plt.xlabel("x")
plt.ylabel("time")
plt.legend()

plt.figure(1)
for i in range(len(methodNames)): 
    plt.plot(ns, getTimeArray[i, :], label=methodNames[i])
plt.title("get")
plt.xlabel("x")
plt.ylabel("time")
plt.legend()

plt.figure(2)
for i in range(len(methodNames)): 
    plt.plot(ns, addTimeArray[i, :], label=methodNames[i])
plt.title("add")
plt.xlabel("x")
plt.ylabel("time")
plt.legend()

plt.figure(3)
for i in range(len(methodNames)): 
    plt.plot(ns, multTimeArray[i, :], label=methodNames[i])
plt.title("mult")
plt.xlabel("x")
plt.ylabel("time")
plt.legend()

plt.figure(4)
for i in range(len(methodNames)): 
    plt.plot(ns, sumTimeArray[i, :], label= methodNames[i])
plt.title("sum")
plt.xlabel("x")
plt.ylabel("time")
plt.legend()

print(setTimeArray)
print(getTimeArray)
print(addTimeArray)
print(multTimeArray)
print(sumTimeArray)

plt.show()

