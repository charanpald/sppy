"""
This is some code to do some meta-programming - take a cython file and 
expand classes written as: 
    cdef template[T] class X: 
where T is a template type. 
"""
import string 
import re 
import logging 
import sys 
import numpy 
import itertools 
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
numpy.set_printoptions(suppress=True)

def expandTemplate(inFileName, outFileName, templateList): 
    if os.path.exists(outFileName) and os.path.getmtime(inFileName) < os.path.getmtime(outFileName): 
        logging.debug("No new changes changes in " + inFileName)
        return     
    
    inFile = open(inFileName, 'r')
    outFile = open(outFileName, 'w')
    
    lines = inFile.readlines()
    inClass = False 
    
    classDefLines = []
    
    for line in lines: 
        if string.find(line, "cdef template") == 0: 
            inClass = True
            words = line.split()
            #print(words)
            className = string.strip(words[-1], ":")
            
            templateParams = words[1:-2] 
            templateParams[0] = string.replace(templateParams[0], "template[", "")
            templateParams[-1] = string.replace(templateParams[-1], "]", "")
            
            #Remove "," from params 
            for i in range(len(templateParams)):
                templateParams[i] = templateParams[i].strip(",")
            
        elif not inClass: 
            outFile.write(line)
        elif inClass and (len(string.split(line, "   ")[0]) == 0 or len(string.strip(line)) == 0): 
            inClass = True
            classDefLines.append(line)       
        else: 
            inClass = False 
    
    inFile.close()     
    logging.debug("Read input file " + inFileName)
    
    for templateTypes in templateList:
        newClassName = className + "_" + string.replace(string.join(templateTypes), " ", "_")
        outFile.write("cdef class " + newClassName +  ":\n")
        logging.debug("Writing class " + newClassName)
        
        #Something like csarray[float, rowMajor]
        templateClassName = className + "["
        for i, templateParam in enumerate(templateParams):
            if i != len(templateTypes)-1:
                templateClassName += str(templateParam) + ", " 
            else: 
                templateClassName += str(templateParam) 
        templateClassName += "]"
        #print(templateClassName, newClassName)
        
        for line in classDefLines:
            outLine = line  
            outLine = string.replace(outLine, templateClassName, newClassName)
            #print(outLine)
            
            for i, templateType in enumerate(templateTypes): 
                outLine = (string.replace(outLine, templateParams[i], templateType))
                
            outFile.write(outLine)
                
    outFile.close() 
    logging.debug("Wrote output file " + outFileName)

def expand_base(workdir='.'):
    typeList = ["signed char", "short", "int", "long", "float", "double"]
    storageList = ["colMajor", "rowMajor"]
    templateList = list(itertools.product(typeList, storageList))
    #templateList = [["float", "colMajor"]]                                                                                                                                              

    inFileName = os.path.join(workdir, "csarray_base.pyx")
    outFileName = os.path.join(workdir, "csarray_sub.pyx")
    expandTemplate(inFileName, outFileName, templateList)

    inFileName = os.path.join(workdir, "csarray_base.pxd")
    outFileName = os.path.join(workdir, "csarray_sub.pxd")
    expandTemplate(inFileName, outFileName, templateList)


    templateList = [["signed char"], ["short"], ["int"], ["long"], ["float"], ["double"]]
    inFileName = os.path.join(workdir, "csarray1d_base.pyx")
    outFileName = os.path.join(workdir, "csarray1d_sub.pyx")
    expandTemplate(inFileName, outFileName, templateList)
    
    inFileName = os.path.join(workdir, "csarray1d_base.pxd")
    outFileName = os.path.join(workdir, "csarray1d_sub.pxd")
    expandTemplate(inFileName, outFileName, templateList)

if __name__ == '__main__':
    expand_base()
