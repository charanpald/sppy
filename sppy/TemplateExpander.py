"""
This is some code to do some meta-programming - take a cython file and 
expand classes written as: 
    cdef template[T] class X: 
where T is a template type. 
"""
import re 
import logging 
import sys 
import numpy 
import itertools 
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
numpy.set_printoptions(suppress=True)

def expandTemplate(inFileName, outFileName, templateList, force=False): 
    if os.path.exists(outFileName) and os.path.getmtime(inFileName) < os.path.getmtime(outFileName) and not force: 
        logging.debug("No new changes changes in " + inFileName)
        return     
    
    inFile = open(inFileName, 'r')
    outFile = open(outFileName, 'w')
    
    lines = inFile.readlines()
    inClass = False 
    
    classDefLines = []
    
    for line in lines: 
        if line.find("cdef template") == 0: 
            inClass = True
            words = line.split()
            #print(words)
            className = words[-1].strip(":")
            
            templateParams = words[1:-2] 
            templateParams[0] = templateParams[0].replace("template[", "")
            templateParams[-1] = templateParams[-1].replace("]", "")
            
            #Remove "," from params 
            for i in range(len(templateParams)):
                templateParams[i] = templateParams[i].strip(",")
            
        elif not inClass: 
            outFile.write(line)
        elif inClass and (len(line.split("   ")[0]) == 0 or len(line.strip()) == 0): 
            inClass = True
            classDefLines.append(line)       
        else: 
            inClass = False 
    
    inFile.close()     
    logging.debug("Read input file " + inFileName)
    
    
    #Create regular expression for csarray[float,colMajor]  -> csarray_float_colMajor 
    findString = className + "\[" 
    replaceString = className + "_"
    for i in range(len(templateList[0])):
        if i != len(templateList[0])-1:
            findString += "\s*([\w_ ]+)\s*,\s*"
            replaceString += "\\" + str(i+1) + "_"
        else: 
            findString += "\s*([\w_ ]+)\s*\]"
            replaceString += "\\" + str(i+1)    
    
    p = re.compile(findString)
    
    for templateTypes in templateList:
        newClassName = className + "[" + ",".join(templateTypes) + "]"
        newClassName = p.sub(replaceString, newClassName)
        newClassName = newClassName.replace(" ", "_")
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
        
        for line in classDefLines:
            outLine = line  
            
            #Replace e.g. DataType with int 
            for i, templateType in enumerate(templateTypes): 
                outLine = (outLine.replace(templateParams[i], templateType))            

            # csarray[float,colMajor]  -> csarray_float_colMajor 
            reResults = p.search(outLine)

            if reResults != None:  
                outLine = p.sub(replaceString, outLine)
                
                for word in reResults.groups():
                    outLine = outLine.replace(word, word.replace(" ", "_"))

            outFile.write(outLine)
                
    outFile.close() 
    logging.debug("Wrote output file " + outFileName)

def expand_base(workdir='.', force=False):
    typeList = ["signed char", "short", "int", "long", "float", "double"]
    #typeList = ["double"]
    storageList = ["colMajor", "rowMajor"]
    templateList = list(itertools.product(typeList, storageList))
    #templateList = [["float", "colMajor"]]                                                                                                                                              

    inFileName = os.path.join(workdir, "csarray_base.pyx")
    outFileName = os.path.join(workdir, "csarray_sub.pyx")
    expandTemplate(inFileName, outFileName, templateList, force)

    
    inFileName = os.path.join(workdir, "csarray_base.pxd")
    outFileName = os.path.join(workdir, "csarray_sub.pxd")
    expandTemplate(inFileName, outFileName, templateList, force)


    templateList = [["signed char"], ["short"], ["int"], ["long"], ["float"], ["double"]]
    inFileName = os.path.join(workdir, "csarray1d_base.pyx")
    outFileName = os.path.join(workdir, "csarray1d_sub.pyx")
    expandTemplate(inFileName, outFileName, templateList, force)
    
    inFileName = os.path.join(workdir, "csarray1d_base.pxd")
    outFileName = os.path.join(workdir, "csarray1d_sub.pxd")
    expandTemplate(inFileName, outFileName, templateList, force)
    
    
if __name__ == '__main__':
    expand_base(force=False)
