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


def findParams(lines): 
    """
    Figure out the template parameters from e.g. cdef template[DataType, StorageType] class csarray:
    Here the parameters are DataType and StorageType
    Also write the resulting bits not in the class to the output file 
    """
    inClass = False 
    classDefLines = []
    nonClassDefLines = []

    for line in lines: 
        match = re.match(r"cdef[\s]template\[([\w,\s]+)\]\s?class\s?(?P<class>[\w]+)", line)        
        
        if match != None: 
            inClass = True
            groups = list(match.groups())
            words = groups[0].replace(" ", "").split(",")

            #E.g. csarray
            className = groups[1]

            #E.g. ['DataType', 'StorageType', 'CompType']
            templateParams = words
            classDefLines.append(line) 
        elif not inClass: 
            nonClassDefLines.append(line)
        elif inClass and (len(line.split("   ")[0]) == 0 or len(line.strip()) == 0): 
            inClass = True
            classDefLines.append(line)       
        else: 
            inClass = False 
            
    return templateParams, className, classDefLines, nonClassDefLines

def replaceTemplateParams(className, classDefLines, **kwargs):
    """
    Take the first line and replace to get e.g. cdef class csarray[char, rowMajor]:
    and then do the same thing with subsequent lines. 
    """    
    for key, value in kwargs.iteritems(): 
        classDefLines[0] = classDefLines[0].replace(key, value)  
    
    classDefLines[0] = re.sub(r"\s?class\s+[\w]+", "",classDefLines[0])
    classDefLines[0] = classDefLines[0].replace("template", "class " + className)
 
    #Do the same thing for following lines 
    for key, value in kwargs.iteritems(): 
        for i, line in enumerate(classDefLines):
            if i!= 0: 
                #classDefLines[i] = line.replace(key, value)  
                classDefLines[i] = re.sub(r'\b' + key + r'\b', value, line)
                
def replaceTemplateNotation(className, classDefLines): 
    """
    Go through and replace csarray[float,colMajor]  -> csarray_float_colMajor     
    """
    regExp = className + r"\[([\w,\s]+)\]"
    
    for i, line in enumerate(classDefLines): 
        match = re.search(regExp, line)
        
        if match != None:
            params = match.group(1)
            suffix = re.sub(r",?\s", "_", params)
            replaceStr = className + "_" + suffix
            classDefLines[i] = re.sub(regExp, replaceStr, line)

def generateKwargs(paramDict): 
    args = []    
    
    for key, value in paramDict.iteritems(): 
        args.append(value["values"])
        
    paramList = list(itertools.product(*args))
    
    #Now construct kwargs 
    kwargsList = [] 
    for params in paramList: 
        kwargs = {}
        for i, key in enumerate(paramDict.keys()):
            kwargs[key] = params[i]
            
            if paramDict[key]["type"] == bool: 
                values = paramDict[key]["values"]
                if params[i] == values[0]: 
                    kwargs["Not" + key] = values[1]
                else: 
                    kwargs["Not" + key] = values[0]
            
        kwargsList.append(kwargs)
            
    return kwargsList
            
def expandTemplate(inFileName, outFileName, paramDict, force=False): 
    if os.path.exists(outFileName) and os.path.getmtime(inFileName) < os.path.getmtime(outFileName) and not force: 
        logging.debug("No new changes changes in " + inFileName)
        return     
    
    inFile = open(inFileName, 'r')    
    lines = inFile.readlines()
    inFile.close() 
    logging.debug("Read input file " + inFileName)

    templateParams, className, classDefLines, nonClassDefLines = findParams(lines)
    
    #Write the output 
    outFile = open(outFileName, 'w')
    for line in nonClassDefLines: 
        outFile.write(line)    
    
    kwargsList = generateKwargs(paramDict)       
        
    for kwargs in kwargsList: 
        classDefLinesCopy = classDefLines[:]        
        
        replaceTemplateParams(className, classDefLinesCopy, **kwargs)
        replaceTemplateNotation(className, classDefLinesCopy)
        
        for line in classDefLinesCopy: 
            outFile.write(line)
    
    outFile.close() 
    logging.debug("Wrote output file " + outFileName)

def expand_base(workdir='.', force=False):
    paramDict = {}
    paramDict["DataType"] = {"type": list, "values": ["signed char", "short", "int", "long", "float", "double"]} 
    paramDict["StorageType"] = {"type": bool, "values": ["colMajor", "rowMajor"]}
                                                                                                                                              
    inFileName = os.path.join(workdir, "csarray_base.pyx")
    outFileName = os.path.join(workdir, "csarray_sub.pyx")
    expandTemplate(inFileName, outFileName, paramDict, force)

    inFileName = os.path.join(workdir, "csarray_base.pxd")
    outFileName = os.path.join(workdir, "csarray_sub.pxd")
    expandTemplate(inFileName, outFileName, paramDict, force)

    paramDict = {}
    paramDict["DataType"] = {"type": list, "values": ["signed char", "short", "int", "long", "float", "double"]} 
    inFileName = os.path.join(workdir, "csarray1d_base.pyx")
    outFileName = os.path.join(workdir, "csarray1d_sub.pyx")
    expandTemplate(inFileName, outFileName, paramDict, force)
    
    inFileName = os.path.join(workdir, "csarray1d_base.pxd")
    outFileName = os.path.join(workdir, "csarray1d_sub.pxd")
    expandTemplate(inFileName, outFileName, paramDict, force)
    
if __name__ == '__main__':
    expand_base(force=True)
