"""
This is some code to do some meta-programming - take a cython file and 
expand classes written as: 
    cdef template[T] class X: 
where T is a template type. 


"""
import string 
import re 

templateList = [["int"], ["long"], ["float"], ["double"]]

inFileName = "csarray_base.pyx"
outFileName = "csarray_sub.pyx"

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
        
        
    elif not inClass: 
        outFile.write(line)
    elif inClass and (len(string.split(line, "   ")[0]) == 0 or len(string.strip(line)) == 0): 
        inClass = True
        classDefLines.append(line)       
    else: 
        inClass = False 

for templateTypes in templateList:
    newClassName = className + "_" + string.join(templateTypes)
    outFile.write("cdef class " + newClassName +  ":\n")
    
    for line in classDefLines:
        for i, templateType in enumerate(templateTypes): 
            outLine = string.replace(line, className + "[" + templateParams[i] +  "]", newClassName)
            outLine = (string.replace(outLine, templateParams[i], templateType))  
        outFile.write(outLine)
            

inFile.close() 
outFile.close() 
print("Done")