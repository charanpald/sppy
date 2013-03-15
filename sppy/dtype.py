import numpy 

class TwoWayDict(dict):
    def __len__(self):
        return dict.__len__(self) / 2

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

#A dict for converting types between the C++ template type and numpy dtype 
dataTypeDict = TwoWayDict() 
dataTypeDict["float"] = numpy.float32
dataTypeDict["double"] = numpy.float64
dataTypeDict["signed char"] = numpy.int8
dataTypeDict["short"] = numpy.int16
dataTypeDict["int"] = numpy.int32
dataTypeDict["long"] = numpy.int64