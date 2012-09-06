from sppy.csarray import csarray
from sppy.util import diag, eye, rand, ones, zeros
from sppy.version import __version__

def test():
    """
    A function which uses the unittest library to find all tests within sparray (those files
    matching "*Test.py"), and run those tests. 
    """
    try:
        import traceback
        import sys
        import os
        import unittest
        import logging

        logging.disable(logging.WARNING)
        sourceDir = os.path.dirname(__file__) + os.sep
        print("Running tests from " + sourceDir)

        overallTestSuite = unittest.TestSuite()
        overallTestSuite.addTest(unittest.defaultTestLoader.discover(sourceDir, pattern='*Test.py', top_level_dir=sourceDir))
        unittest.TextTestRunner(verbosity=1).run(overallTestSuite)
    except ImportError as error:
        traceback.print_exc(file=sys.stdout)


