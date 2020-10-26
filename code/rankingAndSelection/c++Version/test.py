# ctypes_test.py
import ctypes
import pathlib
from ctypes import *

def c_array(ctype,values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype*len(values))(*values)

if __name__ == "__main__":
    #Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "test.so"
    c_lib = ctypes.CDLL(libname)
    
    A = 2
    M = 6
    Pij = [0.5] * 4
    Pij_array = (c_float * A**2)(*Pij)
    ArrLst = [0.5] * 2
    RhoMtx = [0.5] * 4
    Beta = 0.3
    Tau = 1.0
    C = 3
    Mu = 1.0
    N = 1 
    
    t = c_lib.Test()

    #c_lib.run(Pij_array)
    #print(pyhehe(1,2))