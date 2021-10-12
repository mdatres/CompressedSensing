import numpy as np
from .BinaryRandomMatrix import BinaryRandomMatrix

def RGBBinaryRandomMatrix(n, p, lamda, c, varepsilon):
    Red = BinaryRandomMatrix(n,p,lamda, c, varepsilon)
    Green = BinaryRandomMatrix(n,p,lamda, c, varepsilon)
    Blue = BinaryRandomMatrix(n,p,lamda, c, varepsilon)

    return (Red, Green, Blue)