import numpy as np
import cmath as math
from .GaussianRandomMatrix import GaussianRandomMatrix

def RGBGaussianRandomMatrix(n,lamda, c, varepsilon):
    Red = GaussianRandomMatrix(n,lamda, c, varepsilon)
    Green = GaussianRandomMatrix(n,lamda, c, varepsilon)
    Blue = GaussianRandomMatrix(n,lamda, c, varepsilon)

    return (Red, Green, Blue)