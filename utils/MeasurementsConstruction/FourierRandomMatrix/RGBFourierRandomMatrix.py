import numpy as np 
import cmath as math
from .FourierRandomMatrix import create_measurements, DFT_matrix


def RGBFourierRandomMatrix(n, lamda, c, vaepsilon): 
    DFT = DFT_matrix(n)
    m  = int(c*lamda*math.log(n)*math.log(1/vaepsilon))
    A = create_measurements(F,lamda, m, n )
    FourRed = create_measurements(DFT, lamda, m, n)
    FourGreen = create_measurements(DFT, lamda, m, n)
    FourBlue = create_measurements(DFT, lamda, m, n)
    return (FourRed, FourGreen, FourBlue)