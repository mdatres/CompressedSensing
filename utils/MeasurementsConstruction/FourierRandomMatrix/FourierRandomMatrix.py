import numpy as np 
import cmath as math

def create_measurements(F, lamda, m, n): 
    Omega = np.arange(n)
    indexes = np.random.binomial(1, m/n, n)
    indexes = np.multiply(Omega, indexes)
    indexes = [i for i in indexes if i != 0]
    A = F[indexes,:]
    print(A.shape)
    A = np.array(A)
    return (1/math.sqrt(m))*A



def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * math.pi * 1J / N )
    W = np.power( omega, i * j ) / math.sqrt(N)
    return W

def FourierRandomMatrix(n, lamda, c, varepsilon): 
    DFT = DFT_matrix(n)
    m  = int(c*lamda*math.log(n)*math.log(1/varepsilon))
    print("The number of measurements is:   " + str(m))
    A = create_measurements(F,lamda, m, n )
    return A