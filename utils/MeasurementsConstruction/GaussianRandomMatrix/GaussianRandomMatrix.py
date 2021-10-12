import numpy as np
import cmath as math


def GaussianRandomMatrix(n, lamda, c, varepsilon):
    m  = int(c*lamda*math.log(n)*math.log(1/varepsilon))
    A = list()
    for i in range(m):
        row = list(np.random.normal(0, 1, n))
        A.append(row)

    return np.array(A)/math.sqrt(m)