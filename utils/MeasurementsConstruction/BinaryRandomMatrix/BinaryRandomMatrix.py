import numpy as np
import cmath as math


def BinaryRandomMatrix(n, p,  lamda, c, varepsilon):
    m  = int(c*lamda*math.log(n)*math.log(1/varepsilon))
    A = list()
    for i in range(m):
        row = list(np.random.binomial(1, p, n))
        row = [i if i != 0 else -1 for i in row]
        A.append(row)

    return np.array(A)/math.sqrt(m)