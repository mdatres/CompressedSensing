import cvxpy as cvx

def optimizerLI(n, mat, b, complex=True, alg="ECOS"):
    signal = cvx.Variable(n, complex=True)
    objective = cvx.Minimize(cvx.norm(signal, 1))
    constraints = [mat*signal == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    
    return signal.value