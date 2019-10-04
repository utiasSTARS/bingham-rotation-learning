# QUaternion-based Sparse Sum Of Squares relAxation for Robust alignment
from ncpol2sdpa import generate_variables, SdpRelaxation

from helpers import *


def make_quassosar_equalities(X, redundancies=None):
    N = int(len(X)/4 - 1)
    equalities= [np.dot(X[0:4], np.transpose(X[0:4])) - 1]
    for i in range(1, N+1):
        for j in range(0, 4):
            for k in range(0, 4):
        # Ai = np.dot(np.transpose(X[1:4]), X[1:4]) - np.dot(np.transpose(X[i*4:(i+1)*4]), X[i*4:(i+1)*4])
                equalities.append(X[j]*X[k] - X[i*4+j]*X[i*4+k])
    if redundancies == 'linear':
        pass

    return equalities


def make_quassosar_objective(X, q1, q2, c_bar_2=1, sigma_2_i=1):
    Q = build_cost_function_matrix(q1, q2, c_bar_2, sigma_2_i)
    return np.dot(X, np.dot(Q, np.transpose(X)))


def make_quassosar_sdp(q1, q2, c_bar_2, level=1, redundancies=None, sparsity=True):
    N = q1.shape[0]
    X = generate_variables('x', 4*(N+1))
    equalities = make_quassosar_equalities(X, redundancies=redundancies)
    obj = make_quassosar_objective(X, q1, q2, c_bar_2=c_bar_2)
    sdp = SdpRelaxation(X)
    if sparsity:
        sdp.variables = custom_sparsity(sdp.variables, N)
    # Not sure chordal_extension=True makes a difference yet
    # TODO: FIX SPARSITY
    sdp.get_relaxation(level, objective=obj, equalities=equalities)#, chordal_extension=True)
    return sdp, X


def extract_sparse_pop_solution(sdp, X):
    pass


def custom_sparsity(variables, N):
    return [variables[0:4]+variables[4*i:4*(i+1)] for i in range(1, N+1)]


if __name__=='__main__':
    np.random.seed(8675309)
    level = 1
    N = 30
    N_out = 5
    sigma = 0.01
    c_bar_2 = (3*sigma+1e-4)**2
    q1, q2 = make_random_instance(N, N_out, sigma=sigma)
    sdp, X = make_quassosar_sdp(q1, q2, c_bar_2, level=level, sparsity=False)
    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual)
    print(sdp.find_solution_ranks())
    print(sdp.solution_time)

    # Extract solution
